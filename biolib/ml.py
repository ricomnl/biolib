from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
from sklearn import model_selection
from sklearn import metrics
import scipy
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


def get_wr_sampler(y):
    """Get weighted random sampler for imbalanced data."""
    y = y.to(torch.int32)
    counts = np.bincount(y)
    labels_weights = 1. / counts
    weights = labels_weights[y]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights))


class SCDataset(Dataset):
    def __init__(
        self, 
        X_data, 
        y_data, 
        groups=None,
        test_split=0.1,
        batch_size=256,
        pin_memory=True,
        num_workers=1,
    ):
        self.X_data = X_data
        self.densify = False
        if scipy.sparse.issparse(X_data):
            self.densify = True
        self.y_data = torch.from_numpy(y_data).to(torch.float32)
        self.groups = groups
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.test_split = test_split

        self.setup_subsets(test_split=test_split)
        
    def __getitem__(self, index):
        y = self.y_data[index]
        if self.densify:
            X = self.X_data[index].toarray().squeeze()
        else:
            X = self.X_data[index].squeeze()
        return X, y
        
    def __len__ (self):
        return self.X_data.shape[0]

    def setup_subsets(self, test_split=0.1, random_state=42):
        if test_split is None:
            self.train_subset = Subset(self, np.arange(len(self)))
            return
        test_splitter = model_selection.GroupShuffleSplit(n_splits=1, train_size=1-test_split, random_state=random_state)

        trainval_index, test_index = next(test_splitter.split(X=None, y=None, groups=self.groups))

        self.train_subset = Subset(self, trainval_index)
        self.test_subset = Subset(self, test_index)

    def train_dataloaders(self):
        y_train = self.y_data[self.train_subset.indices]
        train_wr_sampler = get_wr_sampler(y=y_train)
        return DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            sampler=train_wr_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def train_val_dataloaders(
        self,
        n_splits=3,
        val_split=0.1,
        shuffle=True,
        type='group_shuffle',
        random_state=42,
    ):
        if type == 'group_shuffle':
            cv = model_selection.GroupShuffleSplit(n_splits=n_splits, train_size=1-val_split, random_state=random_state)
        elif type == 'stratified_group_kfold':
            cv = model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif type == 'group_kfold':
            cv = model_selection.GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif type == 'stratified_kfold':
            cv = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif type == 'kfold':
            cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            raise ValueError('Invalid type')
        
        X_train = self.X_data[self.train_subset.indices]
        y_train = self.y_data[self.train_subset.indices]
        if 'group' in type:
            assert self.groups is not None, 'Groups must be provided for group splits'
            groups_train = self.groups[self.train_subset.indices]
        else:
            groups_train = None
        for _, (train_index, val_index) in enumerate(
            cv.split(X_train, y_train, groups_train)
        ):
            train = Subset(self.train_subset, train_index)
            val = Subset(self.train_subset, val_index)

            train_wr_sampler = get_wr_sampler(y=y_train[train_index])
            val_wr_sampler = get_wr_sampler(y=y_train[val_index])

            train_dl = DataLoader(
                train,
                batch_size=self.batch_size,
                sampler=train_wr_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            val_dl = DataLoader(
                val,
                batch_size=self.batch_size,
                sampler=val_wr_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )

            yield train_dl, val_dl

    def test_dataloaders(self):
        return DataLoader(
            self.test_subset,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class LightningModule(pl.LightningModule):
    def __init__(
        self, 
        model,
        learning_rate=1e-4,
        weight_decay=0,
        optimizer_cls=torch.optim.Adam,
        lr_scheduler_cls=None,
        criterion=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
            
        self.lr_scheduler_cls = lr_scheduler_cls
        self.optimizer_cls = optimizer_cls
        
        self.model = model
        self.criterion = criterion
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.lr_scheduler_cls is not None:
            scheduler = self.lr_scheduler_cls(optimizer)
            return [optimizer], [scheduler]
        return [optimizer]

    @abstractmethod
    def _calculate_loss(self, batch, mode="train"):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class LightningClassifier(LightningModule):

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=0,
        optimizer=None,
        lr_scheduler=None,
        criterion=None,
        **kwargs
    ):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        super().__init__(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            **kwargs
        )
        self.save_hyperparameters()

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        y = y.argmax(dim=-1)
        acc = (y == y_pred).float().mean()
        f1 = metrics.f1_score(y.detach().cpu(), y_pred.detach().cpu(), average='macro')

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        self.log(f'{mode}_f1_macro', f1)
        return loss


class LightningRegressor(LightningModule):

    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=0,
        optimizer=None,
        lr_scheduler=None,
        criterion=None,
        **kwargs
    ):
        if criterion is None:
            criterion = nn.MSELoss()
        super().__init__(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            **kwargs
        )
        self.save_hyperparameters()

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y.detach().cpu(), y_pred.detach().cpu()))
        mae = metrics.mean_absolute_error(y.detach().cpu(), y_pred.detach().cpu())
        pearsonr = scipy.stats.pearsonr(y.detach().cpu(), y_pred.detach().cpu())[0]
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_rmse', rmse)
        self.log(f'{mode}_mae', mae)
        self.log(f'{mode}_pearsonr', pearsonr)
        return loss


class Classifier(nn.Module):
    def __init__(
        self, 
        n_in,
        n_out=2,
        hidden=[128],
        dropout_rate=0.1,
        logits=False,
        use_batch_norm=True,
        bias=False,
        activation_fn="relu",
    ):
        super(Classifier, self).__init__()

        self.activation_fn = activation_fn
        if activation_fn == "relu":
            self.activation_fn_ = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation_fn_ = nn.LeakyReLU()
        elif activation_fn == "tanh":
            self.activation_fn_ = nn.Tanh()
        elif activation_fn == "sigmoid":
            self.activation_fn_ = nn.Sigmoid()
        else:
            self.activation_fn_ = None

        hidden = [n_in]+hidden
        self.fc_layers = nn.Sequential(
            OrderedDict([
                (
                    f"Layer {i}",
                    nn.Sequential(
                        nn.Linear(n_in, n_out, bias=bias),
                        nn.BatchNorm1d(n_out)
                        if use_batch_norm
                        else None,
                        self.activation_fn_,
                        nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                    ),
                )
                for i, (n_in, n_out) in enumerate(
                    zip(hidden, hidden[1:])
                )
            ])
        )
        out_layers = [
            nn.Linear(hidden[-1], n_out)
        ]
        if not logits:
            out_layers.append(nn.Softmax(dim=-1))
        self.out = nn.Sequential(*out_layers)

    def forward(self, x):
        for _, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return self.out(x)