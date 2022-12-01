import numpy as np
from sklearn import model_selection
import scipy
import torch
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
        groups, 
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
            self.train_subset = self
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
        else:
            raise ValueError('Invalid type')
        
        X_train = self.X_data[self.train_subset.indices]
        y_train = self.y_data[self.train_subset.indices]
        groups_train = self.groups[self.train_subset.indices]
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