import collections
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pca_obs_corr(adata, obs_names, pca_key='X_pca', plot=False):
    """Measure correlation between principal components and each of the passed `obs_names`.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    obs_names : list
        Names of observations to correlate with principal components.
    pca_key : str, optional
        AnnData PCA key, by default 'X_pca'
    plot : bool, optional
        Whether to plot the correlation, by default False

    Returns
    -------
    dict
        Dictionary of correlation values for each observation.
    """    """"""
    obs_corr = collections.defaultdict(list)
    for obs in obs_names:
        for i in range(adata.obsm[pca_key].shape[1]):
            obs_corr[obs].append(
                np.corrcoef(adata.obs[obs], adata.obsm[pca_key][:, i])[1,0]
            )
    if plot:
        for obs in obs_names:
            plt.scatter(np.arange(len(obs_corr[obs])), obs_corr[obs])
            plt.title(obs)
            plt.xlabel(obs)
            plt.ylabel('corr')
            plt.show()
    return obs_corr


def obs_to_cat(adata, obs_name):
    """Convert observation to categorical.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object.
    obs_name : str
        Name of observation to convert to categorical.
    
    Returns
    -------
    AnnData
        AnnData converted obs column.
    """
    adata.obs[obs_name] = adata.obs[obs_name].astype('category')
    mapper = {k:i for i,k in enumerate(adata.obs[obs_name].unique())}
    return adata.obs[obs_name].replace(mapper)


def subset_donors(adata, group_by, donor_id, keep_n=3, plot=False):
    """Subset donors to keep only `keep_n` donors.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    group_by : str
        Name of observation to group by.
    donor_id : str
        Name of observation to use as donor id.
    keep_n : int, optional
        Number of donors to keep, by default 3
    plot : bool, optional
        Whether to plot the number of cells per donor, by default False

    Returns
    -------
    list
        List of donors to keep.
    """
    ct_counts = adata.obs.groupby(group_by)[donor_id].unique()
    keep_donor_ids = [cc for c in ct_counts for cc in random.choices(c, k=keep_n)]
    if plot:
        plt.bar(ct_counts.index, [len(v) for v in ct_counts.values])
        plt.axhline(keep_n, c='r')
        plt.show()
    return keep_donor_ids


def bin_age(adata, max_age=110, step_size=5, return_labels=False):
    """Bin age into bins of `step_size` years.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    max_age : int, optional
        Maximum age to bin, by default 110
    step_size : int, optional
        Size of bins, by default 5
    return_labels : bool, optional
        Whether to return bin labels, by default False

    Returns
    -------
    AnnData
        AnnData object with binned age.
    list
        List of bin labels.
    """
    bins = np.arange(0, max_age, step_size)
    labels = [f'{b}-{b+step_size}' for b in bins[:-1]]
    adata.obs['bin_age'] = pd.cut(adata.obs['age'], bins=bins, labels=labels)
    if return_labels:
        return adata, labels
    return adata