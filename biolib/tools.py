import collections
import multiprocessing
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


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


def lognorm(adata, base=None, target_sum=None, save_counts=True):
    """Log normalize AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object.
    base : float, optional
        Base to use for log normalization, by default None
    target_sum : float, optional
        Target sum to normalize to, by default None
    save_counts : bool, optional
        Whether to save counts to `counts` layer, by default True

    Returns
    -------
    AnnData
        Log normalized AnnData object.
    """
    if save_counts:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata, base=base)
    if save_counts:
        adata.layers['lognorm'] = adata.X
    return adata


def bootstrap_adata(
    adata, 
    n_samples=100,
    sample_size=15,
    groupby=['donor', 'cell_type'], 
    layer='counts',
    include_barcodes=False,
):
    """Bootstrap AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    n_samples : int, optional
        Number of samples to take, by default 100
    sample_size : int, optional
        Size of samples to take, by default 15
    groupby : list, optional
        List of observations to group by, by default ['donor', 'cell_type']
    layer : str, optional
        AnnData layer to use, by default 'counts'
    include_barcodes : bool, optional
        Whether to include barcodes in obs, by default False
    
    Returns
    -------
    adata : AnnData
        AnnData object with bootstrapped samples.
    """
    obs_elems = []
    mtx_elems = []
    for _,group in adata.obs.groupby(groupby):
        for _ in range(n_samples):
            if sample_size >= group.shape[0]:
                g = group.sample(n=sample_size, replace=True)
            else:
                g = group.sample(n=sample_size)
            if layer == 'counts' and layer not in adata.layers:
                mtx_elems.append(np.asarray(adata[g.index, :].X.sum(axis=0)))
            else:
                mtx_elems.append(np.asarray(adata[g.index, :].layers[layer].sum(axis=0)))
            meta = g.iloc[0].to_dict()
            if include_barcodes:
                meta['barcodes'] = ','.join(g.index.to_list())
            obs_elems.append(meta)
    mtx = np.concatenate(mtx_elems)
    obs_df = pd.DataFrame(obs_elems)
    adata_boot = sc.AnnData(mtx, obs=obs_df, var=adata.var)
    adata_boot.layers[layer] = adata_boot.X
    return adata_boot


def bootstrap_adata_parallel(
    adata,
    n_samples=100,
    sample_size=15,
    parallel_group_obs='donor',
    groupby=['donor', 'cell_type'],
    layer='counts',
    n_cores=multiprocessing.cpu_count(),
):
    """Bootstrap AnnData object in parallel.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    n_samples : int, optional
        Number of samples to take, by default 100
    sample_size : int, optional
        Size of samples to take, by default 15
    parallel_group_obs : str, optional
        Observation to group by for parallelization, by default 'donor'
    groupby : list, optional
        List of observations to group by, by default ['donor', 'cell_type']
    layer : str, optional
        AnnData layer to use, by default 'counts'
    n_cores : int, optional
        Number of cores to use, by default multiprocessing.cpu_count()
    
    Returns
    -------
    adata : AnnData
        AnnData object with bootstrapped samples.
    """
    import ray
    ray.init(ignore_reinit_error=True)

    bootstrap_adata_remote = ray.remote(bootstrap_adata)

    # get unique donors and separate list of obs into n_cores disjoint sets
    unique_obs = list(adata.obs[parallel_group_obs].unique())
    donor_chunks = filter(lambda arr: arr.shape[0]>0, np.array_split(unique_obs, n_cores))
    
    # subset your adata into smaller adatas with only the obs in chunk
    adata_subsets = []
    for chunk in donor_chunks:
        adata_subsets.append(adata[adata.obs[parallel_group_obs].isin(chunk)].copy())

    futures = []
    for subset in adata_subsets:
        futures.append(bootstrap_adata_remote.remote(
            subset,
            n_samples=n_samples, 
            sample_size=sample_size, 
            groupby=groupby,
            layer=layer,
        ))
    return sc.concat(ray.get(futures))
