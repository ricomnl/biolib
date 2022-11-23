import collections
import multiprocessing
import random
from typing import Optional

from anndata import AnnData
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import pandas as pd
import scanpy as sc
from scanpy import logging as logg
from scipy import sparse


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


def lognorm(adata, base=None, target_sum=None, save_counts=True, copy=False):
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
    copy : bool, optional
        Whether to return a copy of the AnnData object, by default False

    Returns
    -------
    AnnData
        Log normalized AnnData object.
    """
    adata = adata.copy() if copy else adata
    if save_counts:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata, base=base)
    if save_counts:
        adata.layers['lognorm'] = adata.X
    return adata if copy else None


def aggregate_groups(
    adata,
    groupby,
    layer='counts',
    include_barcodes=False,
    agg='sum'
):
    """Aggregate counts by group.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    groupby : str
        Name of observation to group by.
    layer : str, optional   
        AnnData layer to aggregate, by default 'counts'
    include_barcodes : bool, optional
        Whether to include barcodes in the output, by default False
    agg : str, optional
        Aggregation method, by default 'sum'
    
    Returns
    -------
    AnnData
        Aggregated counts.
    """
    agg_methods = {
        "sum": np.sum,
        "mean": np.mean,
    }
    
    obs_elems = []
    mtx_elems = []
    for _,group in adata.obs.groupby(groupby):
        if layer == 'counts' and layer not in adata.layers:
            mtx_elems.append(np.asarray(agg_methods[agg](adata[group.index, :].X, axis=0)))
        else:
            mtx_elems.append(np.asarray(agg_methods[agg](adata[group.index, :].layers[layer], axis=0)))
        meta = group.iloc[0].to_dict()
        meta['cells_per_metacell'] = group.shape[0]
        if include_barcodes:
            meta['barcodes'] = ','.join(group.index.to_list())
        obs_elems.append(meta)
    mtx = np.concatenate(mtx_elems)
    obs_df = pd.DataFrame(obs_elems)
    adata_meta = sc.AnnData(mtx, obs=obs_df, var=adata.var)
    return adata_meta


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
    # TODO: unify with aggregate_groups
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
    include_barcodes=False,
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
    include_barcodes : bool, optional
        Whether to include barcodes in obs, by default False
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
            include_barcodes=include_barcodes,
            layer=layer,
        ))
    return sc.concat(ray.get(futures))


def transform_age(arr, adult_age=20):
    """Horvath's transform age function.

    Parameters
    ----------
    arr : array
        Ages to transform
    adult_age : int, optional
        Adult age, by default 20

    Returns
    -------
    array
        Transformed ages
    """
    return np.where(
        arr<=adult_age,
        np.log2(arr+1)-np.log2(adult_age+1),
        (arr-adult_age)/(adult_age+1),
    )


def rev_transform_age(arr, adult_age=20):
    """Horvath's reverse transform age function.

    Parameters
    ----------
    arr : array
        Ages to reverse transform
    adult_age : int, optional
        Adult age, by default 20

    Returns
    -------
    array
        Reversed transformed ages
    """
    return np.where(
        arr<0,
        (1+adult_age)*(2**arr)-1,
        (1+adult_age)*arr+adult_age,
    )


def walktrap(
    adata: AnnData,
    gamma: float = None,
    *,
    key_added: str = 'walktrap',
    adjacency: Optional[sparse.spmatrix] = None,
    directed: bool = True,
    use_weights: bool = True,
    steps: int = 4,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
    **partition_kwargs,
) -> Optional[AnnData]:
    """\
    Computing communities in large networks using random walks, https://arxiv.org/abs/physics/0512106.

    This requires having ran :func:`~scanpy.pp.neighbors` first.

    Parameters
    ----------
    adata
        The annotated data matrix.
    gamma
        The graining level of data (proportion of number of single cells 
        in the initial dataset to the number of metacells in the final dataset).
        By default None (uses optimal cutoff).
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
    directed
        Whether to treat the graph as directed or undirected.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    steps   	
        Integer constant, the length of the random walks. 
        Typically, good results are obtained with values between 
        3-8 with 4-5 being a reasonable default.
    neighbors_key
        Use neighbors connectivities as adjacency.
        If not specified, leiden looks .obsp['connectivities'] for connectivities
        (default storage place for pp.neighbors).
        If specified, leiden looks
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    obsp
        Use .obsp[obsp] as adjacency. You can't specify both
        `obsp` and `neighbors_key` at the same time.
    copy
        Whether to copy `adata` or modify it inplace.
    **partition_kwargs
        Any further arguments to pass to `~leidenalg.find_partition`
        (which in turn passes arguments to the `partition_type`).

    Returns
    -------
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    `adata.uns['walktrap']['params']`
        A dict with the values for the parameters `gamma`, `steps`,
        and `n_clusters`.
    """    
    partition_kwargs = dict(partition_kwargs)

    start = logg.info('running Walktrap clustering')
    adata = adata.copy() if copy else adata
    
    # number of clusters if gamma is not None
    n_clusters = round(adata.shape[0]/gamma) if gamma else None

    # are we clustering a user-provided graph or the default AnnData one?
    if adjacency is None:
        adjacency = sc._utils._choose_graph(adata, obsp, neighbors_key)

    # convert it to igraph
    g = sc._utils.get_igraph_from_adjacency(adjacency, directed=directed)
    if use_weights:
        partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    partition_kwargs['steps'] = steps
    
    # clustering proper
    part = g.community_walktrap(**partition_kwargs)
    # store output into adata.obs
    groups = np.array(part.as_clustering(n=n_clusters).membership)
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(map(str, np.unique(groups))),
    )
    # store information on the clustering parameters
    adata.uns['walktrap'] = {}
    adata.uns['walktrap']['params'] = dict(
        gamma=gamma,
        steps=steps,
        n_clusters=n_clusters,
    )
    logg.info(
        '    finished',
        time=start,
        deep=(
            f'found {len(np.unique(groups))} clusters and added\n'
            f'    {key_added!r}, the cluster labels (adata.obs, categorical)'
        ),
    )
    return adata if copy else None


def metacell_pca(
    adata,
    n_comps=10,
    mode='highly_variable',
    n_top_genes=3000,
    copy=False
):
    """Perform PCA for metacell construction.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_comps : int, optional
        Number of components to compute, by default 10
    mode : ['highly_variable', 'freeman_tukey'], optional
        How to select genes for PCA, by default 'highly_variable'
    n_top_genes : int, optional
        Number of top genes to use, by default 3000
    copy : bool, optional
        Whether to copy adata, by default False
    
    Returns
    -------
    AnnData
        Annotated data matrix.
    """
    adata = adata.copy() if copy else adata
    if mode == 'highly_variable':
        lognorm(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=True)
    elif mode == 'freeman_tukey':
        # median-normalize
        num_transcripts = np.asarray(np.sum(adata.X, axis=1))[:, 0]
        X = ((np.median(num_transcripts) / num_transcripts).reshape(-1,1)) * adata.X.toarray()
        # freeman-tukey transform
        adata.X = np.sqrt(X) + np.sqrt(X+1)
        sc.tl.pca(adata, n_comps=n_comps)
    else:
        adata = lognorm(adata)
        sc.tl.pca(adata, n_comps=n_comps)
    return adata if copy else None


def gen_metacells(
    adata, 
    groupby=['donor'], 
    n_neighbors=5,
    gamma=20,
    use_rep='X_pca'
):
    """Generate metacells from adata.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : list, optional
        Groupby list, by default ['donor']
    n_neighbors : int, optional
        Number of neighbors, by default 5
    gamma : int, optional
        The graining level of data (proportion of number of single cells 
        in the initial dataset to the number of metacells in the final dataset).
        By default None (uses optimal cutoff). By default 20
    use_rep : str, optional
        Representation to use, by default 'X_pca'
    
    Returns
    -------
    AnnData
        Annotated data matrix.
    """
    adatas = []
    for _,group in adata.obs.groupby(groupby):
        adata_sub = adata[group.index]
        sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, use_rep=use_rep)
        walktrap(adata_sub, gamma=gamma)
        adatas.append(aggregate_groups(adata_sub, groupby='walktrap'))    
    return sc.concat(adatas, merge='same', join='outer')


def gen_metacells_parallel(
    adata,
    parallel_group_obs='donor',
    groupby=['donor'], 
    n_neighbors=5, 
    gamma=20,
    use_rep='X_pca',
    n_cores=64,
):
    """Generate metacells from adata in parallel.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    parallel_group_obs : str, optional
        Observation to use for parallelization, by default 'donor'
    groupby : list, optional
        Groupby list, by default ['donor']
    n_neighbors : int, optional
        Number of neighbors, by default 5
    gamma : int, optional
        The graining level of data (proportion of number of single cells 
        in the initial dataset to the number of metacells in the final dataset).
        By default None (uses optimal cutoff). By default 20
    use_rep : str, optional
        Representation to use, by default 'X_pca'
    n_cores : int, optional
        Number of cores to use, by default 64
    
    Returns
    -------
    AnnData
        Annotated data matrix.
    """
    import ray
    ray.init(ignore_reinit_error=True)

    gen_metacells_remote = ray.remote(gen_metacells)

    # get unique donors and separate list of obs into n_cores disjoint sets
    unique_obs = list(adata.obs[parallel_group_obs].unique())
    donor_chunks = filter(lambda arr: arr.shape[0]>0, np.array_split(unique_obs, n_cores))

    # subset your adata into smaller adatas with only the obs in chunk
    adata_subsets = []
    for chunk in donor_chunks:
        adata_subsets.append(adata[adata.obs[parallel_group_obs].isin(chunk)].copy())

    futures = []
    for subset in adata_subsets:
        futures.append(gen_metacells_remote.remote(
            subset,
            groupby=groupby, 
            n_neighbors=n_neighbors, 
            gamma=gamma,
            use_rep=use_rep,
        ))
    return sc.concat(ray.get(futures))