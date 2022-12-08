import matplotlib.pyplot as plt
import numpy as np
import scanpy.external as sce

from plotting import cmo_count_density


def is_outlier(adata, metric: str, nmads: int):
    """Determine outliers based on a metric.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    metric : str
        Name of the metric to use.
    nmads : int
        Number of median absolute deviations to use as cutoff.
    
    Returns
    -------
    outlier : np.ndarray
        Boolean array indicating whether a cell is an outlier.
    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * M.mad()) | (
        np.median(M) + nmads * M.mad() < M
    )
    return outlier


def merge_hashsolo(
    adata_cdna, 
    adata_cmo, 
    cmo_map, 
    count_density_widths=20, 
    n_barcodes=1,
    singlet_doublet_priors=None,
    singlets_hypothesis_prob_cutoff=0.9,
):
    """Merge CMO and CDNA data and run hashsolo.
    
    Parameters
    ----------
    adata_cdna : anndata.AnnData
        AnnData object containing the CDNA data.
    adata_cmo : anndata.AnnData
        AnnData object containing the CMO data.
    cmo_map : dict
        Mapping of CMO barcodes to CDNA barcodes.   
    count_density_widths : int, optional
        Width of the count density plot, by default 20
    n_barcodes : int, optional
        Number of barcodes to use, by default 1
    singlet_doublet_priors : list, optional
        List of priors for singlets and doublets, by default None
    singlets_hypothesis_prob_cutoff : float, optional
        Cutoff for singlets, by default 0.9
    
    Returns
    -------
    adata_cdna : anndata.AnnData
        AnnData object containing the merged data.
    """
    adata_cdna = adata_cdna[adata_cdna.obs.merge(adata_cmo.obs, left_index=True, right_index=True, how='inner').index, :]
    adata_cdna.obs = adata_cdna.obs.join(adata_cmo.to_df().astype(np.int32))
    assert adata_cdna.obs_names.duplicated().sum() == 0
    count_depth = adata_cdna.obs[cmo_map.keys()].sum(axis=1).values
    singlets, doublets = cmo_count_density(count_depth, calc_cutoff=True, widths=count_density_widths, title='Before doublet removal')
    if singlet_doublet_priors:
        print(f"Overwriting default hashsolo priors for singlets and doublets: {' '.join(map(str, singlet_doublet_priors))}")
        singlets, doublets = singlet_doublet_priors
    plt.show()
    sce.pp.hashsolo(
        adata=adata_cdna,
        cell_hashing_columns=cmo_map.keys(),
        priors=[0.01, singlets, doublets-0.01],
        number_of_noise_barcodes=len(cmo_map.keys())-n_barcodes,
    )
    adata_cdna = adata_cdna[~adata_cdna.obs['Classification'].isin(['Doublet', 'Negative'])]
    cmo_count_density(count_depth=adata_cdna.obs[cmo_map.keys()].sum(axis=1), calc_cutoff=False)
    adata_cdna = adata_cdna[adata_cdna.obs["singlet_hypothesis_probability"] >= singlets_hypothesis_prob_cutoff]
    cmo_count_density(count_depth=adata_cdna.obs[cmo_map.keys()].sum(axis=1), calc_cutoff=False, title='After doublet removal', color='r')
    plt.show()
    adata_cdna.obs = adata_cdna.obs[['Classification']]
    adata_cdna.obs = adata_cdna.obs.rename({'Classification': 'classification'}, axis=1)
    adata_cdna.obs['classification'] = adata_cdna.obs['classification'].replace(cmo_map)
    return adata_cdna
