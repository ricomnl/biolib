import time
from math import log, ceil

import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA


def _median_normalize(X):
    """Performs median-normalization.
    
    Parameters
    ----------
    X : numpy.ndarray
        A p-by-n expression matrix containing UMI counts for p genes and n
        cells.
    
    Returns
    -------
    numpy.ndarray
        A p-by-n expression matrix containing the normalized UMI counts.
    
    Notes
    -----
    We first determine the median total UMI count per cell, and then scale
    each expression profile so that its total UMI count equals that number.
    This normalization method was originally described as "Model I" in
    Grün et al., Nature Methods 2014).
    """
    num_transcripts = np.sum(X, axis=1)
    X_norm = (np.median(num_transcripts) / num_transcripts).reshape(-1,1) * X
    return X_norm


def _freeman_tukey_transform(X):
    """Applies the Freeman-Tukey transformation, y = sqrt(x) + sqrt(x+1).
    
    Parameters
    ----------
    X : numpy.ndarray
        A n-by-p expression matrix containing UMI counts for n cells 
        and p genes (usually after median-normalization).
    
    Returns
    -------
    numpy.ndarray
        A n-by-p expression matrix containing the Freeman-Tukey-transformed
        UMI counts.
    
    Notes
    -----
    The Freeman-Tukey transformation serves to stabilize the variance of
    Poisson-distributed random variables. For X ~ Pois(l) with l >= 1, Freeman
    and Tukey (1953) show that Var(X) = 1 (+- 6%).
    """
    return np.sqrt(X) + np.sqrt(X+1)


def _calculate_pc_scores(matrix, d, seed=0, verbose=False):
    """Projects the cells onto their first d principal components.
    
    Parameters
    -----
    X: `numpy.ndarray`
        A n-by-p expression matrix containing the UMI counts for n cells
        and p genes.
    
    Returns
    -------
    `numpy.ndarray`
        A n-by-d matrix containing the coordinates of n cells in d-dimensional
        principal component space.

    Notes
    -----
    We perform median-normalization and Freeman-Tukey-transformation to the UMI
    counts, before performing PCA. Median-normalization serves to counteract
    efficiency noise (Grün et al., 2014), whereas Freeman-Tukey transformation
    stabilizes the technical variance of the data. While PCA does not require
    homoskedastic data, variance-stabilization ensures that the increased
    technical variance of highly expressed genes does not result in the first
    PCs being biased towards highly expressed genes.
    We specify svd_solver='randomized', which invokes the randomized algorithm
    by Halko et al. (2009) to efficiently calculate the first d principal
    components. (We assume that d << min(p, n-1).)
    """
    # median-normalize
    tmatrix = _median_normalize(matrix)
    # Freeman-Tukey transform
    tmatrix = _freeman_tukey_transform(tmatrix)
    pca = PCA(n_components=d, svd_solver='arpack', random_state=seed)
    t0 = time.time()
    tmatrix = pca.fit_transform(tmatrix)
    t1 = time.time()
    var_explained = np.cumsum(pca.explained_variance_ratio_)[-1]
    if verbose:
        print('\tPCA took %.1f s.' % (t1-t0))
        print('\tThe fraction of variance explained by the top %d PCs is %.1f %%.'
              % (d, 100*var_explained))

    return tmatrix


def _calculate_pairwise_distances(X, num_jobs=1):
    """Calculates the distances between all cells in X.
    
    Parameters
    -----
    X: numpy.ndarray
        A n-by-d matrix containing the coordinates of n cells in d-dimensional
        space.
    
    Returns
    -------
    numpy.ndarray
        A n-by-n matrix containing the pairwise distances between all cells.
    
    Notes
    -----
    This uses the Euclidean metric.
    """
    D = pairwise_distances(X, n_jobs=num_jobs, metric='euclidean')
    return D


def knn_smoothing(X, k, d=10, dither=0.03, seed=0, verbose=False):
    """K-nearest neighbor smoothing for UMI-filtered single-cell RNA-Seq data.
    
    This function implements an improved version of the kNN-smoothing 2
    algorithm by Wagner et al.
    (https://www.biorxiv.org/content/early/2018/04/09/217737).
    
    Parameters
    ----------
    X : numpy.ndarray
        A n-by-p expression matrix containing UMI counts for p genes and n
        cells. Must contain floating point values, i.e. dtype=np.float64.
    k : int
        The number of neighbors to use for smoothing.
    d : int, optional
        The number of principal components to use for identifying neighbors.
        Default: 10.
    dither : float, optional
        Amount of dither to apply to the partially smoothed and PCA-transformed
        data in each step. Specified as the fraction of the range of the
        cell scores for each PC. Default: 0.03.
    seed : int, optional
        The seed for initializing the pseudo-random number generator used by
        the randomized PCA algorithm. This usually does not need to be changed.
        Default: 0.
    verbose : bool, optional
        If True, print progress information. Default: False.
    
    Returns
    -------
    numpy.ndarray
        A n-by-p expression matrix containing the smoothed expression values.
        The matrix is not normalized. Therefore, even though efficiency noise
        is usually dampened by the smoothing, median-normalization of the
        smoothed matrix is recommended.
    
    Raises
    ------
    ValueError
        If X does not contain floating point values.
        If k is invalid (k < 1, or k >= n).
        If d is invalid (d < 1 or d > # principal components).
    """
    np.random.seed(seed)

    if not (X.dtype == np.float64 or X.dtype == np.float32):
        raise ValueError('X must contain floating point values! Try X = np.float64(X).')

    n, p = X.shape
    num_pcs = min(p, n-1)  # the number of principal components

    if k < 1 or k > n:
        raise ValueError('k must be between 1 and and %d.' % n)
    if d < 1 or d > num_pcs:
        raise ValueError('d must be between 1 and %d.' % num_pcs)

    if verbose:
        print('Performing kNN-smoothing v2.1 with k=%d, d=%d, and dither=%.3f...' % (k, d, dither))

    t0_total = time.time()

    if k == 1:
        num_steps = 0
    else:
        num_steps = ceil(log(k)/log(2))
    
    S = X.copy()
    
    for t in range(1, num_steps+1):
        k_step = min(pow(2, t), k)
        if verbose:
            print('Step %d/%d: Smooth using k=%d' % (t, num_steps, k_step))
        
        Y = _calculate_pc_scores(S, d, seed=seed, verbose=verbose)
        if dither > 0:
            for l in range(d):
                ptp = np.ptp(Y[:, l])
                dy = (np.random.rand(Y.shape[0])-0.5)*ptp*dither
                Y[:, l] = Y[:, l] + dy

        # determine cell-cell distances using smoothed matrix
        t0 = time.time()
        D = _calculate_pairwise_distances(Y)
        t1 = time.time()
        if verbose:
            print('\tCalculating pair-wise distance matrix took %.1f s.' % (t1-t0))
        
        t0 = time.time()
        A = np.argsort(D, axis=1, kind='mergesort')
        t1 = time.time()
        if verbose:
            print('\tRunning argsort took %.1f s.' % (t1-t0))
        
        t0 = time.time()
        for j in range(X.shape[0]):
            ind = A[j, :k_step]
            S[j, :] = np.sum(X[ind, :], axis=0)
        t1 = time.time()
        if verbose:
            print('\tCalculating the smoothed expression matrix took %.1f s.' % (t1-t0))

    t1_total = time.time()
    if verbose:
        print('kNN-smoothing finished in %.1f s.' % (t1_total-t0_total))

    return S


def knn_smooth_adata(
    adata,
    groupby=['donor'],
    k=16,
    n_components=20,
    dither=0.03,
    random_state=42,
):
    """Wrapper for NN-smoothing for use with AnnData objects.
    
    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the expression matrix in adata.X.
    groupby : str or list of str, optional
        The key(s) of the observations grouping to consider. Default: ['donor'].
    k : int, optional
        The number of neighbors to use for smoothing. Default: 16.
    n_components : int, optional
        The number of principal components to use for identifying neighbors.
        Default: 20.
    dither : float, optional
        Amount of dither to apply to the partially smoothed and PCA-transformed
        data in each step. Specified as the fraction of the range of the
        cell scores for each PC. Default: 0.03.
    random_state : int, optional
        The seed for initializing the pseudo-random number generator used by
        the randomized PCA algorithm. This usually does not need to be changed.
        Default: 42.

    Returns
    -------
    AnnData
        An AnnData object containing the smoothed expression matrix in adata.X.
    """
    adatas = []
    for _, subset_df in adata.obs.groupby(groupby):
        adata_sub = adata[subset_df.index, :].copy()
        sc.pp.filter_genes(adata_sub, min_cells=1)
        sc.pp.filter_cells(adata_sub, min_genes=1)
        S = knn_smoothing(adata_sub.X.toarray(), k=k, d=n_components, dither=dither, seed=random_state)
        adata_sub.X = csr_matrix(S)
        adatas.append(adata_sub)
    adata_knn = sc.concat(adatas, join='outer', merge='first')
    return adata_knn


# if __name__ == '__main__':
#     import scanpy as sc
#     import cellrank as cr

#     adata = cr.datasets.reprogramming_schiebinger()
#     sc.pp.subsample(adata, fraction=0.1)
#     sc.pp.filter_genes(adata, min_cells=10)
#     sc.pp.filter_cells(adata, min_genes=100)
#     S = knn_smoothing(adata.X.toarray(), k=2, d=10, dither=0, seed=42, verbose=True)