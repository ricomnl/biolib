import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn import metrics


def prediction_boxplot(
    pred_df=None,
    y_true=None,
    y_pred=None,
    title=None,
    xlims=(-10, 110),
    ylims=(-10, 110),
    n_samples_reg=5_000,
    plot_metrics=True,
    dpi=200
):
    if pred_df is None and y_true is not None and y_pred is not None:
        pred_df = pd.DataFrame({'age': y_true, 'pred_age': y_pred})
    fig, ax = plt.subplots(dpi=dpi)
    for age in np.sort(pred_df["age"].unique()):
        plt.boxplot(
            pred_df[pred_df["age"]==age]["pred_age"],
            positions=[int(age)],
            sym="",
            widths=2,
            patch_artist=True,
            boxprops={"color" : "black"},
            notch = True
        )
    df_sample = pred_df.sample(n=min(y_true.shape[0], n_samples_reg))
    sns.regplot(
        x=df_sample["age"],
        y=df_sample["pred_age"],
        line_kws={"color": "red", "zorder": 3},
        scatter=False
    )
    sns.despine()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(range(*xlims, 10), labels=range(*xlims, 10))
    if ylims == xlims:
        plt.yticks(range(*ylims, 10), labels=range(*ylims, 10));
    plt.plot(xlims, ylims, linestyle="--", color="black")
    plt.ylabel("Predicted age", fontsize=10, labelpad=8)
    plt.xlabel("Chronological age", fontsize=10, labelpad=8)
    if plot_metrics:
        pearsonr = scipy.stats.pearsonr(y_true, y_pred)[0]
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        mae = metrics.median_absolute_error(y_true, y_pred)
        plt.text(0.05, 0.8, f"pearsonr = {pearsonr:.2f}\n" + 
                            f"rmse = {rmse:.2f} years\n" + 
                            f"mae = {mae:.2f} years\n" + 
                            f"n = {y_true.shape[0]:,} cells",
                 transform=ax.transAxes, fontstyle="italic",
                 fontweight="bold", fontsize=8)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=18, y=1.05)
    return fig


def knee(adata_filtered, adata_unfiltered=None, plot_cutoff=None, figsize=(8, 5), show=True):
    """The "knee plot" was introduced in the Drop-seq paper: 
    - Macosko et al., Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets, 
    2015. DOI:10.1016/j.cell.2015.05.002

    In this plot cells are ordered by the number of UMI counts associated to them (shown on the x-axis), 
    and the fraction of droplets with at least that number of cells is shown on the y-axis. 
    The idea is that "real" cells have a certain number of UMI counts and that a threshold on the 
    UMI counts filters those cells.

    Parameters
    ----------
    adata_filtered : anndata.AnnData
        AnnData object containing the filtered data.
    adata_unfiltered : anndata.AnnData, optional
        AnnData object containing the unfiltered data, by default None
    plot_cutoff : int, optional
        If provided, a vertical and horizontal line will be plotted at the cutoff, by default None
    figsize : tuple, optional
        Figure size, by default (8, 5)
    show : bool, optional
        Whether to show the plot, by default True
    """
    knee = np.sort((np.array(adata_filtered.X.sum(axis=1))).flatten())[::-1]
    _, ax = plt.subplots(figsize=figsize)
    if adata_unfiltered is not None:
        uf_knee = np.sort((np.array(adata_unfiltered.X.sum(axis=1))).flatten())[::-1]
        ax.loglog(range(len(uf_knee)), uf_knee, linewidth=5, color="k")
        if plot_cutoff:
            cell_set = np.arange(len(uf_knee))
            num_cells = cell_set[uf_knee>plot_cutoff][::-1][0]
            ax.axhline(y=plot_cutoff, linewidth=3, color="k")
            ax.axvline(x=num_cells, linewidth=3, color="k")
    ax.loglog(range(len(knee)), knee, linewidth=5, color="g")
    ax.set_title("Knee Plot")
    ax.set_xlabel("Set of Barcodes")
    ax.set_ylabel("UMI Counts")
    plt.grid(True, which="both")
    if show:
        plt.show()


def lib_saturation(adata, figsize=(8, 5), color="green", alpha=0.01, ax=None,show=True):
    """Test for library saturation.
    For each cell we ask how many genes did we detect (or see non-zero expression). 
    The idea is that if we have "saturated" our sequencing library then increasing 
    the number of UMI counts (x-axis) will not yield an appreciable increase in 
    the number of genes detected (y-axis).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    figsize : tuple, optional
        Size of the figure, by default (8, 5)
    color : str, optional
        Color of the points, by default "green"
    alpha : float, optional
        Alpha of the points, by default 0.01
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot on, by default None
    show : bool, optional
        Whether to show the plot, by default True
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.asarray(adata.X.sum(axis=1))[:,0], np.asarray(np.sum(adata.X>0, axis=1))[:,0], color=color, alpha=alpha)
    ax.set_title("Library Saturation")
    ax.set_xlabel("UMI Counts")
    ax.set_ylabel("Genes Detected")
    ax.set_xscale('log')
    ax.set_yscale('log')
    if show:
        plt.show()


def obs_comparison(adata, obs_x, obs_y, log2x=False, log2y=False, title=None, s=8, alpha=0.5, height=5, show=True):
    """Plot a scatter plot of two observations.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    obs_x : str
        Name of the observation to plot on the x-axis.
    obs_y : str
        Name of the observation to plot on the y-axis.
    log2x : bool, optional
        Whether to take the log2 of the x-axis, by default False
    log2y : bool, optional
        Whether to take the log2 of the y-axis, by default False
    s : int, optional
        Size of the points, by default 8
    alpha : float, optional
        Alpha of the points, by default 0.5
    height : int, optional
        Height of the plot, by default 5
    show : bool, optional
        Whether to show the plot, by default True
    """
    x = adata.obs[obs_x]
    y = adata.obs[obs_y]
    x_title = obs_x
    y_title = obs_y
    if log2x:
        x = np.log2(x)
        x_title = f"log2({x_title})"
    if log2y:
        y = np.log2(y)
        y_title = f"log2({y_title})"
    ax = sns.jointplot(
        x=x,
        y=y,
        s=s,
        alpha=alpha,
        legend=True,
        height=height,
        marginal_kws={'common_norm':False}
    )
    ax.set_axis_labels(x_title, y_title)
    if title is None:
        plt.suptitle(f'{obs_x} vs {obs_y}')
    else:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def cmo_count_density(count_depth, calc_cutoff=True, widths=20, title='', singlet_doublet_priors=[0.8, 0.2], color=None):
    """Plot the count depth distribution and calculate the cutoff for singlets and doublets.
    If no peak detected, returns sce.pp.hashsolo defaults for singlets and doublets

    Parameters
    ----------
    count_depth : np.ndarray
        Array of count depths.
    calc_cutoff : bool, optional
        Whether to calculate the cutoff, by default True
    widths : int, optional
        Widths for the peak detection, by default 20
    title : str, optional
        Title of the plot, by default ''
    singlet_doublet_priors : list, optional
        Priors for singlets and doublets, by default [0.8, 0.2]
    color : str, optional
        Color of the plot, by default None
    
    Returns
    -------
    tuple
        The cutoff for singlets and doublets.
    """
    ax = sns.kdeplot(np.log2(count_depth), c=color)
    plt.xlabel('log2(count depth)')
    plt.title(title)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    if calc_cutoff:
        peak_indices = scipy.signal.find_peaks_cwt(y, widths=widths)
        valley_indices = scipy.signal.argrelextrema(y, np.less)[0]
        valley_indices = [v for v in valley_indices if v >= peak_indices[0] and v < peak_indices[-1]]
        if len(peak_indices)<=1:
            print(
                f"No bimodal distribution detected, returning hashsolo priors for singlets and doublets: {' '.join(map(str, singlet_doublet_priors))}"
            )
            return singlet_doublet_priors
        cutoff = x[valley_indices[0]]
        plt.plot(x[valley_indices], y[valley_indices], marker='o')
        doublets = (np.log2(count_depth)>cutoff).sum()/count_depth.shape[0]
        singlets = (np.log2(count_depth)<=cutoff).sum()/count_depth.shape[0]
        plt.axvline(cutoff, c='k', linewidth=1)
        plt.text(0.88, 0.95, f"Singlets: {singlets:.2f}", ha='center', va='center', transform=ax.transAxes)
        plt.text(0.88, 0.9, f"Doublets: {doublets:.2f}", ha='center', va='center', transform=ax.transAxes)
        return [singlets, doublets]
    

def cmo_distributions(
    adata,
    cmo_map,
    filter_point=2,
):
    """Plot the distributions of CMOs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data.
    cmo_map : dict
        Dictionary mapping CMOs to their corresponding genes.
    filter_point : int, optional
        Point to draw a vertical line at, by default 2
    """
    pairs = [(cmo, list(set(cmo_map.keys()) - set([cmo]))) for cmo in cmo_map.keys()]
    for cmo1,cmo2 in pairs:
        denom = adata.to_df()[cmo2]
        denom = denom.sum(axis=1)
        cmo_ratio = np.log2(adata.to_df()[cmo1]+1) - np.log2(denom+1)
        _ = sns.kdeplot(cmo_ratio, fill = True, color = "orange")
        sns.despine()
        ylims = plt.ylim()
        plt.plot([0, 0], ylims, linewidth = 1.5, linestyle = "--", color = "black")
        plt.plot([-filter_point, -filter_point], ylims, linewidth = 1.5, linestyle = "--", color = "red")
        plt.plot([filter_point, filter_point], ylims, linewidth = 1.5, linestyle = "--", color = "red")
        plt.xlim(-10, 10)
        plt.title("CMO distribution")
        plt.xlabel(f"log2({cmo1} / all)")
        plt.ylabel("Density")
        plt.title("n = {:,} cells".format(len(adata)), loc = "right", y = 0.9)
        plt.show()