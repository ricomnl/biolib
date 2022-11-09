import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn import metrics


def plot_prediction_boxplot(
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
    df_sample = pred_df.sample(n_samples_reg)
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