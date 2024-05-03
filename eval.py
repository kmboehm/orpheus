import pandas as pd
import argparse
from sklearn.metrics import (
    # confusion_matrix,
    r2_score,
    # classification_report,
    # average_precision_score,
    # roc_auc_score,
    # precision_recall_curve,
    # roc_curve,
    # auc,
    mean_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
import json
import torch
from torchmetrics import ConcordanceCorrCoef
import matplotlib.pyplot as plt
import seaborn as sns
import os


def bootstrap_ci(preds, truth, metric_function, n_bootstraps=1000, ci=0.95):
    """
    Calculate the confidence interval of a metric using the bootstrap method.
    Args:
    preds: list of predictions
    truth: list of true values
    metric_function: function that takes in preds and truth and returns a scalar
    n_bootstraps: number of bootstrap samples to take
    ci: confidence interval
    Returns:
    lower_bound: lower bound of the confidence interval
    upper_bound: upper bound of the confidence interval
    """
    n = len(preds)
    bootstrapped_metrics = []
    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(range(n), n, replace=True)
        sample_preds = preds[sample_indices]
        sample_truth = truth[sample_indices]
        bootstrapped_metrics.append(metric_function(sample_preds, sample_truth))
    bootstrapped_metrics = np.array(bootstrapped_metrics)
    two_tail_val = (1 - ci) / 2
    lower_bound = np.percentile(
        bootstrapped_metrics, two_tail_val * 100
    )
    upper_bound = np.percentile(
        bootstrapped_metrics, (ci + two_tail_val) * 100
    )
    return lower_bound, upper_bound


def calculate_metrics(pred_col, split, df, model_type, results_dict):
    prefix = f"{model_type}_{split}"
    CCC = ConcordanceCorrCoef()
    mask_ = df[pred_col].notna() & df["score"].notna() & (df["split"] == split)
    pred = df.loc[mask_, pred_col].values
    truth = df.loc[mask_, "score"].values
    results_dict[f"{prefix}_r2"] = r2_score(truth, pred)
    results_dict[f"{prefix}_pearsonr"] = {'value': pearsonr(truth, pred)[0], "ci": bootstrap_ci(pred, truth, lambda x,y: pearsonr(x, y)[0])}
    results_dict[f"{prefix}_pearsonp"] = pearsonr(truth, pred)[1]
    results_dict[f"{prefix}_mae"] = {'value': mean_absolute_error(truth, pred), "ci": bootstrap_ci(pred, truth, mean_absolute_error)}
    results_dict[f"{prefix}_ccc"] = {'value': CCC(torch.tensor(pred), torch.tensor(truth)).item(), "ci": bootstrap_ci(torch.tensor(pred), torch.tensor(truth), CCC)}
    for k, v in results_dict.items():
        if k.startswith(prefix):
            print(f"{k}: {v}")
    print('-'*50)
    return results_dict


def plot_barplots(metrics):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # convert to pandas dataframe
    df = pd.DataFrame(metrics).T
    print(df)
    df['Split'] = df.index.str.split('_').str[1].map({"train": "train", "val": "val.", "test": "test"})
    df['metric'] = df.index.str.split('_').str[2]
    df['Model'] = df.index.str.split('_').str[0].map({"multimodal": "VL", "linguistic": "L", "visual": "V"})
    df["order"] = df["Split"].map({"train": 0, "val.": 3, "test": 6})
    df.loc[df["Model"] == "V", "order"] += 1
    df.loc[df["Model"] == "VL", "order"] += 2
    df['ci'] = df['ci'].apply(lambda x: [x, x] if isinstance(x, float) else x)
    df['lower_bound'] = df['ci'].apply(lambda x: x[0])
    df['upper_bound'] = df['ci'].apply(lambda x: x[-1])
    df = df.drop(columns=['ci'])
    print(df)
    for metric in df['metric'].unique():
        plot_barplot(df[df['metric'] == metric], metric)


def plot_barplot(df, metric):        
    df = df.sort_values(by=["order"])

    print(df)

    pal = sns.color_palette("cubehelix", 3)
    palette_dict = {"L": pal[0], "V": pal[1], "VL": pal[2]}

    fig = plt.figure(figsize=(2.65, 2.25), tight_layout=True)
    g = sns.barplot(
        data=df,
        x="Split",
        hue="Model",
        palette=palette_dict,
        alpha=1.0,
        y="value",
        order=["train", "val.", "test"],
        hue_order=["L", "V", "VL"],
    )
    # plot error bars contained as (lower_bound, upper_bound_ in df["r_95%_C.I."]
    for idx, x in enumerate(
        [
            -0.4,
            -0.13333333333333333,
            0.13333333333333336,
            0.6,
            0.8666666666666667,
            1.1333333333333333,
            1.6,
            1.8666666666666667,
            2.1333333333333333,
        ]
    ):
        lower_bound = df["lower_bound"].iloc[idx]
        upper_bound = df["upper_bound"].iloc[idx]

        # plot black vertical line from lower bound to upper bound
        plt.plot(
            [x + 0.25 / 2] * 2,
            [lower_bound, upper_bound],
            color="black",
            linewidth=0.5,
        )

    plt.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax = plt.gca()
    sns.move_legend(ax, labelspacing=0, loc="center left", bbox_to_anchor=(1.01, 0.9))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.ylabel(metric)
    plt.savefig(f"plots/{metric}.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--pred_df_path', type=str, required=True)
    ARGS = PARSER.parse_args()

    DF = pd.read_csv(ARGS.pred_df_path, low_memory=False)
    DF['case_id'] = DF['case_id'].astype(str)

    METRICS = dict()

    METRICS = calculate_metrics("pred_mul", "train", DF, "multimodal", METRICS)
    METRICS = calculate_metrics("pred_lan", "train", DF, "linguistic", METRICS)
    METRICS = calculate_metrics("pred_vis", "train", DF, "visual", METRICS)
    METRICS = calculate_metrics("pred_mul", "val", DF, "multimodal", METRICS)
    METRICS = calculate_metrics("pred_lan", "val", DF, "linguistic", METRICS)
    METRICS = calculate_metrics("pred_vis", "val", DF, "visual", METRICS)
    METRICS = calculate_metrics("pred_mul", "test", DF, "multimodal", METRICS)
    METRICS = calculate_metrics("pred_lan", "test", DF, "linguistic", METRICS)
    METRICS = calculate_metrics("pred_vis", "test", DF, "visual", METRICS)
    with open("metrics.json", "w") as f:
        json.dump(METRICS, f)

    plot_barplots(METRICS)
    