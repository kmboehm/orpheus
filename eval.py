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
    results_dict[f"{prefix}_pearsonr_p"] = pearsonr(truth, pred)[1]
    results_dict[f"{prefix}_mae"] = {'value': mean_absolute_error(truth, pred), "ci": bootstrap_ci(pred, truth, mean_absolute_error)}
    results_dict[f"{prefix}_ccc"] = {'value': CCC(torch.tensor(pred), torch.tensor(truth)).item(), "ci": bootstrap_ci(torch.tensor(pred), torch.tensor(truth), CCC)}
    for k, v in results_dict.items():
        if k.startswith(prefix):
            print(f"{k}: {v}")
    print('-'*50)
    return results_dict



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