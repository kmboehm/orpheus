import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import argparse
from glob import glob
import os


def align(df):
    train_mask = (df.split == 'train') & (df.pred_lan.notna()) & (df.pred_vis.notna()) & (df.pred_mul.notna())
    x_tr = df.loc[train_mask, ["pred_lan", "pred_vis", "pred_mul"]]
    y_tr = df.loc[train_mask, "score"].values

    # train classifier
    clf = LinearRegression()
    clf.fit(x_tr, y_tr)

    # predict
    val_pred_mask = (df.split == 'val') & (df.pred_lan.notna()) & (df.pred_vis.notna()) & (df.pred_mul.notna())
    df.loc[val_pred_mask, "pred_mul"] = clf.predict(df.loc[val_pred_mask, ["pred_lan", "pred_vis", "pred_mul"]])
    test_pred_mask = (df.split == 'test') & (df.pred_lan.notna()) & (df.pred_vis.notna()) & (df.pred_mul.notna())
    df.loc[test_pred_mask, "pred_mul"] = clf.predict(df.loc[test_pred_mask, ["pred_lan", "pred_vis", "pred_mul"]])
    train_pred_mask = (df.split == 'train') & (df.pred_lan.notna()) & (df.pred_vis.notna()) & (df.pred_mul.notna())
    df.loc[train_pred_mask, "pred_mul"] = clf.predict(df.loc[train_pred_mask, ["pred_lan", "pred_vis", "pred_mul"]])
    
    return df


def load_data(df, pred_dir, pred_col):
    pred_df = pd.concat([load_subset_data(pred_dir, split, pred_col) for split in ["train", "val", "test"]])
    pred_df['case_id'] = pred_df['case_id'].astype(str)
    pred_df = pred_df.set_index("case_id")
    assert not pred_df.index.duplicated().any()
    df = df.merge(pred_df, how="left", left_index=True, right_index=True, suffixes=("", "_pred"))
    assert df.loc[df['split_pred'].notna(), 'split'].equals(df.loc[df['split_pred'].notna(), 'split_pred'])
    df = df.drop(columns=["split_pred"])
    print(f"for {pred_col}, {len(df[df[pred_col].isna()])} missing")
    return df

def load_subset_data(pred_dir, split, pred_col):
    pred_files = glob(os.path.join(pred_dir, split, "*.pt"))
    case_ids = [os.path.basename(f).replace(".pt", "") for f in pred_files]
    preds = [torch.load(f, map_location='cpu').item() for f in pred_files]
    pred_df = pd.DataFrame({"case_id": case_ids, pred_col: preds, "split": split})
    return pred_df

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--df_path', type=str, required=True)
    PARSER.add_argument('--img_pred_dir', type=str, default="preds/visual")
    PARSER.add_argument('--lan_pred_dir', type=str, default="preds/linguistic")
    PARSER.add_argument('--mult_pred_dir', type=str, default="preds/multimodal")
    PARSER.add_argument("--output_df_path", type=str, default="all_predictions.csv")
    ARGS = PARSER.parse_args()

    DF = pd.read_csv(ARGS.df_path, low_memory=False)
    DF['case_id'] = DF['case_id'].astype(str)
    DF = DF.set_index("case_id")
    DF = load_data(DF, ARGS.img_pred_dir, "pred_vis")
    DF = load_data(DF, ARGS.lan_pred_dir, "pred_lan")
    DF = load_data(DF, ARGS.mult_pred_dir, "pred_mul")
    DF = align(DF)
    print(DF)
    DF.to_csv(ARGS.output_df_path, index=True)
