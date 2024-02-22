import pandas as pd
import os
import numpy as np


def validate_dataframe(df):
    assert "case_id" in df.columns, "df lacks 'case_id' column"
    assert "score" in df.columns, "df lacks 'score' column"
    assert "input_visual_embedding_path" in df.columns, "df lacks 'input_visual_embedding_path' column"
    assert "text" in df.columns, "df lacks 'text' column"
    assert "split" in df.columns, "df lacks 'split' column"
    assert "output_visual_embedding_path" in df.columns, "df lacks 'output_visual_embedding_path' column"
    assert "output_linguistic_embedding_path" in df.columns, "df lacks 'output_linguistic_embedding_path' column"
    assert len(df) > 0, "df is empty"

    assert set(df['split'].tolist()) - set(["train", "val", "test"]) == set(), "split column should only contain 'train', 'val', or 'test'"

    assert not df.case_id.duplicated().any(), "case_id is not unique"
    assert not df.isnull().values.any(), "df contains NaNs"

    assert np.all([os.path.exists(pt_path) for pt_path in df.input_visual_embedding_path]), "Not all input_visual_embedding_paths exist"
    assert np.all([0.0 <= s <= 1.0 for s in df.score]), "Scores should be in [0, 1]"


def make_example_data(n=100):
    import torch
    os.makedirs("scratch/input", exist_ok=True)
    case_ids = [f"case_{i}" for i in range(n)]
    scores = np.random.rand(n)
    intput_visual_embedding_paths = [f"scratch/input/{case_id}.pt" for case_id in case_ids]
    for pt_path in intput_visual_embedding_paths:
        torch.save(torch.randn(np.random.randint(5,10), 768), pt_path)
    text = [f"this is case {case_id}" for case_id in case_ids]
    splits = np.random.choice(["train", "val", "test"], n)
    output_visual_embedding_paths = [f"scratch/output/{case_id}.pt" for case_id in case_ids]
    output_linguistic_embedding_paths = [f"scratch/output/{case_id}.pt" for case_id in case_ids]
    df = pd.DataFrame({
        "case_id": case_ids,
        "score": scores,
        "input_visual_embedding_path": intput_visual_embedding_paths,
        "text": text,
        "split": splits,
        "output_visual_embedding_path": output_visual_embedding_paths,
        "output_linguistic_embedding_path": output_linguistic_embedding_paths
    })
    df.to_csv("scratch/example.csv", index=False)


if __name__ == '__main__':
    make_example_data()