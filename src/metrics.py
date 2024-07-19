import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def get_metrics(
    path_to_data: str, y_true_col_name: str, y_score_col_name: str
) -> dict[str, float]:
    """
    Computes the Pearson correlation and roc-auc metric.

    :param path_to_data (str): The path to the pandas dataframe is in .tsv format. The dataframe must contain true "Yes" and "No" labels, and a probability estimate of the class "Yes".
    :param y_true_col_name (str): The name of the column containing the true "Yes" and "No" class labels.
    :param y_score_col_name (str): The name of the column containing the probability estimates of the "Yes" class.

    :return (dict[str, float]): Returns a dictionary with corr and roc keys: corr is the Pearson correlation, roc is the roc-auc metric.
    """

    df = pd.read_csv(path_to_data, sep="\t")
    mask = df[y_true_col_name].isin(["Yes", "No"])

    y_true = df[y_true_col_name][mask].values
    y_true = (y_true == "Yes").astype(int)

    y_score = df[y_score_col_name][mask].values

    corr = np.corrcoef(y_true, y_score)[0, 1]
    roc = roc_auc_score(y_true, y_score)

    return {"corr": corr, "roc": roc}


def main(args):
    metrics = get_metrics(args.path_to_data, args.true_col_name, args.score_col_name)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="Path to data in .tsv extension containing summary and article fields - the text from which the summary is derived.",
    )
    parser.add_argument(
        "--true_col_name",
        type=str,
        required=True,
        help="The name of the column containing the true 'Yes' and 'No' class labels.",
    )
    parser.add_argument(
        "--score_col_name",
        type=str,
        default="1_prob",
        help="The name of the column containing the probability estimates of the 'Yes' class.",
    )

    args = parser.parse_args()

    main(args)
