import argparse
import os
import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from more_itertools import chunked
from tqdm import tqdm


def rate_samples_with_model(
    tokenizer,
    model,
    articles: list[str],
    summaries: list[str],
    prompt: str = "premise: {} hypothesis: {}",
) -> list[dict[str, np.ndarray]]:
    """
    Computes an estimate of the probabilities of the "0" and "1" classes for a batch of articles and summaries using a model.

    :param tokenizer: tokenizer matching the model-estimator.
    :param model: seq2seq model-estimator.
    :param articles (list[str]): a list of strings containing the articles.
    :param summaries (list[str]): a list of strings containing a summary.
    :param prompt (str): a template line where the article and summary will be substituted. Default prompt = "premise: {} hypothesis: {}".

    :return (list[dict[str, np.ndarray]]): returns a list of the following form: [{"0": prob_0, "1": prob_1}, ...].
    """

    prompts_with_placeholders = [
        prompt.format(a, s) for a, s in zip(articles, summaries)
    ]
    inputs = tokenizer(
        prompts_with_placeholders,
        return_tensors="pt",
        padding=True,
        max_length=2048,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1, output_logits=True, return_dict_in_generate=True
        )

    logits = outputs.logits
    logits_for_class_token = logits[0]
    probs_for_class_token = logits_for_class_token.softmax(-1).cpu().numpy()
    results = [
        {"0": prob_0, "1": prob_1}
        for prob_0, prob_1 in probs_for_class_token[:, [497, 333]]
    ]
    # 333 - "1", 497 - "0"

    return results


def rate_dataset(
    eval_model_name: str,
    articles: list[str],
    summaries: list[str],
    device: torch.device,
    batch_size: int = 10,
    dtype: torch.dtype = torch.float32,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Computes probability estimates of the "0" and "1" classes.

    :param eval_model_name (str): model name.
    :param articles (list[str]): a list of strings containing the articles.
    :param summaries (list[str]): a list of strings containing a summary.
    :param device (torch.device): computing device.
    :param batch_size (int): batch size for the model.
    :param dtype (torch.dtype): data type for calculations.
    "param prefix (str): prefix for resulting columns "0_prob" and "1_prob".

    :return (pd.DataFrame): Returns a pandas dataframe that contains two fields prefix + "0_prob" and prefix + "1_prob", which are probability estimates of classes "0" and "1" respectively.
    """

    result_df = pd.DataFrame()

    tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(eval_model_name)

    model.eval()
    model.to(dtype)
    model.to(device)

    chunked_articles = chunked(articles, batch_size)
    chunked_summaries = chunked(summaries, batch_size)

    result = []  # [{"0": 0_prob, "1": 1_prob}, ...]

    total = len(articles) // batch_size + (len(articles) % batch_size != 0)
    for chunk_article, chunk_summary in tqdm(
        zip(chunked_articles, chunked_summaries), total=total
    ):
        sample_results = rate_samples_with_model(
            tokenizer, model, chunk_article, chunk_summary
        )

        result.extend(sample_results)

    assert len(result) == len(articles)

    probs_for_0 = [el["0"] for el in result]
    probs_for_1 = [el["1"] for el in result]

    result_df[prefix + "0_prob"] = probs_for_0
    result_df[prefix + "1_prob"] = probs_for_1

    return result_df


def main(args: argparse.Namespace):
    rating_df = pd.read_csv(args.path_to_data, sep="\t")
    articles = rating_df[args.article_col_name].values.tolist()
    summaries = rating_df[args.summary_col_name].values.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32
    seahorse_rating_df = rate_dataset(
        args.model_name,
        articles,
        summaries,
        device,
        args.batch_size,
        dtype,
        args.prefix,
    )

    result_df = pd.concat([rating_df, seahorse_rating_df], axis=1)

    os.makedirs(os.path.dirname(args.path_to_result), exist_ok=True)
    result_df.to_csv(args.path_to_result, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="Path to data in .tsv extension containing summary and article fields - the text from which the summary is derived.",
    )
    parser.add_argument(
        "--article_col_name",
        type=str,
        default="article",
        help="The column name that contains the source text for the summary.",
    )
    parser.add_argument(
        "--summary_col_name",
        type=str,
        default="summary",
        help="The name of the column that contains the summary.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model Name. Model name with HF and path to checkpoint are supported.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for the model."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use type fp16 for calculations."
    )
    parser.add_argument(
        "--path_to_result",
        type=str,
        required=True,
        help="The path to the output data in the .tsv extension.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the resulting columns '0_prob' and '1_prob'. There will be names prefix + '0_prob' and prefix + '1_prob'.",
    )

    args = parser.parse_args()

    main(args)
