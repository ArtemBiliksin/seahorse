import argparse
import os
import datasets
import pandas as pd


LANGUAGE2GEM_DATASETS = {
    "de": {"mlsum": "de", "wiki_lingua": "wiki_lingua_german_de"},
    "en": {"xlsum": "english", "xsum": None, "wiki_lingua": "wiki_lingua_english_en"},
    "es": {"xlsum": "spanish", "mlsum": "es", "wiki_lingua": "wiki_lingua_spanish_es"},
    "ru": {"xlsum": "russian", "wiki_lingua": "wiki_lingua_russian_ru"},
    "tr": {"xlsum": "turkish", "wiki_lingua": "wiki_lingua_turkish_tr"},
    "vi": {"xlsum": "vietnamese", "wiki_lingua": "wiki_lingua_vietnamese_vi"},
}

GEM_DATASET2SOURCE_NAME_COLUMN = {
    "mlsum": "text",
    "xlsum": "text",
    "xsum": "document",
    "wiki_lingua": "source",
}

GEM_DATASET2REVISION = {
    path: "main" for path in ["mlsum", "wiki_lingua", "xlsum", "xsum"]
}
# Version of the wiki_lingua dataset used in the SEAHORSE article
GEM_DATASET2REVISION["wiki_lingua"] = "b864b63e079d381a1bd92723d36107153ba23921"


def prepare_gem_dataset(
    gem_dataset: datasets.Dataset, gem_dataset_name: str, text_column: str = "source"
) -> pd.DataFrame:
    """
    Processes dataset, leave two fields: gem_id and after with source for summary.

    :param gem_dataset (datasets.Dataset): dataset from GEM.
    :param gem_dataset_name (str): dataset name from GEM.
    :param text_columns (str): the name of the text field into which the source text for summary will be written.

    :return (pd.DataFrame): pandas dataframe containing gem_id and `text_column` field.
    """

    source_name_column = GEM_DATASET2SOURCE_NAME_COLUMN[gem_dataset_name]
    gem_dataset = gem_dataset.select_columns(["gem_id", source_name_column])
    gem_dataset = gem_dataset.rename_columns({source_name_column: text_column})
    gem_dataset = gem_dataset.to_pandas()

    return gem_dataset


def collect_gem_dataset(
    languages: list[str], text_column: str = "source"
) -> dict[str, pd.DataFrame]:
    """
    Collects data from GEM.

    :param languages (list[str]): list of languages: de, en, es, ru, tr, vi.
    :param text_column (str): the name of the text field into which the source text for summary will be written.

    :return (dict[str, pd.DataFrame]): returns a dictionary containing pandas dataframe containing gem_id and `text_column` field.
    """

    gem_dataset = {"validation": [], "test": []}
    for language in languages:
        for gem_dataset_name, subset_dataset_name in LANGUAGE2GEM_DATASETS[
            language
        ].items():
            val_dataset, test_dataset = datasets.load_dataset(
                path="GEM/" + gem_dataset_name,
                name=subset_dataset_name,
                split=["validation", "test"],
                revision=GEM_DATASET2REVISION[gem_dataset_name],
            )
            val_dataset = prepare_gem_dataset(
                val_dataset, gem_dataset_name, text_column
            )
            test_dataset = prepare_gem_dataset(
                test_dataset, gem_dataset_name, text_column
            )
            gem_dataset["validation"].append(val_dataset)
            gem_dataset["test"].append(test_dataset)

    gem_dataset["validation"] = pd.concat(gem_dataset["validation"], ignore_index=True)
    gem_dataset["test"] = pd.concat(gem_dataset["test"], ignore_index=True)

    return gem_dataset


def add_source_for_seahorse(
    languages: list[str], data_dir: str, text_column: str = "source"
) -> dict[str, pd.DataFrame]:
    """
    Adds the source from which the summary was obtained.

    :param languages (list[str]): list of languages: de, en, es, ru, tr, vi.
    :param data_dir (str): a directory with the original SEAHORSE dataset.
    :param text_column (str): the name of the text field into which the source text for summary will be written.

    :return (dict[str, pd.DataFrame]): returns a dictionary containing the SEAHORSE dataset with the text source for the summary.
    """

    gem_dataset = collect_gem_dataset(languages, text_column)

    seahorse_with_source = {}
    for seahorse_split in ["train", "validation", "test"]:
        data_path = os.path.join(data_dir, seahorse_split + ".tsv")
        seahorse = pd.read_csv(data_path, sep="\t")

        # Let's remove the nan in the gem_id column,
        # otherwise the merge cannot be performed
        seahorse = seahorse[seahorse["gem_id"].notna()]

        # Specific partitioning of data from gem motivated by the article SEAHORSE
        gem_split = "validation" if seahorse_split != "test" else "test"

        seahorse_with_source[seahorse_split] = seahorse.merge(
            gem_dataset[gem_split], on="gem_id", how="inner"
        )

    return seahorse_with_source


def save_dataset(split2dataset: dict[str, pd.DataFrame], data_dir: str) -> None:
    """
    Saves the dataset.

    :param split2dataset (dict[str, pd.DataFrame]): a dictionary containing the dataframe to be saved.
    :param data_dir (str): a path to save the dataset.
    """
    os.makedirs(data_dir, exist_ok=True)

    for split, dataset in split2dataset.items():
        dataset.to_csv(os.path.join(data_dir, split + ".tsv"), sep="\t", index=False)


def main(args):
    seahorse_with_source = add_source_for_seahorse(
        languages=args.languages,
        data_dir=args.path_to_data,
        text_column=args.text_column,
    )
    save_dataset(seahorse_with_source, data_dir=args.path_to_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    help_message = "Filter by language: German (de), English (en), Spanish (es), Russian (ru), Turkish (tr), Vietnamese (vi)."
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        choices=["de", "en", "es", "ru", "tr", "vi"],
        required=True,
        help=help_message,
    )
    parser.add_argument(
        "--path_to_data", type=str, required=True, help="Path to SEAHORSE dataset."
    )
    parser.add_argument(
        "--path_to_result",
        type=str,
        required=True,
        help="Path to the processed SEAHORSE dataset.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="article",
        help="The name of the text field that will contain the source text.",
    )
    args = parser.parse_args()

    main(args)
