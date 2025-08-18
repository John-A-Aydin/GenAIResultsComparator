from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def to_iterable(obj: Any) -> np.ndarray | pd.Series | List:
    """
    Convert object to an iterable, preserving numpy arrays and pandas Series.

    :param obj: The object to convert
    :type obj: Any
    :return: An iterable version of the object
    :rtype: np.ndarray | pd.Series | List
    """
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return list(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.values
    elif isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, str):
        return [obj]
    else:
        try:
            return list(iter(obj))
        except TypeError:
            return [obj]


def get_ngrams(text: str, n: int) -> Dict[str, int]:
    """
    Generate n-grams from a given text.

    :param text: The input text
    :type text: str
    :param n: The number of words in each n-gram
    :type n: int
    :return: A dictionary of n-grams and their counts
    :rtype: Dict[str, int]
    """
    words: List[str] = text.lower().split()  # Split the text into words
    ngrams: zip = zip(*[words[i:] for i in range(n)])  # Create n-grams
    return Counter(" ".join(ngram) for ngram in ngrams)  # Count the n-grams


def batch_get_ngrams(texts: np.ndarray | pd.Series | List[str], n: int) -> List[Dict[str, int]]:
    """
    Generate n-grams for a batch of texts.

    :param texts: The input texts
    :type texts: np.ndarray | pd.Series | List[str]
    :param n: The number of words in each n-gram
    :type n: int
    :return: A list of dictionaries of n-grams and their counts
    :rtype: List[Dict[str, int]]
    """
    if isinstance(texts, np.ndarray):
        return [get_ngrams(text, n) for text in texts]
    elif isinstance(texts, pd.Series):
        return texts.apply(lambda x: get_ngrams(x, n)).tolist()
    else:
        return [get_ngrams(text, n) for text in texts]


def prepare_results_dataframe(
    results_dict: Dict[str, Dict[str, Any]],
    model_col: str = "model_name",
    metric_col: str = "metric_name",
    score_col: str = "score",
    index_col: str = "item_index",
) -> pd.DataFrame:
    """
    Converts a nested dictionary of results into a long-format DataFrame suitable for plotting.
    Handles both single-item results and batch results (lists of scores).

    Example Input `results_dict` (batch):
    {
        'ModelA': {'BLEU': [0.8, 0.85], 'ROUGE': [{'f1': 0.75}, {'f1': 0.78}]},
    }
    Example Output DataFrame:
       item_index model_name metric_name  score
    0           0     ModelA        BLEU   0.80
    1           1     ModelA        BLEU   0.85
    2           0     ModelA    ROUGE_f1   0.75
    3           1     ModelA    ROUGE_f1   0.78

    :param results_dict: Nested dictionary of results. Scores can be single values or lists.
    :type results_dict: Dict[str, Dict[str, Any]]
    :param model_col: Name for the model column.
    :type model_col: str
    :param metric_col: Name for the metric column.
    :type metric_col: str
    :param score_col: Name for the score column.
    :type score_col: str
    :param index_col: Name for the item index column in batch mode.
    :type index_col: str
    :return: A pandas DataFrame in long format.
    :rtype: pd.DataFrame
    """
    records = []
    is_batch = False
    for model_name, metrics_data in results_dict.items():
        for metric_name, score_value in metrics_data.items():
            # Normalize score_value to a list to handle single and batch cases uniformly
            score_list = score_value if isinstance(score_value, list) else [score_value]

            if len(score_list) > 1:
                is_batch = True

            for i, item_score in enumerate(score_list):
                if isinstance(item_score, dict):
                    for sub_metric, sub_score in item_score.items():
                        full_metric_name = f"{metric_name}_{sub_metric}"
                        if isinstance(sub_score, (int, float)):
                            records.append(
                                {
                                    index_col: i,
                                    model_col: model_name,
                                    metric_col: full_metric_name,
                                    score_col: sub_score,
                                }
                            )
                elif isinstance(item_score, (int, float)):
                    records.append(
                        {
                            index_col: i,
                            model_col: model_name,
                            metric_col: metric_name,
                            score_col: item_score,
                        }
                    )

    if not records:
        return pd.DataFrame(columns=[index_col, model_col, metric_col, score_col])

    df = pd.DataFrame(records)
    # If it wasn't a batch run, the index column is not needed.
    if not is_batch and index_col in df.columns:
        df = df.drop(columns=[index_col])

    return df
