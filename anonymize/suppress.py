from typing import List
import numpy as np
import pandas as pd
from anonymetrics.anonymetrics import get_groups, calculate_l_diversity, \
    calculate_sensitive_attr_prob_dist, emd_categorical


def suppress_float(df: pd.DataFrame, column_index: int) -> pd.DataFrame:
    """
    Suppresses float attributes in one column in the given dataframe by assigning the mean of values in the column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to anonymize.
    column_index : int
        The index of the attribute column.
    """

    # Get the mean value of attributes
    values = df.iloc[:, column_index].values
    mean = np.mean(values)

    # Suppress the column attributes
    df.iloc[:, column_index] = mean


def suppress_categorical(df: pd.DataFrame, attribute_value: str, column_index: int):
    """
    Suppresses specified categorical attribute values in the given dataframe by replacing them with a common value.

    Parameters
    -----------
    df : pandas DataFrame
        The input dataframe.
    attribute_value : str
        The value that should be suppressed.
    column_index : int
        The column index, where values should be suppressed.
    """

    # get the column name from its index
    column_name = df.columns[column_index]

    # get unique values in the column
    unique_values = df[column_name].unique()

    frozenset_value = [val for val in unique_values if isinstance(val, frozenset)]

    unique_values = [val for val in unique_values if not isinstance(val, frozenset)]

    print(frozenset_value)

    if len(frozenset_value) != 0:
        df.loc[df[column_name] == attribute_value, column_name] \
            = [frozenset_value[0]] * len(df[df[column_name] == attribute_value])
    else:
        # replace attribute_value with unique_values
        df.loc[df[column_name] == attribute_value, column_name] \
            = [frozenset(unique_values)] * len(df[df[column_name] == attribute_value])


def remove_groups(df: pd.DataFrame, qa_indices: List[int], k: int):
    """
    Removes records in df, that are part of groups with < k records.

    Parameters
    -----------
    df : pandas DataFrame
        The input dataframe.
    qa_indices : list of int
        A list of column names to group by.
    k : int
        The desired k to obtain k-anonymity.
    """

    groups = get_groups(df, qa_indices)

    # Collect the indices of all small groups
    small_group_indices = [group.index for group in groups if len(group) < k]
    small_group_indices = [index for sublist in small_group_indices for index in sublist]

    # Drop rows from small groups
    df.drop(small_group_indices, inplace=True)


def remove_groups_with_diversity_smaller_l(df: pd.DataFrame, qa_indices: List[int], sa_indices: List[int], l: int):
    """
    Removes records in df, that are part of groups with < l diversity.

    Parameters
    -----------
    sa_indices : ...
    df : pandas DataFrame
        The input dataframe.
    qa_indices : list of int
        A list of column names to group by.
    l : int
        The minimum l.
    """

    groups = get_groups(df, qa_indices)

    # Collect the indices of all small groups
    small_group_indices = [group.index for group in groups if calculate_l_diversity(group, qa_indices, sa_indices) < l]
    small_group_indices = [index for sublist in small_group_indices for index in sublist]

    # Drop rows from small groups
    df.drop(small_group_indices, inplace=True)


def remove_groups_with_closeness_higher_t(df: pd.DataFrame, qa_indices: List[int], sa_index: int, t: float):
    """
    Removes records in df, that are part of groups with > t closeness.

    Parameters
    -----------
    sa_index : ...
    df : pandas DataFrame
        The input dataframe.
    qa_indices : list of int
        A list of column names to group by.
    t : int
        The maximum t.
    """

    groups = get_groups(df, qa_indices)

    dist_dataset = calculate_sensitive_attr_prob_dist(df, sa_index)

    # Collect the indices of all small groups
    small_group_indices = [group.index for group in groups if
                           emd_categorical(pd.Series(data=calculate_sensitive_attr_prob_dist(group, sa_index),
                                                     index=dist_dataset.index).fillna(0), dist_dataset) > t]
    small_group_indices = [index for sublist in small_group_indices for index in sublist]

    # Drop rows from small groups
    df.drop(small_group_indices, inplace=True)
