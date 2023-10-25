from typing import List
import numpy as np
import pandas as pd

import sys

sys.path.append(".")


def discretize(df: pd.DataFrame, column_index: int, L: float):
    """
    Discretizes a column of real numbers in a dataframe into intervals of fixed length L.
    Each interval is represented as a tuple of (lower_ij, upper_ij).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to anonymize.
    column_index : int
        The index of the numerical attribute column.
    L : float
        The length of the intervals.

    """

    # Check if L is positive
    if L <= 0:
        raise ValueError("Interval length must be positive.")

    # Discretize the sensitive attribute
    df.iloc[:, column_index] = df.iloc[:, column_index].apply(lambda x: ((x // L) * L, ((x // L) * L) + L - 1))


def generalize_categorical(df: pd.DataFrame, indices: List[int], values: List):
    """
     Generalizes specified categorical attribute values in the given dataframe by replacing them with a common value.
     The resulting new DataFrame has the specified values embedded in a list in the corresponding columns.

     Parameters
     -----------
     df : pandas DataFrame
         The input dataframe.
     indices : list of int
         The indices of the columns to generalize.
     values : list of any hashable type
         The categorical attribute values to be generalized.

     """

    values = frozenset(values)

    for j in indices:
        new_col = []
        for val in df.iloc[:, j]:
            if val in values:
                new_col.append(values)
            else:
                new_col.append(val)
        df.iloc[:, j] = np.array(new_col)
