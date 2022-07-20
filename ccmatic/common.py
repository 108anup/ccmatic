import numpy as np
import pandas as pd
from typing import List, Union


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def flatten(l) -> list:
    ret = []
    if(isinstance(l, list) or isinstance(l, np.ndarray)):
        for item in l:
            ret.extend(flatten(item))
        return ret
    else:
        return [l]


def lcs(arr: Union[List[str], np.ndarray]) -> str:
    """
    Least common substring of a list of strings
    """
    n = len(arr)
    larr = [len(x) for x in arr]
    for i in range(min(larr)):
        same = True
        for j in range(1, n):
            if(arr[j][i] != arr[0][i]):
                same = False
                break
        if(not same):
            return arr[0][:i]
    return arr[0][:min(larr)]


def get_name_for_list(l: Union[np.ndarray, List[str], str]) -> str:
    # import ipdb; ipdb.set_trace()
    if(isinstance(l, list) or isinstance(l, np.ndarray)):
        return lcs(l) + "t"
    elif(isinstance(l, str)):
        return l
    else:
        raise TypeError("Expected list or str")
