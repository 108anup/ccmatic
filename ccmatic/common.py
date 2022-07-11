from typing import List, Union


def flatten(l) -> list:
    ret = []
    if(isinstance(l, list)):
        for item in l:
            ret.extend(flatten(item))
        return ret
    else:
        return [l]


def lcs(arr: List[str]) -> str:
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


def get_name_for_list(l: Union[List[str], str]) -> str:
    if(isinstance(l, list)):
        return lcs(l) + "t"
    elif(isinstance(l, str)):
        return l
    else:
        raise TypeError("Expected list or str")
