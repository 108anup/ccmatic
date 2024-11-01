import z3
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable
from ccmatic.cegis import CegisConfig

from cegis import NAME_TEMPLATE


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 280)
# pd.set_option('display.max_colwidth', None)


def flatten(l) -> list:
    ret = []
    if(isinstance(l, list) or isinstance(l, np.ndarray)):
        for item in l:
            ret.extend(flatten(item))
        return ret
    else:
        return [l]


def flatten_dict(l: Union[dict, list]) -> list:
    ret = []
    if(isinstance(l, dict)):
        for item in l.values():
            ret.extend(flatten_dict(item))
        return ret
    if(isinstance(l, list) or isinstance(l, np.ndarray)):
        for item in l:
            ret.extend(flatten_dict(item))
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


def remove_none(l: List) -> List:
    ret = []
    for x in l:
        if(x is not None):
            ret.append(x)
    return ret


def get_val_list(model: z3.ModelRef, l: List) -> List:
    ret = []
    for x in l:
        if(isinstance(x, z3.BoolRef)):
            try:
                val = bool(model.eval(x))
            except z3.z3types.Z3Exception:
                # # Ideally this should not happen
                # # This will mostly only happen in buggy cases.
                # assert False
                # Happens when mode_f[n][0] is don't care
                val = -1
            ret.append(val)
        else:
            raise NotImplementedError
    return ret


def get_product_ite_cc(cc: CegisConfig, discrete_var, cts_var, discrete_domain) -> z3.ArithRef:
    if(cc.opt_pdt):
        return get_product_ite(discrete_var, cts_var, discrete_domain)
    else:
        return discrete_var * cts_var


def get_product_ite(discrete_var, cts_var, discrete_domain) -> z3.ArithRef:
    term_list = []
    for val in discrete_domain:
        term_list.append(z3.If(discrete_var == val, val * cts_var, 0))
    ret = z3.Sum(*term_list)
    assert isinstance(ret, z3.ArithRef)
    return ret


def get_renamed_vars(var_list, n_cex):
    renamed_var_list = []
    name_template = NAME_TEMPLATE + str(n_cex)
    for def_var in var_list:
        renamed_var = z3.Const(
            name_template.format(def_var.decl().name()), def_var.sort())
        renamed_var_list.append(renamed_var)
    return renamed_var_list


def substitute_values_df(assumption_record: pd.Series,
                         name_template: str,
                         var_dict: Dict[str, z3.ExprRef]):
    # var_dict is mostly used to infer var sorts.
    expr_list = []
    for vname, val in assumption_record.items():
        assert isinstance(vname, str)
        v = var_dict[vname]
        new_name = name_template.format(v.decl().name())
        expr_list.append(z3.Const(new_name, v.sort()) == val)
    return z3.And(*expr_list)


def try_except(function: Callable):
    try:
        return function()
    except Exception:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


def try_except_wrapper(function):
    def func_to_return(*args, **kwargs):
        def func_to_try():
            return function(*args, **kwargs)
        return try_except(func_to_try)
    return func_to_return
