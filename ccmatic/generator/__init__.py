import z3
import enum

from typing import List, Union


class TemplateType(enum.Enum):
    IF_ELSE_CHAIN = 1
    IF_ELSE_COMPOUND_DEPTH_1 = 2
    TARGET_RATE = 3

    IF_ELSE_3LEAF_UNBALANCED = 4


def value_if_else_chain(
        conds: List[z3.BoolRef],
        exprs: List[z3.ArithRef]) -> z3.ArithRef:
    n_cond = len(conds)
    n_expr = len(exprs)

    """
    if(c1):
        e1
    elif(c2):
        e2
    ...
    else:
        ek
    """
    ret = exprs[-1]
    for ci in range(n_cond-1, -1, -1):
        ret = z3.If(conds[ci], exprs[ci], ret)
    assert isinstance(ret, z3.ArithRef)
    return ret


def value_if_else_compound_depth_1(
    conds: List[z3.BoolRef],
    exprs: List[z3.ArithRef]) -> z3.ArithRef:
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr % 2 == 0
    """
    if(c1):
        if(c2):
            e1
        else:
            e2
    elif(c3):
        if(c4):
            e1
        else:
            e2
    ...
    else:
        if(ck-1):
            ek-1
        else:
            ek
    """
    # The tree is always single depth for now.
    root_conds = []
    root_exprs = []
    for i in range(int(n_expr/2)):
        e1 = exprs[2*i+0]
        e2 = exprs[2*i+1]

        if(2*i+1 == n_cond):
            c1 = None
            c2 = conds[2*i+0]
        else:
            c1 = conds[2*i+0]
            c2 = conds[2*i+1]

        root_expr = value_if_else_chain(
            [c2], [e1, e2])
        root_exprs.append(root_expr)
        if(c1 is not None):
            root_conds.append(c1)
    return value_if_else_chain(
        root_conds, root_exprs)


def value_on_template_execution(
        template_type: TemplateType,
        conds: List[z3.BoolRef],
        exprs: List[z3.ArithRef]) -> z3.ArithRef:

    assert template_type != TemplateType.TARGET_RATE
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == n_cond + 1

    if(template_type == TemplateType.IF_ELSE_CHAIN):
        return value_if_else_chain(conds, exprs)

    elif(template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1):
        return value_if_else_compound_depth_1(conds, exprs)

    else:
        assert False


def add_indent(l: List[str]):
    ret = []
    for s in l:
        ret.append("    " + s)
    return ret


def str_if_else_chain(
    conds: List[str],
    exprs: List[Union[str, List[str]]]) -> List[str]:

    n_cond = len(conds)
    n_expr = len(exprs)
    """
    if(c1):
        e1
    elif(c2):
        e2
    ...
    else:
        ek
    """
    ret = []
    for ci in range(n_cond):
        IF = "if" if ci == 0 else "elif"
        ret.append(f"{IF} ({conds[ci]}):")
        expr = exprs[ci]
        if(isinstance(expr, list)):
            ret.extend(add_indent(expr))
        else:
            ret.append(f"    {expr}")
    ret.append(f"else:")
    expr = exprs[n_cond]
    if(isinstance(expr, list)):
        ret.extend(add_indent(expr))
    else:
        ret.append(f"    {expr}")
    return ret


def str_if_else_compound_depth1(
    conds: List[str],
    exprs: List[Union[str, List[str]]]) -> List[str]:
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr % 2 == 0
    """
    if(c1):
        if(c2):
            e1
        else:
            e2
    elif(c3):
        if(c4):
            e1
        else:
            e2
    ...
    else:
        if(ck-1):
            ek-1
        else:
            ek
    """
    # The tree is always single depth for now.
    root_conds = []
    root_exprs = []
    for i in range(int(n_expr/2)):
        e1 = exprs[2*i+0]
        e2 = exprs[2*i+1]

        if(2*i+1 == n_cond):
            c1 = None
            c2 = conds[2*i+0]
        else:
            c1 = conds[2*i+0]
            c2 = conds[2*i+1]

        root_expr = str_if_else_chain([c2], [e1, e2])
        root_exprs.append(root_expr)
        if(c1 is not None):
            root_conds.append(c1)
    return str_if_else_chain(root_conds, root_exprs)


def str_on_template_execution(
        template_type: TemplateType,
        conds: List[str],
        exprs: List[Union[str, List[str]]]) -> List[str]:

    assert template_type != TemplateType.TARGET_RATE
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == n_cond + 1

    if(template_type == TemplateType.IF_ELSE_CHAIN):
        return str_if_else_chain(conds, exprs)

    elif(template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1):
        return str_if_else_compound_depth1(conds, exprs)

    else:
        assert False
