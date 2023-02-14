from collections import defaultdict
from dataclasses import dataclass, field
import z3
import enum

from typing import Callable, Dict, List, Tuple, Union
from ccac.config import ModelConfig
from ccac.variables import Variables

from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten_dict, get_product_ite_cc
from cegis.util import get_raw_value


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

        if (2*i+1 == n_cond):
            c1 = None
            c2 = conds[2*i+0]
        else:
            c1 = conds[2*i+0]
            c2 = conds[2*i+1]

        root_expr = value_if_else_chain(
            [c2], [e1, e2])
        root_exprs.append(root_expr)
        if (c1 is not None):
            root_conds.append(c1)
    return value_if_else_chain(
        root_conds, root_exprs)


def value_if_else_3leaf_unbalanced(
        conds: List[z3.BoolRef],
        exprs: List[z3.ArithRef]) -> z3.ArithRef:
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == 3
    """
    if(c1):
        if(c2):
            e1
        else:
            e2
    else:
        e3
    """
    mid_expr = value_if_else_chain([conds[1]], exprs[:-1])
    return value_if_else_chain([conds[0]], [mid_expr, exprs[-1]])


def value_on_template_execution(
        template_type: TemplateType,
        conds: List[z3.BoolRef],
        exprs: List[z3.ArithRef]) -> z3.ArithRef:

    assert template_type != TemplateType.TARGET_RATE
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == n_cond + 1

    if (template_type == TemplateType.IF_ELSE_CHAIN):
        return value_if_else_chain(conds, exprs)
    elif (template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1):
        return value_if_else_compound_depth_1(conds, exprs)
    elif (template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED):
        return value_if_else_3leaf_unbalanced(conds, exprs)
    else:
        assert False


def add_indent(lines: List[str]):
    ret = []
    for s in lines:
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
        if (isinstance(expr, list)):
            ret.extend(add_indent(expr))
        else:
            ret.append(f"    {expr}")
    ret.append(f"else:")
    expr = exprs[n_cond]
    if (isinstance(expr, list)):
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

        if (2*i+1 == n_cond):
            c1 = None
            c2 = conds[2*i+0]
        else:
            c1 = conds[2*i+0]
            c2 = conds[2*i+1]

        root_expr = str_if_else_chain([c2], [e1, e2])
        root_exprs.append(root_expr)
        if (c1 is not None):
            root_conds.append(c1)
    return str_if_else_chain(root_conds, root_exprs)


def str_if_else_3leaf_unbalanced(
        conds: List[str],
        exprs: List[Union[str, List[str]]]) -> List[str]:
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == 3
    """
    if(c1):
        if(c2):
            e1
        else:
            e2
    else:
        e3
    """
    mid_expr = str_if_else_chain([conds[1]], exprs[:-1])
    return str_if_else_chain([conds[0]], [mid_expr, exprs[-1]])


def str_on_template_execution(
        template_type: TemplateType,
        conds: List[str],
        exprs: List[Union[str, List[str]]]) -> List[str]:

    assert template_type != TemplateType.TARGET_RATE
    n_cond = len(conds)
    n_expr = len(exprs)
    assert n_expr == n_cond + 1

    if (template_type == TemplateType.IF_ELSE_CHAIN):
        return str_if_else_chain(conds, exprs)
    elif (template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1):
        return str_if_else_compound_depth1(conds, exprs)
    elif (template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED):
        return str_if_else_3leaf_unbalanced(conds, exprs)
    else:
        assert False


class TemplateTermType(enum.Enum):
    VAR = 0
    CONST = 1


class TemplateTermUnit(enum.Enum):
    TIME = 0
    BYTES_OR_RATE = 1


@dataclass
class TemplateTerm:
    name: str
    type: TemplateTermType
    unit: TemplateTermUnit
    coeff_search_space: Tuple[float, ...]

    def __hash__(self):
        return self.name.__hash__()


@dataclass
class TemplateBuilder:
    """
    Used to build assignments to lvalues.
    """
    n_exprs: int
    n_conds: int
    template_type: TemplateType

    expr_terms: List[TemplateTerm]
    cond_terms: List[TemplateTerm]

    get_value_for_term: Callable[
        [TemplateTerm, ModelConfig, Variables, int, int],
        Union[z3.ArithRef, float]]

    ctx: z3.Context = z3.main_ctx()

    cond_coeffs: List[Dict[TemplateTerm, z3.ArithRef]] = field(init=False)
    expr_coeffs: List[Dict[TemplateTerm, z3.ArithRef]] = field(init=False)

    cond_terms_by_name: Dict[str, TemplateTerm] = field(init=False)
    expr_terms_by_name: Dict[str, TemplateTerm] = field(init=False)

    expr_terms_by_type: Dict[TemplateTermType, List[TemplateTerm]] = \
        field(init=False)
    cond_terms_by_type: Dict[TemplateTermType, List[TemplateTerm]] = \
        field(init=False)

    expr_terms_by_unit: Dict[TemplateTermUnit, List[TemplateTerm]] = \
        field(init=False)
    cond_terms_by_unit: Dict[TemplateTermUnit, List[TemplateTerm]] = \
        field(init=False)

    def __post_init__(self):
        self.build_coeffs()

        self.expr_terms_by_name = {}
        self.cond_terms_by_name = {}
        for et in self.expr_terms:
            assert et.name not in self.expr_terms_by_name
            self.expr_terms_by_name[et.name] = et
        for ct in self.cond_terms:
            assert ct.name not in self.cond_terms_by_name
            self.cond_terms_by_name[ct.name] = ct

        self.expr_terms_by_type = defaultdict(list)
        self.cond_terms_by_type = defaultdict(list)
        for et in self.expr_terms:
            self.expr_terms_by_type[et.type].append(et)
        for ct in self.cond_terms:
            self.cond_terms_by_type[ct.type].append(ct)

        self.expr_terms_by_unit = defaultdict(list)
        self.cond_terms_by_unit = defaultdict(list)
        for et in self.expr_terms:
            self.expr_terms_by_unit[et.unit].append(et)
        for ct in self.cond_terms:
            self.cond_terms_by_unit[ct.unit].append(ct)

    def build_coeffs(self):
        self.expr_coeffs = [
            {et: z3.Real(
                f"Gen__expr_coeff_{ei}_{et.name}", self.ctx)
                for et in self.expr_terms}
            for ei in range(self.n_exprs)]

        self.cond_coeffs = [
            {ct: z3.Real(
                f"Gen__cond_coeff_{ci}_{ct.name}", self.ctx)
                for ct in self.cond_terms}
            for ci in range(self.n_conds)]

    def get_critical_generator_vars(self) -> List[z3.ArithRef]:
        ret = []
        for ci in range(self.n_conds):
            for tt, ct in self.cond_coeffs[ci].items():
                if(tt.type == TemplateTermType.VAR):
                    ret.append(ct)
        for ei in range(self.n_exprs):
            for tt, et in self.expr_coeffs[ei].items():
                if(tt.type == TemplateTermType.VAR):
                    ret.append(et)
        return ret

    def get_generator_vars(self) -> List[z3.ArithRef]:
        return flatten_dict(self.cond_coeffs) + flatten_dict(self.expr_coeffs)

    def get_search_space_constraints(self):
        sc = []
        for ei in range(self.n_exprs):
            for et in self.expr_terms:
                coeff = self.expr_coeffs[ei][et]
                sc.append(z3.Or(*[coeff == c for c in et.coeff_search_space]))

        for ci in range(self.n_conds):
            for ct in self.cond_terms:
                coeff = self.cond_coeffs[ci][ct]
                sc.append(z3.Or(*[coeff == c for c in ct.coeff_search_space]))

        return sc

    def get_same_units_constraints(self):
        # Group terms by units.

        # Check that all terms in a group have the same unit.
        domain_clauses = []
        for ci in range(self.n_conds):
            non_zero_cond_units = []
            for _, ct_list in self.cond_terms_by_unit.items():
                non_zero_cond_units.append(
                    z3.Or(*[self.cond_coeffs[ci][ct] != 0 for ct in ct_list]))
            domain_clauses.append(z3.Sum(*non_zero_cond_units) <= 1)

        for ei in range(self.n_exprs):
            non_zero_expr_units = []
            for _, et_list in self.expr_terms_by_unit.items():
                non_zero_expr_units.append(
                    z3.Or(*[self.expr_coeffs[ei][et] != 0 for et in et_list]))
            domain_clauses.append(z3.Sum(*non_zero_expr_units) <= 1)

        return domain_clauses

    def get_expr_non_const_constr(self):
        ret = []
        for ei in range(self.n_exprs):
            terms = []
            for et in self.expr_terms:
                if (et.type == TemplateTermType.VAR):
                    terms.append(self.expr_coeffs[ei][et] != 0)
            ret.append(z3.Or(*terms))
        return ret

    @staticmethod
    def get_product(
            cc: CegisConfig, coeff: z3.ArithRef,
            term_eval: z3.ArithRef, term: TemplateTerm) -> z3.ArithRef:
        if (term.type == TemplateTermType.CONST):
            ret = coeff * term_eval
            assert isinstance(ret, z3.ArithRef)
            return ret
        elif (term.type == TemplateTermType.VAR):
            return get_product_ite_cc(
                cc, coeff, term_eval, term.coeff_search_space)
        else:
            assert False

    def get_value_on_execution(
            self, cc: CegisConfig, c: ModelConfig, v: Variables,
            n: int, t: int) -> z3.ArithRef:

        conds = []
        for ci in range(self.n_conds):
            cond_lhs = 0
            for ct in self.cond_terms:
                coeff = self.cond_coeffs[ci][ct]
                term_eval = self.get_value_for_term(ct, c, v, n, t)
                cond_lhs += TemplateBuilder.get_product(
                    cc, coeff, term_eval, ct)
            conds.append(cond_lhs > 0)

        exprs = []
        for ei in range(self.n_exprs):
            expr_rhs = 0
            for et in self.expr_terms:
                coeff = self.expr_coeffs[ei][et]
                term_eval = self.get_value_for_term(et, c, v, n, t)
                expr_rhs += TemplateBuilder.get_product(
                    cc, coeff, term_eval, et)
            exprs.append(expr_rhs)

        return value_on_template_execution(self.template_type, conds, exprs)

    def get_str_on_execution(
            self, solution: z3.ModelRef):

        conds = []
        for ci in range(self.n_conds):
            cond_lhs_list = []
            for ct in self.cond_terms:
                coeff = self.cond_coeffs[ci][ct]
                coeff_value = get_raw_value(solution.eval(coeff))
                if(coeff_value != 0):
                    cond_lhs_list.append(
                        f"+ {coeff_value}{ct.name}"
                    )
            conds.append(" ".join(cond_lhs_list) + " > 0")

        exprs = []
        for ei in range(self.n_exprs):
            expr_rhs_list = []
            for et in self.expr_terms:
                coeff = self.expr_coeffs[ei][et]
                coeff_value = get_raw_value(solution.eval(coeff))
                if(coeff_value != 0):
                    expr_rhs_list.append(
                        f"+ {coeff_value}{et.name}"
                    )
            exprs.append(" ".join(expr_rhs_list))

        return str_on_template_execution(self.template_type, conds, exprs)

    def get_expr_coeff(self, ei: int, et_name: str):
        et = self.expr_terms_by_name[et_name]
        return self.expr_coeffs[ei][et]

    def get_cond_coeff(self, ci: int, ct_name: str):
        ct = self.cond_terms_by_name[ct_name]
        return self.cond_coeffs[ci][ct]


class SynthesisType(enum.Enum):
    RATE_ONLY = 1
    CWND_ONLY = 2
    BOTH_CWND_AND_RATE = 3
