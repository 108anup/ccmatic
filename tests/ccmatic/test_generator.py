import z3
from ccmatic.common import try_except
from ccmatic.generator import TemplateType, str_on_template_execution, value_on_template_execution

TEST_TEMPLATES = [
    TemplateType.IF_ELSE_CHAIN,
    TemplateType.IF_ELSE_COMPOUND_DEPTH_1,
]


def test_tempate_execution():
    n_cond = 9
    n_expr = n_cond + 1
    conds = [z3.Bool(f"cond_{i+1}") for i in range(n_cond)]
    exprs = [z3.Real(f"expr_{i+1}") for i in range(n_expr)]

    for template_type in TEST_TEMPLATES:
        print(value_on_template_execution(
              template_type, conds, exprs))

    print(value_on_template_execution(
          TemplateType.IF_ELSE_3LEAF_UNBALANCED, conds[:2], exprs[:3]))


def test_template_strs():
    n_cond = 9
    n_expr = n_cond + 1
    conds = [f"cond_{i+1}" for i in range(n_cond)]
    exprs = [f"expr_{i+1}" for i in range(n_expr)]

    for template_type in TEST_TEMPLATES:
        ret = str_on_template_execution(
            template_type, conds, exprs)
        print("\n".join(ret))

    ret = str_on_template_execution(
        TemplateType.IF_ELSE_3LEAF_UNBALANCED, conds[:2], exprs[:3])
    print("\n".join(ret))


if (__name__ == "__main__"):
    try_except(test_tempate_execution)
    try_except(test_template_strs)
