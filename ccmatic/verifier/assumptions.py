import z3
from ccac.cca_aimd import cca_aimd
from ccac.cca_bbr import cca_bbr
from ccac.cca_copa import cca_copa
from ccac.config import ModelConfig
from ccac.model import cca_const, cca_paced
from ccac.utils import make_periodic
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from pyz3_utils.my_solver import MySolver


def get_periodic_constraints(cc: CegisConfig, c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    make_periodic(c, s, v, cc.history)
    return z3.And(*s.assertion_list)


def get_cca_definition(cc: CegisConfig, c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    if c.cca == "const":
        cca_const(c, s, v)
    elif c.cca == "aimd":
        cca_aimd(c, s, v)
    elif c.cca == "bbr":
        cca_bbr(c, s, v)
    elif c.cca == "copa":
        cca_copa(c, s, v)
    elif c.cca == "any":
        pass
    elif c.cca == "paced":
        cca_paced(c, s, v)
    else:
        assert False, "CCA {} not found".format(c.cca)

    return z3.And(*s.assertion_list)
