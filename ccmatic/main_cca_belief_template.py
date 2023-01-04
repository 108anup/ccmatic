import argparse
import copy
import logging
from fractions import Fraction
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import z3

import ccmatic.common  # Used for side effects
from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables
from ccmatic import CCmatic, OptimizationStruct
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, flatten_dict, get_product_ite, try_except
from ccmatic.verifier import get_cex_df
from cegis.multi_cegis import MultiCegis
from cegis.util import Metric, fix_metrics, get_raw_value, optimize_multi_var, z3_min
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
R = 1
D = 1
HISTORY = R

domain_clauses = []

n_cond = 5
expr_coeffs = [z3.Real(f"Gen__coeff_cond{c}") for c in range(n_cond)]
expr_consts = [z3.Real(f"Gen__const_cond{c}") for c in range(n_cond)]


