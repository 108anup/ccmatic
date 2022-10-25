import copy
import functools
from typing import Callable, List, Optional

import z3
from ccac.variables import VariableNames

from ccmatic.cegis import CegisCCAGen, CegisConfig, CegisMetaData
from ccmatic.common import get_renamed_vars
from ccmatic.verifier import (get_cex_df, get_desired_necessary,
                              get_desired_ss_invariant, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic.verifier.ideal import IdealLink
from cegis import NAME_TEMPLATE


class CCmatic():
    # Boilerplate for common steps
    # Ad-hoc, created from past runs...

    # Call setup_config_vars
    # Define template
    # Call setup_cegis_loop
    # Define method overrides (for printing candidate solutions)
    # Call run_cegis

    cc: CegisConfig

    def __init__(self, cc: CegisConfig):
        self.cc = cc

    def setup_config_vars(self):
        cc = self.cc
        if(cc.ideal_link):
            (c, s, v,
             ccac_domain, ccac_definitions, environment,
             verifier_vars, definition_vars) = IdealLink.setup_cegis_basic(cc)
        else:
            (c, s, v,
             ccac_domain, ccac_definitions, environment,
             verifier_vars, definition_vars) = setup_cegis_basic(cc)

        self.c = c
        self.s = s
        self.v = v
        self.ccac_domain = ccac_domain
        self.ccac_definitions = ccac_definitions
        self.environment = environment
        self.verifier_vars = verifier_vars
        self.definition_vars = definition_vars

        if(cc.synth_ss):
            d = get_desired_ss_invariant(cc, c, v)
            desired = d.desired_invariant
        else:
            d = get_desired_necessary(cc, c, v)
            desired = d.desired_necessary
        self.d = d
        self.desired = desired

    def setup_cegis_loop(
            self, search_constraints: z3.ExprRef,
            template_definitions: List[z3.ExprRef],
            generator_vars: List[z3.ExprRef],
            get_solution_str: Callable[[z3.ModelRef, List[z3.ExprRef], int], str]):
        vn = VariableNames(self.v)

        self.search_constraints = search_constraints
        self.ctx = z3.main_ctx()
        self.specification = z3.Implies(self.environment, self.desired)
        assert(isinstance(self.specification, z3.ExprRef))
        self.definitions = z3.And(
            self.ccac_domain, self.ccac_definitions, *template_definitions)
        assert(isinstance(self.definitions, z3.ExprRef))

        if(self.d.steady_state_variables):
            for sv in self.d.steady_state_variables:
                generator_vars.append(sv.lo)
                generator_vars.append(sv.hi)
        self.generator_vars = generator_vars
        self.critical_generator_vars = generator_vars

        # Closures
        v = self.v
        c = self.c
        cc = self.cc
        d = self.d

        def get_counter_example_str(
                counter_example: z3.ModelRef,
                verifier_vars: List[z3.ExprRef]) -> str:
            df = get_cex_df(counter_example, v, vn, c)
            desired_string = d.to_string(cc, c, counter_example)
            ret = "{}\n{}, alpha={}.".format(df, desired_string,
                                             counter_example.eval(v.alpha))
            return ret

        def get_verifier_view(
                counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
                definition_vars: List[z3.ExprRef]) -> str:
            return get_counter_example_str(counter_example, verifier_vars)

        def get_generator_view(
                solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                definition_vars: List[z3.ExprRef], n_cex: int) -> str:
            df = get_gen_cex_df(solution, v, vn, n_cex, c)

            dcopy = copy.copy(d)
            name_template = NAME_TEMPLATE + str(n_cex)
            dcopy.rename_vars(
                self.verifier_vars + self.definition_vars, name_template)
            desired_string = dcopy.to_string(cc, c, solution)

            renamed_alpha = get_renamed_vars([v.alpha], n_cex)[0]
            gen_view_str = "{}\n{}, alpha={}.".format(df, desired_string,
                                                      solution.eval(renamed_alpha))
            return gen_view_str

        self.get_counter_example_str = get_counter_example_str
        self.get_verifier_view = get_verifier_view
        self.get_generator_view = get_generator_view
        self.get_solution_str = get_solution_str

    @staticmethod
    def get_pretty_term(val, suffix: str):
        if(val == 0):
            return None
        if(val == 1):
            return f'+{suffix}'
        if(val == -1):
            return f'-{suffix}'
        if(val > 0):
            return f'+{val}{suffix}'
        if(val < 0):
            return f'-{-val}{suffix}'

    def run_cegis(self, known_solution: Optional[z3.ExprRef]):
        # Directly update any closures or critical_generator_vars
        # or any other expression before calling run function.

        assert self.get_solution_str is not None, \
            "This needs to be set manually based on template."

        # Debugging:
        debug_known_solution = None
        if self.cc.DEBUG:
            debug_known_solution = known_solution
            assert known_solution is not None
            # self.search_constraints = z3.And(
            #     self.search_constraints, known_solution)
            # assert(isinstance(self.search_constraints, z3.ExprRef))

            # Definitions (including template)
            with open('tmp/definitions.txt', 'w') as f:
                assert(isinstance(self.definitions, z3.ExprRef))
                f.write(self.definitions.sexpr())

        try:
            md = CegisMetaData(self.critical_generator_vars)
            cg = CegisCCAGen(
                self.generator_vars, self.verifier_vars, self.definition_vars,
                self.search_constraints, self.definitions, self.specification,
                self.ctx, debug_known_solution, md)
            cg.get_solution_str = self.get_solution_str
            cg.get_counter_example_str = self.get_counter_example_str
            cg.get_generator_view = self.get_generator_view
            cg.get_verifier_view = self.get_verifier_view
            run_verifier = functools.partial(
                run_verifier_incomplete, c=self.c, v=self.v, ctx=self.ctx)
            cg.run_verifier = run_verifier
            cg.run()

        except Exception:
            import sys
            import traceback

            import ipdb
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
