from dataclasses import dataclass
from typing import Union, List, Optional

import z3

from cegis import Cegis, remove_solution


@dataclass
class CegisMetaData:
    critical_generator_vars: Optional[List[z3.ExprRef]]


class CegisConfig:
    # template
    history: int = 4
    template_cca_lower_bound: float = 0.01
    template_queue_bound: bool = False
    template_mode_switching: bool = False
    template_loss_oracle: bool = True  # Use ECN marks
    template_qdel: bool = False  # For copa like algos

    # desired
    desired_util_f: Union[float, z3.ArithRef]
    desired_queue_bound_multiplier: Union[float, z3.ArithRef]
    desired_loss_count_bound: Union[float, z3.ArithRef]
    desired_loss_amount_bound_multiplier: Union[float, z3.ArithRef]

    # environment
    infinite_buffer: bool = False
    buffer_size_multiplier: float = 1  # Used if dynamic_buffer = False
    dynamic_buffer: bool = False

    deterministic_loss: bool = True
    N: int = 1
    T: int = 9
    R: int = 1
    D: int = 1
    C: float = 100
    cca: str = "paced"
    compose: bool = True

    synth_ss: bool = False
    feasible_response: bool = False

    DEBUG: bool = False


class CegisCCAGen(Cegis):

    def __init__(
            self, generator_vars: List[z3.ExprRef],
            verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef], search_constraints: z3.ExprRef,
            definitions: z3.ExprRef, specification: z3.ExprRef,
            ctx: z3.Context, known_solution: Optional[z3.ExprRef] = None,
            metadata: Optional[CegisMetaData] = None,
            solution_log_path: Optional[str] = None):
        super(CegisCCAGen, self).__init__(
            generator_vars, verifier_vars, definition_vars,
            search_constraints, definitions, specification, ctx,
            known_solution, solution_log_path)
        self.metadata = metadata

    def remove_solution(self, solution: z3.ModelRef):
        """
        Used to remove solutions that vary only in the constant term
        """
        critical_generator_vars = self.generator_vars
        if(self.metadata and self.metadata.critical_generator_vars):
            critical_generator_vars = self.metadata.critical_generator_vars
        remove_solution(
            self.generator, solution, critical_generator_vars, self.ctx)

    # def get_counter_example_str(counter_example: z3.ModelRef,
    #                             verifier_vars: List[z3.ExprRef]):
    #     vdict: Dict[str, z3.ExprRef] = {}
    #     for v in verifier_vars:
    #         vdict[v.decl().name] = v
    #     vnames = list(vdict.keys())
    #     trunc_vnames = []
    #     for vname in vnames:
    #         regex = '[0-9]*$'
    #         if(vname)
    #     groups =
    #     return get_model_hash(counter_example, verifier_vars)
