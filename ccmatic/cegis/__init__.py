from dataclasses import dataclass
from typing import Union, List, Optional

import z3

from cegis import Cegis, remove_solution


@dataclass
class CegisMetaData:
    critical_generator_vars: Optional[List[z3.ExprRef]]


class CegisConfig:
    name: Optional[str] = None

    # template
    history: int = 4
    template_cca_lower_bound: float = 0.01
    template_queue_bound: bool = False
    template_mode_switching: bool = False
    template_loss_oracle: bool = True  # Use ECN marks
    template_qdel: bool = False  # For copa like algos
    template_fi_reset: bool = False
    # ^^ For infrequent losses with fast increase, revert to last cwnd on loss.
    template_beliefs: bool = False
    template_beliefs_use_buffer: bool = False

    # desired
    desired_util_f: Union[float, z3.ArithRef]
    desired_queue_bound_multiplier: Union[float, z3.ArithRef]
    desired_queue_bound_alpha: Union[float, z3.ArithRef] = 0
    desired_loss_count_bound: Union[float, z3.ArithRef]
    desired_large_loss_count_bound: Union[float, z3.ArithRef] = 0
    desired_loss_amount_bound_multiplier: Union[float, z3.ArithRef]
    desired_loss_amount_bound_alpha: Union[float, z3.ArithRef] = 0

    desired_fast_decrease: bool = False
    desired_fast_increase: bool = False

    rate_or_window: str = 'default'
    # choices: ['default', 'rate', 'window']

    ideal_link: bool = False

    # environment
    infinite_buffer: bool = False
    buffer_size_multiplier: float = 1  # Used if dynamic_buffer = False
    dynamic_buffer: bool = False
    app_limited: bool = False
    fix_stale__max_c: bool = False
    fix_stale__min_c: bool = False

    deterministic_loss: bool = True
    N: int = 1
    T: int = 9
    R: int = 1
    D: int = 1
    C: float = 100
    cca: str = "paced"
    compose: bool = True

    synth_ss: bool = False
    use_belief_invariant: bool = False
    feasible_response: bool = False

    # For assumption synthesis
    use_ref_cca: bool = False
    monotonic_inc_assumption: bool = False

    DEBUG: bool = False

    def desire_tag(self) -> str:
        items = [
            'desired_util_f', 'desired_queue_bound_multiplier',
            'desired_queue_bound_alpha', 'desired_loss_count_bound',
            'desired_large_loss_count_bound',
            'desired_loss_amount_bound_multiplier',
            'desired_loss_amount_bound_alpha']
        item_strs = []
        for item in items:
            item_strs.append(f"{item}={getattr(self, item)}")
        return ", ".join(item_strs)

    def reset_desired_z3(self, pre: str):
        self.desired_util_f = \
            z3.Real(f'{pre}Desired__util_f')
        self.desired_queue_bound_multiplier = \
            z3.Real(f'{pre}Desired__queue_bound_multiplier')
        self.desired_queue_bound_alpha = \
            z3.Real(f'{pre}Desired__queue_bound_alpha')
        self.desired_loss_count_bound = \
            z3.Real(f'{pre}Desired__loss_count_bound')
        self.desired_large_loss_count_bound = \
            z3.Real(f'{pre}Desired__large_loss_count_bound')
        self.desired_loss_amount_bound_multiplier = \
            z3.Real(f'{pre}Desired__loss_amount_bound')
        self.desired_loss_amount_bound_alpha = \
            z3.Real(f'{pre}Desired__loss_amount_alpha')


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
            self.generator, solution, critical_generator_vars,
            self.ctx, self._n_proved_solutions)

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
