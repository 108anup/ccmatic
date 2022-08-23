from typing import Union

import z3

from cegis import Cegis


class CegisConfig:
    # template
    history: int = 4
    template_cca_lower_bound: float = 0.01
    template_queue_bound: bool = False
    template_mode_switching: bool = False
    template_loss_oracle: bool = True  # Use ECN marks

    # desired
    desired_util_f: Union[float, z3.ExprRef]
    desired_queue_bound_multiplier: Union[float, z3.ExprRef]
    desired_loss_bound: Union[float, z3.ExprRef]

    # environment
    infinite_buffer: bool = False
    buffer_size_multiplier: float = 1  # Used if dynamic_buffer = False
    dynamic_buffer: bool = False

    deterministic_loss: bool = True
    N: int = 1


class CegisCCAGen(Cegis):

    pass

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
