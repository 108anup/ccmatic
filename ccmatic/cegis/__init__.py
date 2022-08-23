from cegis import Cegis


class CegisConfig:
    # template
    history: int = 4
    template_queue_bound: bool = True
    template_loss_oracle: bool = True  # Use ECN marks
    template_mode_switching: bool = False
    template_cca_lower_bound: float = 0.01

    # desired
    desired_util_f: float = 0.5
    desired_queue_bound_multiplier: float = 2
    desired_loss_bound: float = 3

    # environment
    deterministic_loss: bool = True
    dynamic_buffer: bool = False
    buffer_size_multiplier: float = 1  # Used if dynamic_buffer = False
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
