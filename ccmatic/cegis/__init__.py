from typing import List, Dict

import z3

from cegis import Cegis


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
