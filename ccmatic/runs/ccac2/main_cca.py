from ccac2.config import Config
from ccac2.model import ModelVariables, all_constraints
from pyz3_utils.my_solver import MySolver


c = Config()
c.unsat_core = False
c.T = 10
c.F = 2
c.inf_buf = False
c.check()
s = MySolver()
v = ModelVariables(c, s)
all_constraints(c, s, v)

s.add(v.times[-1].time >= 5)
for t in range(c.T):
    s.add(v.times[t].flows[0].cwnd == 1)
    s.add(v.times[t].flows[0].rate == 0.5)

sat = s.check()
print(str(sat))