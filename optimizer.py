"""
solve the model(.lp)

Author: Qiujia Liu
Data: 25th August 2021
"""

import cplex
from gurobipy import *

model = read('tests/test-1.lp')
model.optimize()
print("Objective:", model.objVal)
for i in model.getVars():
    print("Parameter:", i.varname, "=", i.x)
