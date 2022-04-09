"""
for test
"""

import numpy as np
from helper import *
from itertools import chain

temp = [[1, 2], [3, 5], [7, 8], [4, 5, 6]]
activate_num = 4
x = [
]
y = [

]
for i in range(activate_num):
    x += [i+1 for _ in range(len(temp[i]))]
    y += temp[i]

print(x)
print(y)

