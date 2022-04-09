import audioop

import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm

utility = np.load("tests/result_ana.npy", allow_pickle=True).item()
sign_activate = np.load("tests/activate-cond.npy", allow_pickle=True).item()

activate_sign = []
for i, j in sign_activate.items():
    if j == 1:
        s = re.sub("\D", "", i)
        activate_sign.append(int(s))
activate_num = len(activate_sign)
temp = []
for i in range(activate_num):
    temp.append([])

u_node = []
u_sign = []
u_value = []
for i, j in utility.items():
    if "_" in i:
        a = re.sub("\D", " ", i)
        b = a.split(" ")
        u_node.append(int(b[1]))
        u_sign.append(int(b[2]))
        u_value.append(j)

for count in range(len(u_value)):
    if int(u_value[count]) != 0:
        n = activate_sign.index(u_sign[count])
        temp[n].append(u_value[count])

plt.figure(figsize=(5, 5))
plt.xlabel('activated sign f')
plt.ylabel('utility(n)(f)')
plt.xlim(xmax=activate_num+1, xmin=0)
plt.ylim(ymax=20, ymin=-20)
area = np.pi * 4 ** 2

x = []
y = []
for i in range(activate_num):
    x += [i + 1 for _ in range(len(temp[i]))]
    y += temp[i]
plt.scatter(x, y, s=area, alpha=0.4, c=y)
for i in range(activate_num):
    plt.scatter(i + 1, np.average(temp[i]), c='black')
plt.plot([0, activate_num+1], [0, 0], linewidth='0.5', color='black')
plt.show()
