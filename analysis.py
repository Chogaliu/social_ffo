import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

utility = np.load("tests/result_ana.npy", allow_pickle=True).item()
sign_activate = np.load("tests/result-2.npy", allow_pickle=True).item()

a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []
for i in range(606):
    na = 'u{}_60'.format(i)
    nb = 'u{}_67'.format(i)
    nc = 'u{}_149'.format(i)
    nd = 'u{}_448'.format(i)
    ne = 'u{}_455'.format(i)
    nf = 'u{}_459'.format(i)
    ng = 'u{}_460'.format(i)
    nh = 'u{}_474'.format(i)
    if na in utility:
        if utility[na] != 0.0:
            a.append(utility[na])
    if nb in utility:
        if utility[nb] != 0.0:
            b.append(utility[nb])
    if nc in utility:
        if utility[nc] != 0.0:
            c.append(utility[nc])
    if nd in utility:
        if utility[nd] != 0.0:
            d.append(utility[nd])
    if ne in utility:
        if utility[ne] != 0.0:
            e.append(utility[ne])
    if nf in utility:
        if utility[nf] != 0.0:
            f.append(utility[nf])
    if ng in utility:
        if utility[ng] != 0.0:
            g.append(utility[ng])
    if nh in utility:
        if utility[nh] != 0.0:
            h.append(utility[nh])
# a = np.array(a)
# b = np.array(b)
# c = np.array(c)
# d = np.array(d)
# e = np.array(e)
# f = np.array(f)
# g = np.array(g)
#

plt.figure(figsize=(5, 5))
plt.xlabel('activated sign f')
plt.ylabel('utility(n)(f)')
plt.xlim(xmax=9, xmin=0)
plt.ylim(ymax=20, ymin=-20)
area = np.pi * 4 ** 2  # 点面积
# 画散点图
x = [1 for _ in range(len(a))] + [2 for _ in range(len(b))] + [3 for _ in range(len(c))] + [4 for _ in
                                                                                            range(len(d))] + [5 for _ in
                                                                                                              range(
                                                                                                                  len(e))] + [
        6 for _ in range(len(f))] + [7 for _ in range(len(g))] + [8 for _ in range(len(h))]
y = a + b + c + d + e + f + g + h

plt.scatter(x, y, s=area, alpha=0.4, c=y)
plt.scatter(range(1, 9), [np.average(np.array(a)),
                          np.average(np.array(b)),
                          np.average(np.array(c)),
                          np.average(np.array(d)),
                          np.average(np.array(e)),
                          np.average(np.array(f)),
                          np.average(np.array(g)),
                          np.average(np.array(h))], c='black')
plt.plot([0, 9], [0, 0], linewidth='0.5', color='black')
plt.show()
