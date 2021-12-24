"""
Used to print the Utility Variation with (E, Cos~Angle)

Author: Qiujia Liu
Data: 2nd Dec 2021
"""

import numpy as np
from typing import List
from helper import *
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

angle = np.arange(-1, 1, 0.1)
e = np.arange(0, 100, 5)
angle, e = np.meshgrid(angle, e)
u = get_utility(x=angle, y=e)
fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(e, angle, u, color='black', linewidth=0, alpha=0.3, antialiased=False)

a = ax.contour(e, angle, u, zdir='x', stride=1, offset=0, cmap=cm.viridis)
b = ax.contour(e, angle, u, zdir='y', stride=1, offset=1, cmap=cm.viridis)
c = ax.contour(e, angle, u, zdir='z', stride=8, offset=-20, cmap=cm.viridis)

ax.set_xlabel('e(n)')
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 100, 20))
ax.set_ylabel('cos<d(f)∙d(n)>')
ax.set_ylim(-1, 1)
ax.set_yticks(np.arange(-1, 1, 1))
ax.set_zlabel('Utility(n)(f)')
ax.set_zlim(-20, 20)
ax.set_zticks(np.arange(-20, 20, 5))
#
# plt.colorbar(a)

plt.show()
