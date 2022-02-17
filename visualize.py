import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [4, 6, 8, 10, 12, 14, 16]
y = [1.6e-10, 159.69, 611.16, 1469.21, 2419.16, 3332.162, 4050.358]
y1 = [2, 4, 7, 7, 9, 9, 9]
# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)
fig, ax1 = plt.subplots()
ax1.plot(x, y, marker='o', mec='r', mfc='w')
ax1.set_ylabel("Utility")
ax2 = ax1.twinx()  # twinx将ax1的X轴共用与ax2，这步很重要
ax2.plot(x, y1, marker='*', ms=10, c='orange')
ax2.set_ylabel("Number of activated signs")
ax1.set_xlabel("m")
ax1.legend(["Utility"],loc=4)
ax2.legend(["Activated signs number"],loc=2)
plt.show()


