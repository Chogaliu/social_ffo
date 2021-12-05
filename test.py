"""
for test
"""
import numpy as np
from helper import *

obs_info = np.array([(0, 10.5, 6, 1, 12),
                     (1, 25.5, 6, 1, 12),
                     (2, 11.5, 12.5, 3, 1),
                     (3, 24.5, 12.5, 3, 1),
                     (4, 40.5, 4, 1, 8,),
                     ])
obs_nodes = []  # list
for obs_id in obs_info[:, 0]:
    id_o = int(obs_id)
    x, y = obs_info[id_o][1:3]
    w, l = obs_info[id_o][3:5]
    obs_nodes.append((id_o, x - w / 2, y - l / 2))
    obs_nodes.append((id_o, x + w / 2, y - l / 2))
    obs_nodes.append((id_o, x - w / 2, y + l / 2))
    obs_nodes.append((id_o, x + w / 2, y + l / 2))
nodes_1 = pooling(obs_nodes_with_id=np.array(obs_nodes), gap_min=1.414)
nodes_2 = pooling(obs_nodes_with_id=nodes_1, gap_min=1.414)
print(nodes_2)
poten_nodes = get_poten_net_nodes(nodes_2)
print(np.shape(poten_nodes))




