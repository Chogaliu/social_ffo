"""
Reference: https://blog.csdn.net/Yuan52007298/article/details/80180839
Construct Dijkstra problem
return shortest_paths, dirs for network_nodes {node}=[path]

Author: Qiujia Liu
Data: 9th Dec 2021
"""
import numpy as np
from helper import check_intersection_state


class DIJKSTRA:
    """
    To define the dijkstra problem
    """

    def __init__(self, po_graph, args):
        # generate the network_nodes (including exits) and matrix with distance
        self.matrix = po_graph.network_matrix
        self.nodes = po_graph.network_nodes
        self.exit_info = po_graph.exit_info
        self.dijkstra_paths = {}
        self.dirs = {}

        # generate the cost with nodes(including exits) - graph
        num_nodes = len(self.nodes)
        num_exits = len(self.exit_info)
        # net_nodes including exits compared with nodes (po_graph.network_nodes)
        self.net_nodes = np.vstack((self.nodes, self.exit_info[:, 1:3]))
        num = len(self.net_nodes)
        _ = float('inf')
        graph = _ * np.ones((num, num))
        for i in range(num_nodes - 1):
            for j in range(i + 1, num_nodes):
                if self.matrix[i, j] == 1:
                    dis = np.linalg.norm(self.nodes[i] - self.nodes[j])
                    graph[i, j] = dis
                    graph[j, i] = dis
        for i in range(num_nodes):
            for e in range(num_exits):
                if not check_intersection_state(po_graph.obs_info, self.nodes[i], self.exit_info[e, 1:3]):
                    dis = np.linalg.norm(self.nodes[i] - self.exit_info[e, 1:3])
                    graph[i][num_nodes + e] = dis
        self.graph = graph
        self.cal_shortest()
        np.save(args.filename_3_result, self.dirs)

    def cal_shortest(self):
        """
        generate the minimum path to the exit (of given exits) with the given start point
        return:
        paths: [list] for each net_node
        dirs: [dir_array] for each net_node
        """
        _ = float('inf')
        points = len(self.net_nodes)
        map = self.graph
        for net_node_idx in range(len(self.nodes)):
            pre = [0] * points
            vis = [0] * points
            dis = [_ for i in range(points)]
            start = net_node_idx
            for i in range(points):
                if i == start:
                    dis[i] = 0
                else:
                    dis[i] = map[start][i]
                if map[start][i] != _:
                    pre[i] = start
                else:
                    pre[i] = -1
            vis[start] = 1
            for i in range(points):
                min_dist = _
                for j in range(points):
                    if vis[j] == 0 and dis[j] < min_dist:
                        t = j
                        min_dist = dis[j]
                vis[t] = 1
                for j in range(points):
                    if vis[j] == 0 and dis[j] > dis[t] + map[t][j]:
                        dis[j] = dis[t] + map[t][j]
                        pre[j] = t
            record_exit_dist = []
            record_exit_roads = []
            for end in range(len(self.nodes), points):
                road = [0] * points
                roads = []
                p = end
                l_en = 0
                while p >= 0 and l_en < points:
                    road[l_en] = p
                    p = pre[p]
                    l_en += 1
                l_en -= 1
                while l_en >= 0:
                    roads.append(road[l_en])
                    l_en -= 1
                record_exit_dist.append(dis[end])
                record_exit_roads.append(roads)
            dist_list = [int(i) for i in record_exit_dist]
            idx = dist_list.index(min(dist_list))
            self.dijkstra_paths[start] = record_exit_roads[idx]
            current_loc = self.nodes[start]
            next_loc = self.net_nodes[self.dijkstra_paths[start][1]]
            direction = next_loc - current_loc
            self.dirs[start] = direction / np.linalg.norm(direction)
