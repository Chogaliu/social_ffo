"""
Reference: https://blog.csdn.net/Yuan52007298/article/details/80180839
"""
import numpy as np
from helper import check_intersection_state


class DIJKSTRA:
    """
    To define the dijkstra problem
    """

    def __init__(self, po_graph):
        self.matrix = po_graph.network_matrix
        self.nodes = po_graph.network_nodes
        self.exit_info = po_graph.exit_info

        # generate the cost with nodes(including exits) - graph
        self.num_nodes = len(self.nodes)
        num_nodes = self.num_nodes
        self.num_exits = len(self.exit_info)
        num_exits = self.num_exits
        self.num = self.num_nodes + self.num_exits
        _ = float('inf')
        graph = _ * np.ones((self.num, self.num))
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

    def cal_shortest(self, net_node_idx):
        """
        points点个数，edges边个数,graph路径连通图,start七点,end终点
        def Dijkstra(points, edges, graph, start, end):
        """
        _ = float('inf')
        points = self.num
        pre = [0] * (points + 1)  # 记录前驱
        vis = [0] * (points + 1)  # 记录节点遍历状态
        dis = [_ for i in range(points + 1)]  # 保存最短距离
        map = self.graph
        start = net_node_idx
        for i in range(points + 1):  # 初始化起点到其他点的距离
            if i == start:
                dis[i] = 0
            else:
                dis[i] = map[start][i]
            if map[start][i] != _:
                pre[i] = start
            else:
                pre[i] = -1
        vis[start] = 1
        for i in range(points + 1):  # 每循环一次确定一条最短路
            min = _
            for j in range(points + 1):  # 寻找当前最短路
                if vis[j] == 0 and dis[j] < min:
                    t = j
                    min = dis[j]
            vis[t] = 1  # 找到最短的一条路径 ,标记
            for j in range(points + 1):
                if vis[j] == 0 and dis[j] > dis[t] + map[t][j]:
                    dis[j] = dis[t] + map[t][j]
                    pre[j] = t
        record_exit_dist = []
        record_exit_roads = []
        for end in range(self.num_nodes + 1, self.num):
            road = [0] * (points + 1)  # 保存最短路径
            roads = []
            p = end
            len = 0
            while p >= 1 and len < points:
                road[len] = p
                p = pre[p]
                len += 1
            mark = 0
            len -= 1
            while len >= 0:
                roads.append(road[len])
                len -= 1
            record_exit_dist.append(dis[end])
            record_exit_roads.append(roads)
        min_dist = min(record_exit_dist)
        idx = record_exit_dist.index(min_dist)
        return record_exit_roads[idx]
