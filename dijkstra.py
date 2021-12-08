"""
Reference:https://blog.csdn.net/weixin_40159138/article/details/90693581
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
        self.obs_info = po_graph.obs_info
        # generate the cost with nodes
        num_nodes = len(self.nodes)
        self.graph = {}
        for i in range(num_nodes - 1):
            for j in range(i + 1, num_nodes):
                if self.matrix[i, j] == 1:
                    dis = np.linalg.norm(self.nodes[i] - self.nodes[j])
                    graph["i"]["j"] = dis
                    graph["j"]["i"] = dis


    def dijkstra_run():
        # adding node and the corresponding (feasible) links into the graph
        node = find_lowest_cost_node(costs)  # 在未处理的节点中找到开销最小的节点
        while node is not None:  # 所有节点都被处理过，node为None，循环结束
            cost = costs[node]
            neighbors = graph[node]
            for n in neighbors.keys():  # 遍历当前节点的所有邻居
                new_cost = cost + neighbors[n]
                if costs[n] > new_cost:  # 如果经当前节点前往该邻居更近
                    costs[n] = new_cost  # 就更新该邻居的开销
                    parents[n] = node  # 同时将该邻居的父节点设置为当前节点
            processed.append(node)  # 将当前节点标记为处理过
            node = find_lowest_cost_node(costs)  # 找出接下来要处理的节点，并循环
        shortest_path = find_shortest_path()
        print(shortest_path)


graph["exit_1"] = {}
graph["exit_1"] = {}
graph["exit_1"] = {}

# 创建开销/时间表
infinity = float("inf")  # 无穷大
costs = {}
costs["a"] = 6
costs["b"] = 2
costs["fin"] = infinity  # 暂时将通往终点的时间，定义为无穷大

# 路径中父节点信息
parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["fin"] = None

# 记录处理过的节点的数组
processed = []


# 定义寻找最小节点的函数
def find_lowest_cost_node(costs):
    lowest_cost = float("inf")
    lowest_cost_node = None
    for node in costs:  # 遍历所有节点
        cost = costs[node]
        if cost < lowest_cost and node not in processed:  # 如果当前节点的开销更低且未处理过
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node


# 寻找最短路径
def find_shortest_path():
    node = "fin"
    shortest_path = ["fin"]
    while parents[node] != "start":
        shortest_path.append(parents[node])
        node = parents[node]
    shortest_path.append("start")
    shortest_path.reverse()  # 将从终点到起点的路径反序表示
    return shortest_path
