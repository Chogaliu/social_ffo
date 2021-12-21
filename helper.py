"""
helper functions

Author: Qiujia Liu
Data: 25th August 2021
"""

import numpy as np
from gurobipy import *
from tqdm import tqdm


def intensity_cal(k, q, r):
    if r == 0:
        r = 1e-2
    e = k * q / (r ** 2)
    return e


def cal_var(variables, node, current_idir, guiding_dir):
    """
    calculate the absolute variation before and after the signage applying
    for each node
    contain the bi to obtain the absolute
    """
    b_x = (2 * variables['b{}{}'.format(node, 'x')] - 1)
    b_y = (2 * variables['b{}{}'.format(node, 'y')] - 1)
    var_x = b_x * (current_idir[0] - guiding_dir[0])
    var_y = b_y * (current_idir[1] - guiding_dir[1])
    return var_x, var_y


def cal_after(po_graph, node, influ_x_sign, influ_y_sign):
    """
    calculate the current state after the signage applying
    for each node
    """
    after_e_x = po_graph.nodes[node].e_x + influ_x_sign
    after_e_y = po_graph.nodes[node].e_y + influ_y_sign
    return after_e_x, after_e_y


def cal_confid(exiting_dir, after_e_x, after_e_y):
    """
    calculate the value of after_e on the exiting direction
    exiting_dir = np.array()
    """
    dir_ix = exiting_dir[0] / np.linalg.norm(exiting_dir)
    dir_iy = exiting_dir[1] / np.linalg.norm(exiting_dir)
    confidence = after_e_x * dir_ix + after_e_y * dir_iy
    return confidence


def get_guiding_i_dir(variables, sign):
    '''
    get the direction through s{}{}
    '''
    guiding_i_dir = np.array([sum(np.array([0, 0, -1, 1]) * np.array(list(variables['s{}{}'.format(sign, i)]
                                                                          for i in ['up', 'down', 'left', 'right']))),
                              sum(np.array([1, -1, 0, 0]) * np.array(list(variables['s{}{}'.format(sign, i)]
                                                                          for i in ['up', 'down', 'left', 'right'])))])
    return guiding_i_dir


def get_guiding_dir(po_graph, node, sign, args, guiding_i_dir):
    """
    to get the direction on specific node with influence weight
    """
    r = np.linalg.norm(
        np.array([po_graph.nodes[node].x - po_graph.nodes[sign].x, po_graph.nodes[node].y - po_graph.nodes[sign].y]))
    direction_x = intensity_cal(k=args.k, q=args.sign_q, r=r) * guiding_i_dir[0]
    direction_y = intensity_cal(k=args.k, q=args.sign_q, r=r) * guiding_i_dir[1]
    direction = np.array([direction_x, direction_y])
    return direction


def get_poten_signs(sign_loc, value=1):
    """
    get the list of potential signs generated from step 1
    sign_loc_info: dict
    return: list
    """
    return [k for k, v in sign_loc.items() if v == value]


def get_utility(angle, e):
    """
    calculate the utility of sign a to ped in region b
    """
    u = (-0.6 * e + 1) * (-5 * angle + 15) + 4 * e
    return u


# def find_the_fittest_exit(po_graph):
#     """
#     through exit_info and node_info find the fittest exit at the current state for each node
#     principle: THE CLOSEST ONE
#     return: fittest_exit
#     fittest_exit {node_id}=exit_id
#     """
#     fittest_exit = {}
#     exit_info = po_graph.exit_info
#     for node_id in range(len(po_graph.nodes)):
#         dis = [np.linalg.norm(
#             np.array(exit_info[exit_id][1:3]) - np.array([po_graph.nodes[node_id].x, po_graph.nodes[node_id].y])) for
#             exit_id in
#             range(len(exit_info))]
#         fitting_exit_id = dis.index(min(dis))
#         fittest_exit[node_id] = fitting_exit_id
#     return fittest_exit

def find_the_fittest_dirs(po_graph):
    """
    through dirs and obs_info find the fittest exit at the current state for each node
    principle: according to dir of the closest network_node
    return: exiting_dirs {node_id}=dir_i
    """
    dirs = po_graph.dirs
    network_nodes = po_graph.network_nodes
    exiting_i_dirs = {}
    for node_id in range(len(po_graph.nodes)):
        node_loc = np.array([po_graph.nodes[node_id].x, po_graph.nodes[node_id].y])
        dist_min = float('inf')
        node_min = 'zero'
        for net_node_id in range(len(network_nodes)):
            net_node_loc = network_nodes[net_node_id]
            dist = np.linalg.norm(net_node_loc-node_loc)
            if dist < dist_min:
                if check_intersection_state(po_graph.obs_info, node_loc, net_node_loc):
                    continue
                dist_min = dist
                node_min = net_node_id
        exiting_i_dir = dirs[node_min]
        exiting_i_dirs[node_id] = exiting_i_dir
    return exiting_i_dirs


class Point(object):

    def __init__(self, x, y):
        self.x, self.y = x, y


class Vector(object):

    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y


def negative(vector):
    return Vector(vector.end, vector.start)


def vector_product(vectorA, vectorB):
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y


def is_intersected(A, B, C, D):
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)

    return (vector_product(AC, AD) * vector_product(BC, BD) <= 1e-9) \
           and (vector_product(CA, CB) * vector_product(DA, DB) <= 1e-9)


def check_intersection_state(obs_info, node_1, node_2):
    """
    used to check if the link between node_1 and node_2 have intersection with obs
    node_1/2, obs_info: array
    return:
    True: have intersection(s)
    False: No intersections
    """
    for obs in range(len(obs_info)):
        obs_x, obs_y = obs_info[obs][1:3]
        obs_w, obs_l = obs_info[obs][3:5]
        # test the intersection with all the obstacle
        obs_point_1_x, obs_point_1_y = obs_x, obs_y
        obs_point_2_x, obs_point_2_y = obs_x + obs_w, obs_y + obs_l
        obs_point_3_x, obs_point_3_y = obs_x, obs_y + obs_l
        obs_point_4_x, obs_point_4_y = obs_x + obs_w, obs_y
        is_intersect_ns_temp = is_intersected(Point(obs_point_1_x, obs_point_1_y),
                                              Point(obs_point_2_x, obs_point_2_y),
                                              Point(node_1[0], node_1[1]),
                                              Point(node_2[0], node_2[1])) \
                               or is_intersected(Point(obs_point_3_x, obs_point_3_y),
                                                 Point(obs_point_4_x, obs_point_4_y),
                                                 Point(node_1[0], node_1[1]),
                                                 Point(node_2[0], node_2[1]))
        if is_intersect_ns_temp:
            return True
    return False


def generate_dist_matrix_ns(po_graph, args):
    """
    generate the relationship between node and sign and save
    return: dist_matrix_ns[node, sign]=0/1 1:the sign can be seen by node
    """
    num_node = len(po_graph.nodes)
    obs_info = po_graph.obs_info
    dist_matrix_ns = np.ones((num_node, num_node))
    for node in tqdm(range(num_node - 1)):
        for sign in range(node + 1, num_node):
            node_loc = np.array([po_graph.nodes[node].x, po_graph.nodes[node].y])
            sign_loc = np.array([po_graph.nodes[sign].x, po_graph.nodes[sign].y])
            dis_ns = np.linalg.norm(node_loc - sign_loc)
            if dis_ns > args.per_dis:
                dist_matrix_ns[node, sign] = 0
                dist_matrix_ns[sign, node] = 0
            else:
                # for the potential signage position in perception area
                if check_intersection_state(obs_info, node_loc, sign_loc):
                    dist_matrix_ns[node, sign] = 0
                    dist_matrix_ns[sign, node] = 0
    np.save(args.filename_1_result_2, dist_matrix_ns)
    return dist_matrix_ns


def pooling_within(obs_nodes_with_id, gap_min=1.42):
    """
    input obs_nodes_with_id array [id,x,y]
    monitor the close nodes within same obstacle and combine them into one node by taking the middle point
    in other word, if the same pooled point achieved in the same obstacle twice, delete one
    return: pooled nodes with minimum gap limit
    """
    size_ = np.shape(obs_nodes_with_id)[0]
    dele_ = []
    for i in range(size_ - 1):
        for j in range(i + 1, size_):
            if obs_nodes_with_id[i][0] == obs_nodes_with_id[j][0]:
                dis = np.linalg.norm(obs_nodes_with_id[i][1:3] - obs_nodes_with_id[j][1:3])
                if dis > gap_min:
                    continue
                mid_point = (obs_nodes_with_id[i][1:3] + obs_nodes_with_id[j][1:3]) / 2
                obs_nodes_with_id[i][1:3] = mid_point
                obs_nodes_with_id[j][1:3] = mid_point
                dele_.append(j)
    print(dele_)
    return np.delete(obs_nodes_with_id, dele_, axis=0)


def pooling_combine_id(pooled_nodes_with_id, gap_min=1.42):
    """
    input obs_nodes_with_id array [id,x,y]
    Scene: point is infeasible due to the close distance to pass through
    monitor the close nodes from different obstacles and label them into the obstacle id
    return: pooled_nodes_ids
    """
    size_ = np.shape(pooled_nodes_with_id)[0]
    pooled_nodes_ids = {}
    for i in range(size_):
        pooled_nodes_ids[i] = []
        for j in range(size_):
            dis = np.linalg.norm(pooled_nodes_with_id[i][1:3] - pooled_nodes_with_id[j][1:3])
            if dis <= gap_min:
                pooled_nodes_ids[i].append(pooled_nodes_with_id[j][0])
    return pooled_nodes_ids


def pooling(obs_nodes_with_id):
    # pooling_within may not necessary
    pooled_nodes_with_id = pooling_within(pooling_within(obs_nodes_with_id))
    pooled_nodes_ids = pooling_combine_id(pooled_nodes_with_id)
    return pooled_nodes_with_id, pooled_nodes_ids


def get_poten_net_nodes(pooled_nodes_with_id, pooled_nodes_ids, obs_info):
    """
    input pooled obs_nodes_with_id array [id,x,y] generated from pooling
    generate the middle point for each nodes pair
    (nodes pairs are not allowed within the same obstacle or intersection with obstacles)
    (nodes pairs only generated from/with the node with one obstacle id)
    return: potential nodes for network [x,y]
    """
    size_ = np.shape(pooled_nodes_with_id)[0]
    poten_net_nodes = []
    appeared = []
    for i in range(size_ - 1):

        # # alter 1:  57*57
        # if len(pooled_nodes_ids[i]) > 1:
        #     continue
        # for j in range(i + 1, size_):
        #     if len(pooled_nodes_ids[j]) > 1:
        #         continue

        # alter 2:  135*135
        for j in range(i + 1, size_):
            if 1 not in [len(pooled_nodes_ids[i]), len(pooled_nodes_ids[j])]:
                continue

            if list(set(pooled_nodes_ids[i]) & set(pooled_nodes_ids[j])):
                continue
            if str(pooled_nodes_ids[i]) + ';' + str(pooled_nodes_ids[j]) in appeared or \
                    str(pooled_nodes_ids[j]) + ';' + str(pooled_nodes_ids[i]) in appeared:
                continue
            appeared.append(str(pooled_nodes_ids[i]) + ';' + str(pooled_nodes_ids[j]))
            node_i = pooled_nodes_with_id[i][1:3]
            node_j = pooled_nodes_with_id[j][1:3]
            non_temp = list(map(int, list(set(pooled_nodes_ids[i]) | set(pooled_nodes_ids[j]))))
            obs_info_temp = np.delete(obs_info, non_temp, axis=0)
            if check_intersection_state(obs_info_temp, node_i, node_j):
                continue
            mid_point = (node_i + node_j) / 2
            # Problem remain: overlap with obstacle
            poten_net_nodes.append(mid_point)
    return np.array(poten_net_nodes)


def get_net_links(obs_info, poten_net_nodes, args):
    """
    poten_net_nodes: array
    obs_info: array
    exit_info: array
    generate the network link and nodes
    with the consideration of feasible links (no obstacle intersection)
    and no non-feasible_link node
    return: matrix 0/1 nodes-nodes and feasible nodes location
    """
    # poten_net_nodes = np.vstack((poten_net_nodes, exit_info[:, 1:3]))
    size_ = np.shape(poten_net_nodes)[0]
    matrix_temp = np.zeros((size_, size_))
    for i in tqdm(range(size_ - 1)):
        for j in range(i + 1, size_):
            node_i = poten_net_nodes[i]
            node_j = poten_net_nodes[j]
            matrix_temp[i, j] = 1
            matrix_temp[j, i] = 1
            if check_intersection_state(obs_info, node_i, node_j):
                matrix_temp[i, j] = 0
                matrix_temp[j, i] = 0
    dele_ = []
    for i in range(size_):
        if sum(matrix_temp[i]) == 0:
            dele_.append(i)
    matrix_temp = np.delete(matrix_temp, dele_, axis=0)
    matrix = np.delete(matrix_temp, dele_, axis=1)
    nodes = np.delete(poten_net_nodes, dele_, axis=0)
    np.save(args.filename_3_result_2, nodes)
    return matrix, nodes


def optimize_lp_1(po_graph, args):
    """
    solve the .lp file
    return: sign_loc_info (settlement of possible signage locations)
    sign_loc_info x{}=0/1:
    """
    model = read(args.filename_1)
    model.optimize()
    print("Objective:", model.objVal)
    sign_loc_info = {}
    for i in model.getVars():
        # if 'x' in i.varname:
        print("Parameter:", i.varname, "=", i.x)
        index = int(i.varname[1:])
        sign_loc_info[index] = i.x
    po_graph.sign_loc_info = sign_loc_info
    np.save(args.filename_1_result, sign_loc_info)


def optimize_lp_2(po_graph, args):
    """
    solve the .lp file
    return: sign_info - activated
    sign_info s{}{}=0/1:
    """
    model = read(args.filename_2)
    model.optimize()
    print("Objective:", model.objVal)
    sign_activate = {}
    for i in model.getVars():
        print("Parameter:", i.varname, "=", i.x)
        if 's' in i.varname:
            sign_activate[i.varname] = i.x
    po_graph.sign_activate = sign_activate
    np.save(args.filename_2_result, sign_activate)

    return sign_activate
