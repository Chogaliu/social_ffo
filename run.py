"""
Operating script for the Social Force-Field-Optimization model
and output the optimization model (.lp) and solve it

Author: Qiujia Liu
Data: 20th August 2021
"""
import argparse
import sys
from po_graph import PO_GRAPH
import random
import numpy as np
from gurobipy import *
from helper import *
from tqdm import tqdm
from dijkstra import DIJKSTRA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wide', type=int, default=50)
    parser.add_argument('--length', type=int, default=50)
    parser.add_argument('--gap', type=float, default=2)
    parser.add_argument('--A', type=float, default=2000, )
    parser.add_argument('--B_w', type=float, default=20,
                        help='the influence bring by danger')
    parser.add_argument('--B_sd', type=float, default=0.1,)
    parser.add_argument('--B', type=float, default=0.08, )
    parser.add_argument('--conf', type=float, default=20,
                        help='the low-bound of confidence')
    parser.add_argument('--per_dis', type=float, default=5.0,
                        help='the sign influ range of pedestrian')
    parser.add_argument('--h_range', type=float, default=10.0,
                        help='the herding influ range')
    parser.add_argument('--c_range', type=float, default=20,
                        help='the congestion influ range (not required)')
    parser.add_argument('--filename_1', type=str, default="tests/test-1.lp")
    parser.add_argument('--filename_2', type=str, default="tests/test-2.lp")
    parser.add_argument('--filename_3', type=str, default="tests/test-3.lp")
    parser.add_argument('--filename_1_result', type=str, default="tests/result-1.npy", help='sign_loc_info')
    parser.add_argument('--filename_1_result_2', type=str, default="tests/result-1-2.npy", help='dist_matrix_ns')
    parser.add_argument('--filename_2_result', type=str, default="tests/result-2.npy", help='result of step 2')
    parser.add_argument('--filename_3_result', type=str, default="tests/result-3.npy", help='dirs')
    parser.add_argument('--filename_3_result_2', type=str, default="tests/result-3-2.npy", help='network_nodes')
    args = parser.parse_args()

    # # For parameter calibration
    # po_graph = initialize(args)
    # po_graph.printGraph(field_show=True, enviro_show=True)

    # The optimization process is divided into two steps:

    # First step: (1) generate the possible locations of signage (2): generate the exiting_dir
    po_graph = initialize(args)
    # 1)
    write_lp_1(po_graph, args)
    optimize_lp_1(po_graph, args)
    po_graph.printGraph(field_show=True, enviro_show=True)
    # 2)
    po_graph.read_net(args)
    dijkstra = DIJKSTRA(po_graph, args)
    # po_graph.printOpti(enviro_show=False, dijkstra=dijkstra)
    # po_graph.printNetwork(net_show=True, enviro_show=False, dijkstra=False, dijkstra_path_only=False)
    # po_graph.printNetwork(net_show=True, enviro_show=False, dijkstra=dijkstra, dijkstra_path_only=False)

    # Second step: activate the necessary signage
    po_graph = initialize(args)
    po_graph.read_pre_results(args)
    write_lp_2(po_graph, args)
    optimize_lp_2(po_graph, args)
    po_graph.printGraph(field_show=True, enviro_show=True)

    # # update the e on the po_graph
    # po_graph.read_SigntoField(args.k, args.sign_q)


def initialize(args):
    """
    po_graph initialize: load the information and influence of ped, obs, exit, danger, (sign_loc_info)
    obs_info requirements:
    1/ the left-bottom point and the width&length
    2/ in rectangle shape
    3/ must be continuous without intersection by other obstacles
    """

    # # one condition (assumption)
    # obs_info = [(0, 10, 0, 1, 12, args.obs_q),
    #             (1, 25, 0, 1, 12, args.obs_q),
    #             (2, 10, 12, 3, 1, args.obs_q),
    #             (3, 23, 12, 3, 1, args.obs_q),
    #             (4, 40, 0, 1, 8, args.obs_q),
    #             (5, 0, 27, 20, 1, args.obs_q),
    #             (6, 15, 40, 1, 10, args.obs_q),
    #             (7, 15, 39, 12, 1, args.obs_q),
    #             (8, 40, 20, 1, 10, args.obs_q),
    #             (9, 28, 30, 12, 1, args.obs_q),
    #             # walls:
    #             (10, -1, 27, 1, 24, args.obs_q),
    #             (11, -1, 7, 1, 14, args.obs_q),
    #             (12, 0, -1, 10, 1, args.obs_q),
    #             (13, 30, -1, 10, 1, args.obs_q),
    #             (14, 50, -1, 1, 20, args.obs_q),
    #             (15, 50, 31, 1, 20, args.obs_q),
    #             (16, 0, 50, 15, 1, args.obs_q),
    #             (17, 30, 50, 10, 1, args.obs_q),
    #             (18, 40, -1, 10, 1, args.obs_q),
    #             (19, 40, 50, 10, 1, args.obs_q),
    #             (20, 10, -1, 10, 1, args.obs_q),
    #             (21, 15, 50, 5, 1, args.obs_q),
    #             (22, 40, 30, 1, 20, args.obs_q)
    #             ]

    # second condition (Museum of art PuDong 1F)
    obs_info = [(0, 0, 0, 0.7, 22),
                (1, 0.7, 20.7, 9.3, 1.3),
                (2, 10.7, 20.7, 14.6, 1.3),
                (3, 26, 20.7, 7.3, 1.3),
                (4, 10, 11.4, 0.7, 10.6),
                (5, 25.3, 16.7, 0.7, 5.3),
                (6, 25.3, 11.4, 0.7, 3.3),
                (7, 6.7, 0, 0.7, 10.7),
                (8, 10, 10.7, 4, 0.7),
                (9, 16, 10.7, 4, 0.7),
                (10, 10, 7.4, 0.7, 3.3),
                (11, 22, 10.7, 4, 0.7),
                (12, 10, 0, 0.7, 5.4),
                (13, 7.4, 0, 2.6, 2),
                (14, 10.7, 0, 14.6, 2),
                (15, 25.3, 0, 0.7, 10.7),
                (16, 26, 0, 4, 2),
                (17, 28, 5.3, 8.6, 0.7),
                (18, 35.3, 0, 1.3, 5.3),
                (19, 35.3, 6, 1.3, 16),
                ]

    # need more information extracted from trajectory
    ped_info = [(0, 7.5, 8.5, 0, 1),
                (1, 3.5, 5.5, 0, 1),
                (2, 15.5, 5.5, 1, 0),
                (3, 15.5, 12.4, -1, 0),
                (4, 3.5, 3.5, -1, 0)]
    # casual distribution of pedestrian no intersection with obstacles and in the range
    # num_ped =
    # ped_info = []
    # for n in range(num_ped):
    #     ped_info.append(n, x, y, args.ped_q)

    exit_info = [(0, 3.7, -1),
                 (1, 32.6, -1),
                 (2, 34.3, 23)]

    danger_info = [(0, 1, 10, 1, 3),
                   (1, 24, 16, 1, 3),
                   (2, 32, 7, 3, 1)
                   ]

    # generate Po_graph
    po_graph = PO_GRAPH(dim_w=36.6, dim_l=22, gap=1)
    po_graph.read_ObstoField(obs=obs_info, A=args.A, B=args.B)
    po_graph.read_ODtoField(danger=danger_info, exit=exit_info, B_w=args.B_w, B_sd=args.B_sd)
    po_graph.read_PedtoField(ped=ped_info, A=args.A, B=args.B, herd_range=args.h_range)
    return po_graph


def write_lp_1(po_graph, args):
    """
    generate the locations of the signage settlements (activated and inactivated)
    with the consideration of evacuees' cognitive range and obstacle information
    return: the .lp file
    """
    m = Model("A")

    # variables:
    num_node = len(po_graph.nodes)
    x_name = ['x{}'.format(sign) for sign in range(num_node)]  # if the sign is adopted as the potential signage 0/1
    # b_name = ['b{}{}'.format(node, sign) for node in range(num_node) for sign in range(num_node)]
    # 1/0 if the sign(s) can be detected by person at node [we require at least one sign can be detected]

    variables = {}
    for xi in x_name:
        variables[xi] = m.addVar(vtype=GRB.BINARY, name=xi)
    # for bi in b_name:
    #     variables[bi] = m.addVar(vtype=GRB.BINARY, name=bi)
    m.update()

    # objective:
    m.setObjective(sum((variables['x{}'.format(i)] for i in range(num_node))), GRB.MINIMIZE)

    # constraints:
    # construct the matrix to show the intersection condition between node and sign: dis_ns-ok inf-notok
    dist_matrix_ns = generate_dist_matrix_ns(po_graph, args)
    for node in range(num_node):
        # for sign in range(num_node):
        # m.addConstr(
        #     (args.per_dis - dist_matrix_ns[node, sign]) * (2 * variables['b{}{}'.format(node, sign)] - 1) >= 0,
        #     'c_abs{}{}'.format(node, sign))
        m.addConstr(sum(dist_matrix_ns[node, sign]
                        * variables['x{}'.format(sign)] for sign in range(num_node)) >= 1, 'cons_signs{}'.format(node))

    # write in
    m.write(filename=args.filename_1)


def write_lp_2(po_graph, args):
    """
    generate the optimization model with po_graph and save it as 'file'
    return: the .lp file
    Problem: no consideration of exit selection & herding
    """
    m = Model("B")

    # variables
    num_node = len(po_graph.nodes)
    poten_signs = get_poten_signs(po_graph.sign_loc_info)

    # extract the info about relationship between sign and node
    influ_signs_for_n = {}
    for node in range(num_node):
        influ_signs_for_n[node] = []
    influ_nodes_for_s = {}
    for sign in poten_signs:
        influ_nodes_for_s[sign] = []
    u_name = []
    for node in range(num_node):
        for sign in poten_signs:
            if po_graph.dist_matrix_ns[node, sign] == 1:
                influ_signs_for_n[node].append(sign)
                influ_nodes_for_s[sign].append(node)
                u_name.append('u{}_{}'.format(node, sign))

    variables = {}
    x_name = ['s{}{}'.format(i, j) for i in poten_signs for j in ['up', 'down', 'left', 'right']]
    for xi in x_name:
        variables[xi] = m.addVar(vtype=GRB.BINARY, name=xi)
    # b_name = ['b{}{}'.format(i, j) for i in range(num_node) for j in ['x', 'y']]  # adjust negative/positive
    # for bi in b_name:
    #     variables[bi] = m.addVar(vtype=GRB.BINARY, name=bi)
    for ui in u_name:
        variables[ui] = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=ui)
    # after create the variables we need to update in order to apply them in constraints
    m.update()

    # objective
    m.setObjective(sum(variables[ui] for ui in u_name), GRB.MAXIMIZE)

    # constraints
    exiting_i_dirs = find_the_fittest_dirs(po_graph)
    for node in range(num_node):
        current_i_dir = np.array([po_graph.nodes[node].e_ix, po_graph.nodes[node].e_iy])
        exiting_i_dir = exiting_i_dirs[node]

        for sign in influ_signs_for_n[node]:
            sign_activate = sum(variables['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right'])
            guiding_i_dir = get_guiding_i_dir(variables, sign)
            # guiding_dir = get_guiding_dir(po_graph, node, sign, args, guiding_i_dir)
            angle_matter_u = sum(current_i_dir * guiding_i_dir)  # [-1,1]
            # angle_matter_u = sum(current_i_dir * exiting_i_dir)  # [-1,1]
            e_matter_u = po_graph.nodes[node].e  # [?]

            # constraint 1 -- direction requirement - to promise the correctness of the guiding direction
            # Problem: default the angle between exiting direction and guiding direction must be acute ?
            m.addConstr(sum(exiting_i_dir * guiding_i_dir) * sign_activate >= 0, 'efficiency{}{}'.format(node, sign))

            # constraint 4 -- for utility calculation
            m.addConstr(
                variables['u{}_{}'.format(node, sign)] <= get_utility(angle_matter_u, e_matter_u) * sign_activate,
                'u_up_limit{}{}'.format(node, sign))
            m.addConstr(
                variables['u{}_{}'.format(node, sign)] >= get_utility(angle_matter_u, e_matter_u) * sign_activate,
                'u_low_limit{}{}'.format(node, sign))

    for sign in poten_signs:
        sign_activate = sum(variables['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right'])

        # constraint 2 -- general - limit the direction choices
        m.addConstr(sign_activate <= 1, 'sign{}'.format(sign))

        # constraint 3 -- utility requirement - promise each sign excess minimum positive influence
        # Problem: + 1 : low-bound of the sign utility
        m.addConstr(sum(variables['u{}_{}'.format(node, sign)] for node in influ_nodes_for_s[sign]) + 1 >= 0,
                    'u_node{}'.format(sign))

        # # constraint 2 -- general - absolute promise of the variation of E:
        # var_x, var_y = cal_var(variables, node, current_i_dir, guiding_dir)
        # m.addConstr(var_x >= 0, 'abs_x{}'.format(node))
        # m.addConstr(var_y >= 0, 'abs_y{}'.format(node))

        # constraint 4 -- confidence requirement - promise the evacuee has the confidence on exiting direction:
        # Problem: may repeat with constraint 3
        # after_e_x, after_e_y = cal_after(po_graph, node, influ_x_sign, influ_y_sign)
        # confidence = cal_confid(exiting_dir, after_e_x, after_e_y)
        # m.addConstr(confidence >= args.conf, 'confidence{}'.format(node))

    # write in
    m.write(filename=args.filename_2)


if __name__ == '__main__':
    main()
