"""
PO-graph data structure script for the structural optimizer.input
Takes the scene information and pedestrian information
P - pedestrian (default: no mask)
O - obstacle (default: in "square" shape)

Author: Qiujia Liu
Data: 25th August 2021
"""

import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import math
import shutil
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator
import time
from helper import *


class PO_GRAPH:
    """
    To show how obstacle and A pedestrian and B pedestrian influence each other in the form of Field change
    generate intensity map
    """

    def __init__(self, dim_w, dim_l, gap):
        """
        dim_w/l: the dimension of the space
        gap: the dimension of the discretization
        return: nodes [x,y,e_x,e_y]
        """
        self.dim_w = dim_w
        self.dim_l = dim_l
        self.gap = gap
        self.wid = int(dim_w / gap) + 1
        self.len = int(dim_l / gap) + 1
        self.obs_info = 0  # array
        self.ped_info = 0  # array
        self.exit_info = 0  # array
        self.danger_info = 0  # array
        self.sign_activate = 0  # dict
        self.sign_loc_info = 0  # dict
        self.dist_matrix_ns = 0  # array
        self.dirs = 0  # dict
        self.network_matrix = 0  # array
        self.network_nodes = 0  # array
        self.grid_size = self.wid * self.len

        self.nodes = [PO_NODE(node % self.wid * self.gap, node // self.wid * self.gap, 0, 0)
                      for node in range(self.grid_size)]
        # self.edges = [{} for i in range(self.grid_size)]

    def reset(self):
        self.nodes = [PO_NODE(node % self.wid * self.gap, node // self.wid * self.gap, 0, 0)
                      for node in range(self.grid_size)]
        # self.edges = [{} for i in range(self.grid_size)]

    def read_pre_results(self, args, activate=False):
        """
        read the sign_loc_info & (exiting) dirs generated from step 1 to po_graph
        return: the updated po_graph
        """
        sign_loc_info = np.load(args.filename_1_result, allow_pickle=True).item()
        dist_matrix_ns = np.load(args.filename_1_result_2, allow_pickle=True)
        dirs = np.load(args.filename_3_result, allow_pickle=True).item()
        network_nodes = np.load(args.filename_3_result_2, allow_pickle=True)
        self.sign_loc_info = sign_loc_info
        self.dist_matrix_ns = dist_matrix_ns
        self.dirs = dirs
        self.network_nodes = network_nodes
        if activate:
            sign_activate = np.load(args.filename_2_result, allow_pickle=True).item()
            self.sign_activate = sign_activate

    def read_net(self, args):
        """
        generate the nodes and feasible links according to obs_info
        1.find nodes:
        1) generate the boundary nodes
        2) pooling nodes
        3) connect N1 with N2 and find the middle nodes as the potential nodes (except ones within same obstacle)
            also no N1,N2 on intersection with obstacle
             * no intersection and no overlap
        2. nodes' feasible links （construct network）
        1) generate all links between nodes
        2) if feasible (no intersection with obstacle)
        3) record all the feasible links and feasible nodes, generate the nodes-links matrix 0/1
        return: nodes-links matrix n-n:0/1 & feasible_nodes [x,y]
        """
        # 1.1)
        obs_info = self.obs_info
        obs_nodes = []  # list
        for obs_idx in range(len(obs_info)):
            x, y = obs_info[obs_idx][1:3]
            w, l = obs_info[obs_idx][3:5]
            obs_nodes.append((obs_idx, x, y))
            obs_nodes.append((obs_idx, x + w, y))
            obs_nodes.append((obs_idx, x, y + l))
            obs_nodes.append((obs_idx, x + w, y + l))
        # 1.2)
        # pooling(gap_min) here is used to deal with the "room" condition and close obs_boundary point
        pooled_nodes_with_id, pooled_nodes_ids = pooling(obs_nodes_with_id=np.array(obs_nodes))
        # 1.3)
        poten_nodes = get_poten_net_nodes(pooled_nodes_with_id, pooled_nodes_ids, obs_info)
        # 2.
        nodes_links_matrix, feasible_nodes = get_net_links(obs_info, poten_nodes, args)

        self.network_nodes = feasible_nodes
        self.network_matrix = nodes_links_matrix

    def read_ObstoField(self, obs, A, B):
        """
        obs_info: lists of list of numpy array, each numpy array corresponds to an obstacle)
        [obs_id,lb_x,lb_y,o_w,o_l]
        also delete the occupied node
        *** prior implementation required
        return:
        # two ways to conduct:
        1.differentiable：every place
        2.non-differentiable: only the vertical place  (current)
        """
        self.obs_info = np.array(obs)
        obs_info = self.obs_info
        dele_node = []  # used for extracting the taken-up area
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for n in range(len(obs_info)):
                obs_w, obs_l = obs_info[n][3:5]
                obs_x, obs_y = obs_info[n][1:3]
                if obs_x <= x <= obs_x + obs_w:
                    if y > obs_y + obs_l:
                        temp_inten += np.array([0, influ_cal_o(A, B, y - (obs_y + obs_l))])
                    elif y < obs_y:
                        temp_inten += np.array([0, -influ_cal_o(A, B, obs_y - y)])
                    else:
                        dele_node.append(node)
                        break
                if obs_y <= y <= obs_y + obs_l:
                    if x > obs_x + obs_w:
                        temp_inten += np.array([influ_cal_o(A, B, x - (obs_x + obs_w)), 0])
                    elif x < obs_x:
                        temp_inten += np.array([-influ_cal_o(A, B, obs_x - x), 0])
                    else:
                        dele_node.append(node)
                        break
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)
        # delete
        for counter, index in enumerate(dele_node):
            index = index - counter
            self.nodes.pop(index)

    def read_PedtoField(self, ped, A, B, herd_range, congest_range=None):
        """
        ped_info: lists of list of numpy array, each numpy array corresponds to a pedestrian
        [ped_id,p_x,p_y,p_iv,p_iu] without consideration of velocity value
        1. the repulsive force
        2. the herding influ with herd_range
        return:
        """
        self.ped_info = np.array(ped)
        ped_info = self.ped_info
        obs_info = self.obs_info
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            node_loc = np.array([x,y])
            temp_inten = np.array([0, 0], dtype=float)
            temp_inten_h = np.array([0, 0], dtype=float)
            for i in range(len(ped_info)):
                ped_x, ped_y = ped_info[i][1], ped_info[i][2]
                ped_iv, ped_iu = ped_info[i][3], ped_info[i][4]
                ped_loc = np.array([ped_x,ped_y])
                # make sure the pedestrian i's behavior can be percepted by the peds located in node
                if check_intersection_state(obs_info, node_loc, ped_loc):
                    continue
                dis = np.array([x - ped_x, y - ped_y])
                d = np.linalg.norm(dis)
                if d <= herd_range:
                    # Superposition of direction
                    temp_inten_h += influ_cal_h(ped_iv,ped_iu)
                e_1 = influ_cal_p(A, B, d)
                if d == 0:
                    self.nodes[node].addInten1(e_1)
                    continue
                temp_inten += np.array([e_1 * (dis[0] / d), e_1 * (dis[1] / d)])
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)
            if list(temp_inten_h) != [0., 0.]:
                self.nodes[node].addInten(temp_inten_h)

    def read_ODtoField(self, danger, exit, B_w, B_sd):
        """
        danger_info:
        [danger_id, d_x, d_y]
        exit_info:
        [exit_id, e_x, e_y]
        return:
        # problem:
        # assume there is no eyesight limitation which is unrealized
        """
        self.danger_info = np.array(danger)
        self.exit_info = np.array(exit)
        danger_info = self.danger_info
        exit_info = self.exit_info
        obs_info = self.obs_info
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            v_d=1e-2

            # danger calculation
            # Problem: it's better to use area instead of point to descirble danger
            for n in range(len(danger_info)):
                r_shortest = float('inf')
                Dis = np.array([0.0, 0.0])
                # Problem： no consideration for large area
                danger_x, danger_y = danger_info[n][1], danger_info[n][2]
                danger_w, danger_l= danger_info[n][3], danger_info[n][4]
                danger_point_x, danger_point_y = danger_x+danger_w/2, danger_y+danger_l/2
                    # dis = np.array([x-danger_x, y-danger_y])
                    # d = np.linalg.norm(dis)
                    # if d < r_shortest:
                    #     r_shortest = d
                    # if d == 0:
                    # #     e_1 = influ_cal_d(B_w, d)
                    #     # self.nodes[node].addInten1(e_1)
                    #     continue
                    # # e = influ_cal_d(B_w, r_shortest)
                    # Dis += dis/d
                    # # temp_inten += np.array([e * (dis[0] / d), e * (dis[1] / d)])
                Dis = np.array([x-danger_point_x, y-danger_point_y])
                D = np.linalg.norm(Dis)
                e_1, v = influ_cal_d(B_w, D)
                if v > v_d:
                    v_d = v
                if D == 0:
                    self.nodes[node].addInten1(e_1)
                else:
                    temp_inten += np.array([e_1 * (Dis[0] / D), e_1 * (Dis[1] / D)])

                temp_inten_danger = temp_inten

            # obstalce-along_surface calculation
            for n in range(len(obs_info)):
                obs_w, obs_l = obs_info[n][3:5]
                obs_x, obs_y = obs_info[n][1:3]
                if obs_x <= x <= obs_x + obs_w:
                    if y > obs_y + obs_l:
                        if -temp_inten_danger[0]>=0:
                            temp_inten += np.array([-influ_cal_o_1(v_d, B_sd, y - (obs_y + obs_l)), 0])
                        else:
                            temp_inten += np.array([influ_cal_o_1(v_d, B_sd, y - (obs_y + obs_l)), 0])
                    elif y < obs_y:
                        if temp_inten_danger[0]>=0:
                            temp_inten += np.array([influ_cal_o_1(v_d, B_sd, obs_y - y), 0])
                        else:
                            temp_inten += np.array([-influ_cal_o_1(v_d, B_sd, obs_y - y), 0])
                if obs_y <= y <= obs_y + obs_l:
                    if x > obs_x + obs_w:
                        if temp_inten_danger[1]>=0:
                            temp_inten += np.array([0, influ_cal_o_1(v_d, B_sd, x - (obs_x + obs_w))])
                        else:
                            temp_inten += np.array([0, -influ_cal_o_1(v_d, B_sd, x - (obs_x + obs_w))])
                    elif x < obs_x:
                        if -temp_inten_danger[1]>=0:
                            temp_inten += np.array([0, -influ_cal_o_1(v_d, B_sd, obs_x - x)])
                        else:
                            temp_inten += np.array([0, influ_cal_o_1(v_d, B_sd, obs_x - x)])

            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)
            # # exit calculation
            # for n in range(len(exit_info)):
            #     exit_x, exit_y = exit_info[n][1], exit_info[n][2]
            #     dis = np.array([exit_x - x, exit_y - y])
            #     d = np.linalg.norm(dis)
            #     if d == 0:
            #         e_1 = influ_cal_e(B_w, r_shortest)
            #         self.nodes[node].addInten1(e_1)
            #         continue
            #     e = influ_cal_e(B_w, r_shortest)
            #     temp_inten += np.array([e * (dis[0] / d), e * (dis[1] / d)])


    # def read_SigntoField(self, k, q):
    #     """
    #     sign_info:
    #     x{-}{-}=0/1
    #     return: updated po_graph
    #     Problem: calculation of sign_influence is different from others
    #     """
    #     sign_activate = self.sign_activate
    #     sign_loc_info = self.sign_loc_info
    #     if sign_loc_info == 0:
    #         print('no sign information')
    #         return
    #     if sign_activate == 0:
    #         print('no activate information')
    #         return
    #     poten_signs = get_poten_signs(sign_loc_info)
    #     num_node = len(self.nodes)
    #     for node in range(num_node):
    #         x = self.nodes[node].x
    #         y = self.nodes[node].y
    #         temp_inten = np.array([0, 0], dtype=float)
    #         for sign in poten_signs:
    #             s_x = self.nodes[sign].x
    #             s_y = self.nodes[sign].y
    #             d = np.linalg.norm(np.array([x - s_x, y - s_y]))
    #             e = inten_cal(k, q, d)
    #             temp_inten += np.array([
    #                 e * sum(np.array([0, 0, -1, 1]) * np.array(
    #                     list(sign_activate['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right']))),
    #                 e * sum(np.array([1, -1, 0, 0]) * np.array(
    #                     list(sign_activate['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right'])))])
    #         if list(temp_inten) != [0., 0.]:
    #             self.nodes[node].addInten(temp_inten)

    def printGraph(self, field_show=True, enviro_show=True):
        """
        Print function for the graph
        For debugging proposes (visualize)
        field_show: if the field is shown on map
        """
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(1, 1, 1)
        num_node = len(self.nodes)
        obs_info = self.obs_info
        ped_info = self.ped_info
        exit_info = self.exit_info
        danger_info = self.danger_info
        sign_loc_info = self.sign_loc_info
        sign_activate = self.sign_activate

        # environment print
        for obs in range(len(obs_info)):
            rect = mpathes.Rectangle(obs_info[obs][1:3], obs_info[obs][3], obs_info[obs][4],
                                     color='black', alpha=0.5)
            ax.add_patch(rect)

        for ped in range(len(ped_info)):
            x, y = ped_info[ped][1], ped_info[ped][2]
            iv, iu = ped_info[ped][3], ped_info[ped][4]
            plt.scatter(x, y, c='orange', alpha=1)
            plt.annotate(
                "",
                xytext=(x, y),
                xy=(x + iv, y + iu),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1),
                size=12,
            )

        if enviro_show:
            plt.scatter(exit_info[:, 1], exit_info[:, 2], c='green', alpha=1)
            for danger in range(len(danger_info)):
                rect = mpathes.Rectangle(danger_info[danger][1:3], danger_info[danger][3], danger_info[danger][4],
                                         color='red', alpha=0.5)
                ax.add_patch(rect)

        # E & settled signage & activate signage print
        for node in range(num_node):
            node_print = self.nodes[node]
            x = node_print.x
            y = node_print.y
            ix, iy = node_print.e_ix, node_print.e_iy
            e = node_print.e
            if field_show:
                plt.annotate(
                    "",
                    xytext=(x, y),
                    xy=(x + ix, y + iy),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1),
                    size=12,
                )
                plt.annotate(
                    str(round(e,1)),
                    xytext=(x, y),
                    xy=(x + ix, y + iy),
                    size=7
                )
                plt.scatter(x, y,
                            s=e * 10,
                            marker='o',
                            facecolors='blue',
                            edgecolors='blue',
                            alpha=0.2)

            if sign_loc_info == 0:
                continue

            if sign_loc_info[node] != 1.0:
                continue
            plt.scatter(x, y, c='black', alpha=1.0)

            if sign_activate == 0:
                continue
            dir_sign = [0, 0]
            if sign_activate['s{}up'.format(node)] == 1:
                dir_sign = [0, 1]
            if sign_activate['s{}down'.format(node)] == 1:
                dir_sign = [0, -1]
            if sign_activate['s{}left'.format(node)] == 1:
                dir_sign = [-1, 0]
            if sign_activate['s{}right'.format(node)] == 1:
                dir_sign = [1, 0]
            # print the sign_direct on the map
            if dir_sign != [0, 0]:
                plt.scatter(x, y, c='green', alpha=1)
                plt.annotate(
                    "",
                    xytext=(x, y),
                    xy=(x + dir_sign[0] * 1, y + dir_sign[1] * 1),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    size=12,
                )
                plt.scatter(x, y,
                            s=5e+4,
                            marker='o',
                            facecolors='green',
                            # edgecolors='red',
                            alpha=0.2)
            # # print the E condition after signage application
            # plt.annotate(
            #     "",
            #     xytext=(x, y),
            #     xy=(x + ix, y + iy),
            #     arrowprops=dict(arrowstyle='->', color='black', lw=1),
            #     size=10,
            # )
            # plt.scatter(x, y,
            #             s=e * 100,
            #             marker='o',
            #             facecolors='red',
            #             edgecolors='red',
            #             alpha=0.2)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(x_major_locator)
        ax1.yaxis.set_major_locator(y_major_locator)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('figure_field{}.png'.format(time.time()))
        # plt.show()

    def printNetwork(self, net_show=False, enviro_show=True, dijkstra=False, dijkstra_path_only=False):
        """
        Print function for the network (nodes-links matrix)
        For debugging proposes (visualize)
        net_show: if the network is shown
        dijkstra: display the dijkstra result
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        obs_info = self.obs_info
        exit_info = self.exit_info
        danger_info = self.danger_info
        network_matrix = self.network_matrix
        network_nodes = self.network_nodes
        num_net_node = len(network_nodes)

        # environment print
        for obs in range(len(obs_info)):
            rect = mpathes.Rectangle(obs_info[obs][1:3], obs_info[obs][3], obs_info[obs][4],
                                     color='black', alpha=0.5)
            ax.add_patch(rect)

        if enviro_show:
            plt.scatter(exit_info[:, 1], exit_info[:, 2], c='green', alpha=1)
            for danger in range(len(danger_info)):
                rect = mpathes.Rectangle(danger_info[danger][1:3], danger_info[danger][3], danger_info[danger][4],
                                         color='red', alpha=0.5)
                ax.add_patch(rect)

        # links print with network_matrix
        if net_show:
            for n in tqdm(range(num_net_node - 1)):
                for n_temp in range(n - 1, num_net_node):
                    if network_matrix[n, n_temp] == 0:
                        continue
                    link_x = [network_nodes[n][0], network_nodes[n_temp][0]]
                    link_y = [network_nodes[n][1], network_nodes[n_temp][1]]
                    plt.plot(link_x, link_y, color='black', linewidth=0.5, alpha=0.5)

        # nodes print
        plt.scatter(network_nodes[:, 0], network_nodes[:, 1], marker='o', s=15, c='orange', alpha=1)

        if dijkstra:
            if dijkstra_path_only:
                n = len(dijkstra_path_only)
                path = np.zeros((n, 2))
                for i in range(n):
                    path[i] = dijkstra.net_nodes[dijkstra_path_only[i]]
                plt.plot(path[:, 0], path[:, 1], color='orange', linewidth=0.8)
            else:
                for node in range(len(dijkstra.nodes)):
                    dijkstra_path = dijkstra.dijkstra_paths[node]
                    length = len(dijkstra_path)
                    path = np.zeros((length, 2))
                    for i in range(length):
                        path[i] = dijkstra.net_nodes[dijkstra_path[i]]
                    plt.plot(path[:, 0], path[:, 1], color='b', linewidth=0.8)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(x_major_locator)
        ax1.yaxis.set_major_locator(y_major_locator)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('figure_network{}.png'.format(time.time()))
        # plt.show()

    def printOpti(self, enviro_show=True, dijkstra=False):
        """
        show the optimized evacuation direction
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        num_node = len(self.nodes)
        obs_info = self.obs_info
        ped_info = self.ped_info
        exit_info = self.exit_info
        danger_info = self.danger_info
        self.dirs = dijkstra.dirs
        exiting_i_dirs = find_the_fittest_dirs(self)

        # environment print
        for obs in range(len(obs_info)):
            rect = mpathes.Rectangle(obs_info[obs][1:3], obs_info[obs][3], obs_info[obs][4],
                                     color='black', alpha=0.5)
            ax.add_patch(rect)

        if enviro_show:
            # plt.scatter(ped_info[:, 1], ped_info[:, 2], c='blue', alpha=1)
            plt.scatter(exit_info[:, 1], exit_info[:, 2], c='green', alpha=1)
            for danger in range(len(danger_info)):
                rect = mpathes.Rectangle(danger_info[danger][1:3], danger_info[danger][3], danger_info[danger][4],
                                         color='red', alpha=0.5)
                ax.add_patch(rect)

        for node in range(num_node):
            node_print = self.nodes[node]
            x = node_print.x
            y = node_print.y
            ix, iy = exiting_i_dirs[node]
            plt.annotate(
                "",
                xytext=(x, y),
                xy=(x + ix, y + iy),
                arrowprops=dict(arrowstyle='->', color='orange', lw=0.8),
                size=15,
            )

        for net_node_idx in range(len(dijkstra.nodes)):
            x, y = dijkstra.nodes[net_node_idx]
            ix, iy = dijkstra.dirs[net_node_idx]
            plt.annotate(
                "",
                xytext=(x, y),
                xy=(x + ix, y + iy),
                arrowprops=dict(arrowstyle='->', color='blue', lw=0.8),
                size=15,
            )
        plt.scatter(dijkstra.nodes[:, 0], dijkstra.nodes[:, 1], marker='o', s=15, c='orange', alpha=1)

        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(x_major_locator)
        ax1.yaxis.set_major_locator(y_major_locator)
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('figure_opt{}.png'.format(time.time()))


class PO_NODE:

    def __init__(self, x, y, e_x, e_y):
        """
        Initializer function for the PO node class
        params:
        x, y - position of the node
        e_x, e_y - the unit direction of the intensity
        e - the volume of the intensity
        """
        self.x, self.y = x, y
        self.e_x, self.e_y = e_x, e_y
        self.e = math.sqrt(sum([e_x ** 2, e_y ** 2]))
        if self.e == 0:
            self.e_ix, self.e_iy = 0, 0
        else:
            self.e_ix, self.e_iy = (self.e_x / self.e), (self.e_y / self.e)

    def addInten(self, inten):
        """
        Add intensity to the existed intensity
        inten: [e_x, e_y] array
        """
        pre_inten = np.array([self.e_x, self.e_y])
        current_inten = pre_inten + inten
        self.e_x, self.e_y = list(current_inten)
        self.e = np.linalg.norm(current_inten)
        self.e_ix, self.e_iy = self.e_x / self.e, self.e_y / self.e

    def addInten1(self, e):
        """
        Add intensity to the existed intensity (theory: no direction, the condition with close dis)
        inten: e
        # Problem:
        when there is a person on the point, it is hard to make the suitable evaluation because the distance is
        so close, and the direction is nan. How to make an influence - combination?
        """
        pre_inten = np.array([self.e_x, self.e_y])
        self.e = np.linalg.norm(pre_inten) + e
        self.e_x, self.e_y = self.e * self.e_ix, self.e * self.e_iy
        # self.e_ix, self.e_iy ---> no change
