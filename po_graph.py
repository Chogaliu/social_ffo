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
from helper import intensity_cal as inten_cal
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
        self.obs_info = 0
        self.ped_info = 0
        self.exit_info = 0
        self.danger_info = 0
        self.sign_activate = 0
        self.sign_loc_info = 0
        self.dist_matrix_ns = 0
        self.grid_size = self.wid * self.len

        self.nodes = [PO_NODE(node % self.wid * self.gap, node // self.wid * self.gap, 0, 0)
                      for node in range(self.grid_size)]
        # self.edges = [{} for i in range(self.grid_size)]

    def reset(self):
        self.nodes = [PO_NODE(node % self.wid * self.gap, node // self.wid * self.gap, 0, 0)
                      for node in range(self.grid_size)]
        # self.edges = [{} for i in range(self.grid_size)]

    def read_ObstoField(self, obs_info, k):
        """
        obs_info: lists of list of numpy array, each numpy array corresponds to an obstacle)
        [obs_id,o_x,o_y,o_w,o_l,o_q]
        also delete the occupied node
        return:
        # two ways to conduct:
        1.differentiable：every place
        2.non-differentiable: only the vertical place  (current)
        """
        self.obs_info = np.array(obs_info)
        dele_node = []  # used for extracting the taken-up area
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for n in range(len(obs_info)):
                obs_x, obs_y = obs_info[n][1], obs_info[n][2]
                obs_w, obs_l, obs_q = obs_info[n][3], obs_info[n][4], obs_info[n][5]
                if obs_x - obs_w / 2 <= x <= obs_x + obs_w / 2:
                    if y > obs_y + obs_l / 2:
                        temp_inten += np.array([0, inten_cal(k, obs_q, y - (obs_y + obs_l / 2))])
                    elif y < obs_y - obs_l / 2:
                        temp_inten += np.array([0, -inten_cal(k, obs_q, (obs_y - obs_l / 2) - y)])
                    else:
                        dele_node.append(node)
                        break
                if obs_y - obs_l / 2 <= y <= obs_y + obs_l / 2:
                    if x > obs_x + obs_w / 2:
                        temp_inten += np.array([inten_cal(k, obs_q, x - (obs_x + obs_w / 2)), 0])
                    elif x < obs_x - obs_w / 2:
                        temp_inten += np.array([-inten_cal(k, obs_q, (obs_x - obs_w / 2) - x), 0])
                    else:
                        dele_node.append(node)
                        break
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)
        # delete
        for counter, index in enumerate(dele_node):
            index = index - counter
            self.nodes.pop(index)

    def read_PedtoField(self, ped_info, k):
        """
        ped_info: lists of list of numpy array, each numpy array corresponds to a pedestrian
        [ped_id,p_x,p_y,p_q]
        return:
        # problem:
        assume there is no eyesight limitation which is unrealized
        # congestion avoidance + herding influence
        """
        self.ped_info = np.array(ped_info)
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for n in range(len(ped_info)):
                ped_x, ped_y = ped_info[n][1], ped_info[n][2]
                ped_q = ped_info[n][3]
                dis = np.array([x - ped_x, y - ped_y])
                d = np.linalg.norm(dis)
                if d == 0:
                    var = 1e-2
                    e_1 = inten_cal(k, ped_q, var)
                    self.nodes[node].addInten1(e_1)
                    continue
                e = inten_cal(k, ped_q, d)
                temp_inten += np.array([e * (dis[0] / d), e * (dis[1] / d)])
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)

    def read_ExittoField(self, exit_info, k):
        """
        exit_info:
        [exit_id, e_x, e_y, e_q]
        return:
        # problem:
        no expected velocity & current velocity considered
        """
        self.exit_info = np.array(exit_info)
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for n in range(len(exit_info)):
                exit_x, exit_y = exit_info[n][1], exit_info[n][2]
                exit_q = exit_info[n][3]
                dis = np.array([x - exit_x, y - exit_y])
                d = np.linalg.norm(dis)
                if d == 0:
                    var = 1e-2
                    e_1 = -inten_cal(k, exit_q, var)
                    self.nodes[node].addInten1(e_1)
                    continue
                e = -inten_cal(k, exit_q, d)
                temp_inten += np.array([e * (dis[0] / d), e * (dis[1] / d)])
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)

    def read_DangertoField(self, danger_info, k):
        """
        danger_info:
        [danger_id, d_x, d_y, d_q]
        return:
        # problem:
        no expected velocity & current velocity considered
        """
        self.danger_info = np.array(danger_info)
        for node in range(len(self.nodes)):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for n in range(len(danger_info)):
                danger_x, danger_y = danger_info[n][1], danger_info[n][2]
                danger_q = danger_info[n][3]
                dis = np.array([x - danger_x, y - danger_y])
                d = np.linalg.norm(dis)
                if d == 0:
                    var = 1e-2
                    e_1 = inten_cal(k, danger_q, var)
                    self.nodes[node].addInten1(e_1)
                    continue
                e = inten_cal(k, danger_q, d)
                temp_inten += np.array([e * (dis[0] / d), e * (dis[1] / d)])
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)

    def read_SigntoField(self, k, q):
        """
        sign_info:
        x{-}{-}=0/1
        return: updated po_graph
        Problem: calculation of sign_influence is different from others
        """
        sign_activate = self.sign_activate
        sign_loc_info = self.sign_loc_info
        if sign_loc_info == 0:
            print('no sign information')
            return
        if sign_activate == 0:
            print('no activate information')
            return
        poten_signs = get_poten_signs(sign_loc_info)
        num_node = len(self.nodes)
        for node in range(num_node):
            x = self.nodes[node].x
            y = self.nodes[node].y
            temp_inten = np.array([0, 0], dtype=float)
            for sign in poten_signs:
                s_x = self.nodes[sign].x
                s_y = self.nodes[sign].y
                d = np.linalg.norm(np.array([x - s_x, y - s_y]))
                e = inten_cal(k, q, d)
                temp_inten += np.array([
                    e * sum(np.array([0, 0, -1, 1]) * np.array(
                        list(sign_activate['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right']))),
                    e * sum(np.array([1, -1, 0, 0]) * np.array(
                        list(sign_activate['s{}{}'.format(sign, j)] for j in ['up', 'down', 'left', 'right'])))])
            if list(temp_inten) != [0., 0.]:
                self.nodes[node].addInten(temp_inten)

    def printGraph(self, field_show=True):
        """
        Print function for the graph
        For debugging proposes (visualize)
        field_show: if the field is shown on map
        """
        fig, ax = plt.subplots()
        num_node = len(self.nodes)
        obs_info = self.obs_info
        ped_info = self.ped_info
        exit_info = self.exit_info
        danger_info = self.danger_info
        sign_loc_info = self.sign_loc_info
        sign_activate = self.sign_activate

        # environment print
        plt.scatter(ped_info[:, 1], ped_info[:, 2], c='blue', alpha=1)
        plt.scatter(exit_info[:, 1], exit_info[:, 2], c='green', alpha=1)
        plt.scatter(danger_info[:, 1], danger_info[:, 2], c='red', alpha=1)
        for obs in range(np.shape(obs_info)[0]):
            rect = mpathes.Rectangle(obs_info[obs][1:3] - obs_info[obs][3:5] / 2, obs_info[obs][3], obs_info[obs][4],
                                     color='black', alpha=0.5)
            ax.add_patch(rect)

        # E & signage print
        if field_show:
            for node in range(num_node):
                node_print = self.nodes[node]
                x = node_print.x
                y = node_print.y
                ix, iy = node_print.e_ix, node_print.e_iy
                e = node_print.e
                plt.annotate(
                    "",
                    xytext=(x, y),
                    xy=(x + ix, y + iy),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1),
                    size=10,
                )
                plt.scatter(x, y,
                            s=e * 100,
                            marker='o',
                            facecolors='blue',
                            edgecolors='blue',
                            alpha=0.2)

                if sign_loc_info == 0:
                    continue

                if sign_loc_info[node] == 1.0:
                    plt.scatter(x, y, c='black', alpha=1.0)
                else:
                    continue

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
                    plt.scatter(x, y, c='red', alpha=1)
                    plt.annotate(
                        "",
                        xytext=(x, y),
                        xy=(x + dir_sign[0]*5, y + dir_sign[1]*5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1),
                        size=10,
                    )
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
        plt.savefig('figure{}.png'.format(time.time()))
        # plt.show()


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
