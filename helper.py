"""
helper functions

Author: Qiujia Liu
Data: 25th August 2021
"""

import numpy as np
from gurobipy import *


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
    u = (-0.6 * e + 1) * (-5 * angle + 15) + 4 * e
    return u


def find_the_fittest_exit(po_graph):
    """
    through exit_info and node_info find the fittest exit at the current state for each node
    principle: THE CLOSEST ONE
    return: fittest_exit
    fittest_exit {node_id}=exit_id
    """
    fittest_exit = {}
    exit_info = po_graph.exit_info
    for node_id in range(len(po_graph.nodes)):
        dis = [np.linalg.norm(
            np.array(exit_info[exit_id][1:3]) - np.array([po_graph.nodes[node_id].x, po_graph.nodes[node_id].y])) for
            exit_id in
            range(len(exit_info))]
        fitting_exit_id = dis.index(min(dis))
        fittest_exit[node_id] = fitting_exit_id
    return fittest_exit


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


def optimize_lp_1(filename):
    """
    solve the .lp file
    return: sign_loc_info (settlement of possible signage locations)
    sign_loc_info x{}=0/1:
    """
    model = read(filename)
    model.optimize()
    print("Objective:", model.objVal)
    sign_loc_info = {}
    for i in model.getVars():
        # if 'x' in i.varname:
        print("Parameter:", i.varname, "=", i.x)
        index = int(i.varname[1:])
        sign_loc_info[index] = i.x
    return sign_loc_info


def optimize_lp_2(filename):
    """
    solve the .lp file
    return: sign_info - activated
    sign_info s{}{}=0/1:
    """
    model = read(filename)
    model.optimize()
    print("Objective:", model.objVal)
    sign_activate = {}
    for i in model.getVars():
        print("Parameter:", i.varname, "=", i.x)
        if 's' in i.varname:
            sign_activate[i.varname] = i.x
    return sign_activate


def optimize_lp_3(filename):
    """
    solve the .lp file
    return: exiting_dir (unit)
    dirs d{}=[ix,iy]:
    """
    model = read(filename)
    model.optimize()
    print("Objective:", model.objVal)
    dirs = {}
    for i in model.getVars():
        # if 'x' in i.varname:
        print("Parameter:", i.varname, "=", i.x)
        index = int(i.varname[1:])
        dirs[index] = i.x
    return dirs
