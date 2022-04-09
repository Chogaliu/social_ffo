"""
reference：https://blog.csdn.net/qq_43741312/article/details/97128745?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_antiscanv2&utm_relevant_index=9
Try the Cluster analysis of the area with convergence trend
data attribute (intention xi, intention yi, opti-dir xi, opti-dir yi, loc x, loc y)
and visualize the result

Author: Qiujia Liu
Data: 4th April 2022
"""

import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFrame(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    minDist = np.min(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster, minDist, minDistIndices


# introduce the dataset from the .npy
def createDataSet():
    raw_data = np.load("tests/data-attri-for-cluster.npy", allow_pickle=True).item()
    dataset1 = []
    for index, k in raw_data.items():
        a = [math.acos(k[0][0] / np.linalg.norm(k[0]))]  # intention
        b = [math.acos(k[1][0] / np.linalg.norm(k[1]))]  # opti-direction
        e = [k[2]]
        loc = [k[3][0], k[3][1]]
        k1 = a + b + e + loc
        dataset1.append(k1)
    # dataset = np.array(dataset1)
    return dataset1


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return '#' + color


if __name__ == '__main__':
    dataset = createDataSet()
    num = 10
    centroids, cluster, minDist, minDistIndices = kmeans(dataset, num)
    colors = [randomcolor() for i in range(num)]
    print('Signs：%s' % centroids)
    # print('Clusters：%s' % cluster)
    for j in range(len(centroids)):
        plt.scatter(centroids[j][3], centroids[j][4], marker='x', color='red', s=20, label='centroid')
        for i in range(len(cluster[j])):
            plt.scatter(cluster[j][i][3], cluster[j][i][4], marker='o', color=colors[j], s=10, label='cluster')
    plt.show()
