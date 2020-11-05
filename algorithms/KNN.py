# -*- coding: utf-8 -*-
# @Time     : 2020/11/2 21:29
# @Author   : Hehua
# @Email    : hehua@nuaa.edu.cn
# @File     : KNN.py
# @Software : PyCharm
import numpy as np

def Minkowski(x, tr_X, p=2):
    '''计算x到tr_X每个样本的距离'''
    x = np.array(x)
    D = []
    for tr_x in tr_X:
        tr_x = np.array(tr_x)
        if p is 'infinite':
            d = np.max(np.abs(x - tr_x))
        else:
            d = np.power(np.sum(np.power(x - tr_x, p)), 1/p)
        D.append(d)
    return D


def distance(x, tr_X, distance_type='Minkowski', p=2):
    '''计算x到tr_X每个样本的距离'''
    if distance_type == 'Minkowski':
        return Minkowski(x, tr_X, p)

def KNN(x, tr_X, tr_Y, k, distance_type='Minkowski', p=2):
    ''' k近邻算法
    x: 测试样本;
    tr_X: 训练数据集的X；
    tr_Y: 训练数据集的Y；
    k: 近邻个数;
    distance_type: 距离度量（如：Minkowski）；
    p: Minkowski距离参数（如：p=2）；
    返回：测试样本的预测类别y.
    '''
    # 1. 求x与每个训练样本的距离；
    D = distance(x, tr_X, distance_type, p)
    # 2. 取距离最近的top k（升序）个样本；
    dict_D = dict(enumerate(D))
    index = dict(sorted(dict_D.items(), key=lambda x:x[1], reverse=False)).keys()[:k]
    # 3. 多数表决，得到y
    dict_y = {}
    for i in index:
        if tr_Y[i] not in dict_y.keys():
            dict_y[tr_Y[i]] = 0
        dict_y[tr_Y[i]] += 1
    y = dict(sorted(dict_y.items(), key=lambda x:x[1], reverse=True)).keys()[0]
    return y

if __name__ == '__main__':
    pass
