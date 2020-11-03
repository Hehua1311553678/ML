# -*- coding: utf-8 -*-
# @Time     : 2020/11/3 20:25
# @Author   : Hehua
# @Email    : hehua@nuaa.edu.cn
# @File     : DecisionTree.py
# @Software : PyCharm
import numpy as np

def Entropy(Y):
    '''计算熵
    Y: 所有的y标记；
    返回: Y的熵
    '''
    classes = set(Y)
    P = []
    for c in classes:
        y_c = np.array([1 if y == c else 0 for y in Y])
        P.append(y_c.sum()/len(Y))
    H = 0 - (P * np.log(P)).sum()
    return H

def InformatinGain(feature, tr_X, tr_Y):
    '''计算feature带来的信息增益
    feature: 特征索引（如：0,1,2,...）；
    tr_X: 训练数据集的X；
    tr_Y: 训练数据集的Y；
    返回: feature对训练数据集带来的信息增益。
    '''
    # 1. 计算训练数据集的经验熵H(D)；
    H = Entropy(tr_Y)

    # 2. 计算特征对训练数据集的经验条件熵H(D|feature)；
    values = set([x[feature] for x in tr_X])
    H_feature = 0
    for v in values:
        X_v  = np.array([True if x == v else False for x in tr_X])
        Y_v = list(np.array(tr_Y)[X_v])
        H_v = Entropy(Y_v)
        H_feature += len(Y_v)/len(tr_Y) * H_v

    # 3. 计算信息增益；
    return H - H_feature

def InformationGainRatio(feature, tr_X, tr_Y):
    '''计算feature带来的信息增益
    feature: 特征索引（如：0,1,2,...）；
    tr_X: 训练数据集的X；
    tr_Y: 训练数据集的Y；
    返回: feature对训练数据集带来的信息增益比。
    '''
    # 1. 计算信息增益；
    g = InformatinGain(feature, tr_X, tr_Y)
    # 2. 计算训练数据集关于特征feature的值的熵；
    X_feature = [x[feature] for x in tr_X]
    H_feature = Entropy(X_feature)
    # 3. 计算信息增益比；
    gR = g/H_feature
    return gR