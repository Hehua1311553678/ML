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
    '''计算feature带来的信息增益比
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

def get_featrue_with_max_IG(features, X, Y):
    '''
    计算信息增益最大的特征
    @param features:特征索引（如：[0,1,2,...]）；
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @return:信息增益最大的特征feature和对应的信息增益
    '''
    max_IG, max_feature = 0, None
    for feature in features:
        IG = InformatinGain(feature, X, Y)
        if IG > max_IG:
            max_IG, max_feature = IG, feature
    return max_feature, max_IG

def get_featrue_with_max_IGR(features, X, Y):
    '''
    计算信息增益比最大的特征
    @param features:特征索引（如：[0,1,2,...]）；
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @return:信息增益比最大的特征feature和对应的信息增益比
    '''
    max_IGR, max_feature = 0, None
    for feature in features:
        IGR = InformationGainRatio(feature, X, Y)
        if IGR > max_IGR:
            max_IGR, max_feature = IGR, feature
    return max_feature, max_IGR

def ID3(X, Y, features, epsilon):
    '''
    ID3算法（生成决策树）
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @param features:特征索引（如：[0,1,2,...]）；
    @param epsilon:信息增益阈值；
    @return:决策树T。
    '''
    # 1. 若Y中只有一类Ck，则T为单节点树，类别为Ck，返回T;
    if len(list(set(Y))) == 1:
        return Y[0]
    # 2. 若features为空，则T为单节点树，类别为Y中实例数最大的类别，返回T;
    if len(features) == 0:
        return max(Y, Y.count)
    else:
        # 3. 否则计算信息增益最大的特征feature;
        feature, IG = get_featrue_with_max_IG(features, X, Y)
        # 4. 若feature的信息增益小于阈值epsilon，则T为单节点树，类别为Y中实例数最大的类别，返回T;
        if IG < epsilon:
            return max(Y, key=Y.count)
        else:
            # 5. 否则，根据feature所有取值划分数据集，构建子节点，返回T;
            dict_node = {}
            values = set(X[feature])
            for value in values:
                index = [True if x[feature] == value else False for x in X]
                # 6. 对第i个子节点，以子数据集和features-featrue，递归调用1-5，得到子树Ti，返回Ti.
                dict_node['value'] = ID3(X[index], Y[index], features.remove(feature), epsilon)
            return {feature: dict_node}

def C45(X, Y, features, epsilon):
    '''
    C4.5算法（生成决策树）
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @param features:特征索引（如：[0,1,2,...]）；
    @param epsilon:信息增益阈值；
    @return:决策树T。
    '''
    # 1. 若Y中只有一类Ck，则T为单节点树，类别为Ck，返回T;
    if len(list(set(Y))) == 1:
        return Y[0]
    # 2. 若features为空，则T为单节点树，类别为Y中实例数最大的类别，返回T;
    if len(features) == 0:
        return max(Y, Y.count)
    else:
        # 3. 否则计算信息增益比最大的特征feature;
        feature, IGR = get_featrue_with_max_IGR(features, X, Y)
        # 4. 若feature的信息增益比小于阈值epsilon，则T为单节点树，类别为Y中实例数最大的类别，返回T;
        if IGR < epsilon:
            return max(Y, key=Y.count)
        else:
            # 5. 否则，根据feature所有取值划分数据集，构建子节点，返回T;
            dict_node = {}
            values = set(X[feature])
            for value in values:
                index = [True if x[feature] == value else False for x in X]
                # 6. 对第i个子节点，以子数据集和features-featrue，递归调用1-5，得到子树Ti，返回Ti.
                dict_node['value'] = ID3(X[index], Y[index], features.remove(feature), epsilon)
            return {feature: dict_node}

def classify(T, te_x):
    '''
    根据决策树T，对te_x进行分类
    @param T:决策树;
    @param te_x:测试示例；
    @return:分类结果。
    '''
    # 1. 根据第一个特征取值找子节点；
    first_feature = int(list(T.keys())[0])
    first_values_dict = T[first_feature]
    first_value = te_x[first_feature]
    second_dict = first_values_dict[first_value]
    # 2. 若子节点是树（字典）则递归查询子树；
    if type(second_dict) == dict:
        return classify(second_dict, te_x)
    else:
        # 3. 否则返回当前类别。
        return second_dict

def calculate_loss(X, Y, j, s):
    '''
    根据j, s计算式（5.21）的值（loss）
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @param j:切分变量；
    @param s:切分点；
    @return:loss值。
    '''
    index1 = np.array([True if x[j] <= s else False for x in X])
    index2 = ~index1
    R1, Y1 = list(np.array(X)[index1]), np.array(Y)[index1]
    R2, Y2 = list(np.array(X)[index2]), np.array(Y)[index2]

    var1, var2 = Y1.var(), Y2.var()
    loss = len(Y1) * var1 + len(Y2) * var2
    return loss, R1, R2, list(Y1), list(Y2), Y1.mean(), Y2.mean()

def Least_Squares_Regression_Tree(X, Y, times, total_times):
    '''
    最小二乘回归树生成算法
    @param X:训练数据集的X；
    @param Y:训练数据集的Y；
    @param times:划分次数;
    @param total_times:划分总次数;
    @return:回归树。
    '''
    if len(X) == 0: return 0
    if len(X) == 1: return Y[0]
    if times == total_times:
        return np.array(Y).mean()
    # 1. 选择最优切分变量j与切分点s；
    features = np.arange(len(X[0]))
    min_loss, best_j, best_s = -1, -1, -1
    for j in features:
        values_j = set([x[j] for x in X])
        for s in values_j:
            loss, _, _, _, _, _, _ = calculate_loss(X, Y, j, s)
            if (min_loss == -1) or (loss < min_loss):
                min_loss, best_j, best_s = loss, j, s
    # 2. 用选定的对(j, s)划分区域并决定相应的输出值；
    _, R1, R2, Y1, Y2, c1, c2 = calculate_loss(X, Y, best_j, best_s)
    # 3. 继续对两个子区域调用步骤1,2，直至满足停止条件；
    T1 = Least_Squares_Regression_Tree(R1, Y1, times + 1, total_times)
    T2 = Least_Squares_Regression_Tree(R2, Y2, times + 1, total_times)
    # 4. 将输入空间划分为M个区域R1, R2, ..., RM生成决策树。
    return {j: {s: T1, s+1: T2}}

if __name__ == '__main__':
    # 测试
    X = [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]]
    Y = ['yes', 'yes', 'no', 'no', 'no']
    T = Least_Squares_Regression_Tree(X, Y, 0, 2)
    print(T)