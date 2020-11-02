import pandas as pd

tr_dir = "G:\hehua\DB\pubg-finish-placement-prediction\\train_V2.csv"
te_dir = "G:\hehua\DB\pubg-finish-placement-prediction\\test_V2.csv"

def data_preprocess(data_dir = None):
    '''处理数据：丢弃特征
    dir：数据集绝对路径。
    '''
    data = pd.read_csv(data_dir)
    # 去掉NULL值
    if 'train' in data_dir:
        data = data.dropna(axis=0)
    # 丢掉7个特征，余22个特征。
    droplist = ['Id', 'groupId', 'matchId', 'matchType', 'rankPoints', 'winPoints', 'maxPlace']
    data_new = data.drop(droplist, axis=1)
    return data_new

if __name__ == '__main__':
    data = pd.read_csv(te_dir)
    print("测试数据集大小：{};".format(len(data)))