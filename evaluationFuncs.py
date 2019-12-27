import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc
# from dataLoading import load, excelAddSheet
from dataLoading import dumps

def sort_(str_):
    match = re.match('^\[(.*),.+', str(str_))
    if match:
        return float(match.group(1))
    else:
        return np.nan

def calculate_psi(train, test, n_cut=10, varN=None):

    '''
    计算PSI指标，用于评估模型稳定性, 主要用于计算得分列

    :param train: 1-array or serise, train score
    :param test: 1-array or serise, train score
    :param n_cut:  numbers of split parts

    :return:

    - PSI表

    '''

    def format_func(x, splits, col):
        crossT = pd.DataFrame()
        cut_ = pd.cut(x, bins=splits, right=False)
        crossT[col] = cut_.value_counts() / len(x)
        crossT = crossT.fillna(0)
        return crossT.reset_index().rename(columns={'index': 'bins'})

    if isinstance(n_cut, (int, float)):
        if n_cut < 1:
            warnings.warn('Expect n_cut ">" 1, got{}. was reset to default!'.format(n_cut))
            n_cut = 10
    else:
        warnings.warn('Expect n_cut int、float, got{}. was reset to default "10"!'.format(n_cut))
        n_cut = 10
    splits = pd.cut(train, n_cut, retbins=True, duplicates='drop')[1]
    crosstable = format_func(train, splits, 'A%')
    table_test = format_func(test, splits, 'B%')
    crosstable = pd.merge(crosstable, table_test, on=['bins'], how='left', sort=False)
    crosstable['A%-B%'] = crosstable['A%'] - crosstable['B%']
    crosstable['ln(A%/B%)'] = crosstable['A%'] / crosstable['B%']
    crosstable['EXP'] = crosstable['A%-B%'] * crosstable['ln(A%/B%)']
    crosstable.loc[-1] = ['All', round(crosstable['A%'].sum()),
                          round(crosstable['B%'].sum()),
                          np.nan, np.nan,
                          crosstable['EXP'].sum()]
    crosstable['sort'] = crosstable['bins'].apply(lambda x: sort_(x))
    crosstable = crosstable.sort_values(by=['sort']).reset_index(drop=True)
    crosstable.drop('sort', axis=1, inplace=True)
    crosstable['var_name'] = varN
    return crosstable

def calculate_auc(data, targetName, probName='prob', plot=True, title=None, figure_save_path=None):
    '''
    绘制ROC曲线，并计算AUC值

    :param data: dataframe, 包含预测概率，label列的数据集
    :param targetName: str, 目标列列名
    :param probName: str, 预测概率列列名
    :param title: str, 曲线名称
    :param figure_save_path: str, 图形存放地址

    :return: AUC: float, AUC值
    '''
    fpr, tpr, threshold = roc_curve(data[targetName], data[probName])
    rocauc = auc(fpr, tpr)
    if plot:
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 14,
                }
        plt.figure(num=0, figsize=(8, 4))
        plt.plot(fpr, tpr, 'b', label='auc = %0.2f' % rocauc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        ax = plt.gca()
        if title is not None:
            ax.set_title(title, font)
        else:
            ax.set_title('roc_curve', font)
        ax.set_xlabel('% fpr', font)
        ax.set_ylabel('% tpr', font)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        if figure_save_path is not None:
            dumps(figure_save_path, plt, adds='/png' + '/' + title + '.png')
            '''在plt.show()调用之后，再对图像进行保存，会生成类似于空白图像，
            主要原因是: 在plt.show()后实际上已经创建了一个新的空白的图片（坐标轴） '''
        plt.show()
        plt.close()
    return rocauc, rocauc * 2 - 1


def calculate_ks(data, targetName, probName='prob',n=None, plot=True, title=None, figure_save_path=None):
    """
    计算ks值，绘制ks曲线

    :param X: dataframe, 由预测列、目标列组成的dataframe
    :param targetName: str, 目标变量列列名
    :param probName: str, 预测列(或者概率列)列名
    :param n: int, 将输入数据集X等分成几份，用于计算ks值
    :param plot: bool, 是否绘制ks曲线
    :param title: str, 图型名称
    :param figure_save_path: str, 图形存放地址

    :return:
    """
    if n is None: n = len(data)  # 等分成几份
    n = n if n <= len(data) else len(data)

    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)
    '''
    将得分列按从小到大排列，即：将probName列按从大到小排列
    p越高，odds越大 odds= p/(1-p), score越小， score= A-Blog(odds), 违约风险越大
    
    这里选择的是概率p, 所以按照从大到小排列， ascending= False
    '''
    df_ks = data.sort_values(probName, ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / n))) \
        .groupby('group')[targetName].agg([n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .assign(
        group=lambda x: (x.index + 1) / len(x.index),
        cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
        cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
    ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))
    df_ks = pd.concat([
        pd.DataFrame(
            {'group': 0, 'good': 0, 'bad': 0, 'cumgood': 0,
             'cumbad': 0, 'ks': 0}, index=np.arange(1)),
        df_ks
    ], ignore_index=True)
    if plot:
        seri_ks = df_ks.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 14
                }
        plt.figure(num=1, figsize=(8, 4))
        l1, = plt.plot(df_ks.group, df_ks.ks, color='blue', linestyle='-')  # 绘制ks曲线
        l2, = plt.plot(df_ks.group, df_ks.cumgood, color='green', linestyle='-')
        l3, = plt.plot(df_ks.group, df_ks.cumbad, 'k-')
        l4, = plt.plot([seri_ks['group'], seri_ks['group']], [0, seri_ks['ks']], 'r--')
        plt.text(seri_ks['group'], max(df_ks['ks']), 'KS = %0.2f' % max(df_ks['ks']), font)
        plt.legend(handles=[l1, l2, l3, l4],
                   labels=['ks-curve', 'fpr-curve', 'tpr-curve'],
                   loc='upper left')
        ax = plt.gca()
        if title is not None:
            ax.set_title(title, font)
        else:
            ax.set_title('k-s', font)
        ax.set_xlabel('% of population', font)
        ax.set_ylabel('% of total Good/Bad', font)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        if figure_save_path is not None:
            dumps(figure_save_path, plt, adds='/png' + '/' + title + '.png')
        plt.show()
        plt.close()
    return max(df_ks['ks'])


def prob2Score(data, probName='prob', thea=50, basescore=600, PDO=20, targetName=None, scoreName=None, **kwargs):
    """
    计算评分卡评分

    :param data: dataframe, 含有预测概率, 可能含有目标列
    :param probName: str, 预测概率列的列名
    :param thea: float、int, 评分卡刻度，thea= 50, 表示:  正常/违约= 50， 也即:  count(good)/count(bad)
    :param basescore: float、int, 表示评分卡刻度在'正常/违约'的比率为50 时的得分为basescore= 600
    :param PDO: float、int, 比例翻番的分数
    :param targetName: str, 目标变量所在的列
    :param kwargs:

    - is_sampled: 用于模型训练数据是否使用了数据重采样方法，默认: 传参样式, is_sampled= False, 训练数据集没有采用数据重采样方法
    - true_badrate: 在训练数据采用了数据重采样方法，即： is_sampled= True, true_badrate表示数据重采样前样本违约率

    值得注意的是：true_badrate这个参数的值可以通过调用DataSampling(data, target, sample_type='over', sample_rate=0.5)
    sampled, badrate = dr.sample_maker()获得, 传参样式：true_badrate= badrate
    :return: X新增一列: 评分转化值

    - sampled_badrate: 采样后，违约率，即: 坏样本占比， 这个程序中可以自动计算
    - true_badrate:  采样前， 违约率，即: 坏样本占比

    评分矫正公式:

        - 未矫正:
            - Score= A-B*log2(P10%/1-P10%)
        - 矫正后:
            - P10%= 1-(1-P30%)**(10%/30%)
            - Score= A-B*log2(((1-P10%)/ P10%)/((1-10%)/ 10%))
            - 10%: 实际客群为违约率
            - 30%: 采样后，训练客群违约率
    """
    is_sampled = False
    true_badrate = None
    if not kwargs:
        pass
    else:
        kwargs = {k.lower(): float(v) for k, v in kwargs.items()}
        if kwargs.get('is_sampled'):
            is_sampled = kwargs.get('is_sampled')
        if kwargs.get('true_badrate'):
            true_badrate = kwargs.get('true_badrate')

    if scoreName is not None:
        if isinstance(scoreName, str):
            pass
        else:
            warnings.warn("Expect scoreName 'str', was reset to default 'score'!")
            scoreName = 'score'
    else:
        scoreName = 'score'
    B = PDO / np.log(2)
    A = basescore + B * np.log(1 / thea)
    X_copy = data.copy(deep=True)
    # 如果数据进行了数据平衡重采样，则需要评分拉伸
    if is_sampled and true_badrate is not None:
        sampled_badrate = X_copy[targetName].agg(lambda x: sum(x == 1) / (sum(x == 0) + sum(x == 1)))
        X_copy[probName] = X_copy[probName].apply(lambda x: 1 - (1 - x) ** (true_badrate / sampled_badrate))
        X_copy[scoreName] = round(
            A - B * np.log(((1 - X_copy[probName]) / X_copy[probName]) / (1 - true_badrate) / true_badrate))
    else:
        X_copy[scoreName] = round(A - B * np.log(X_copy[probName] / (1 - X_copy[probName])))
    X_copy[scoreName] = X_copy[scoreName].astype(int)
    return X_copy

# def calculate_variable_psi
# 计算变量psi的方法,待补充
# def calculate_variable_psi(train_detail_path,
#                            test_detail_path,
#                            columns,
#                            sheet_name='变量psi指标',
#                            save_path=None):
#     """
#     此函数用来计算训练数据和测试数据集相同变量间的psi指标

#     :param train_detail_path: str, 训练数据集分箱明细表加载地址，可以由VarCutBin.transform过程产生
#     :param test_detail_path: str, 测试数据集分箱明细表加载地址，可以由VarCutBin.transform过程产生
#     :param columns: list、tuple、np.array、str等，需要计算psi值的变量
#     :param sheet_name, str, 生成的变量psi指标表保存成excel的sheet名
#     :param save_path: str, psi表保存路径， 文件保存成excel格式， 可以对已经存在的excel进行追加，sheet名是 'sheet_name'

#     :return:

#     """

#     def func(path_):
#         details = load(path_)
#         len_ = details[details['bins'] == 'All']['total'].values[0]
#         details = details[details['bins'] != 'All']
#         return details, len_

#     crosstable = pd.DataFrame()
#     lst_cross = []
#     train_details, train_len = func(train_detail_path)
#     test_details, test_len = func(test_detail_path)

#     if isinstance(columns, (list, tuple, np.ndarray)):
#         pass
#     elif isinstance(columns, str):
#         columns = [columns]
#     else:
#         warnings.warn("Expect columns 'list'、'tuple'、'np.array'、'str', got{}".format(type(columns)))
#         columns = []
#     if not train_details.empty and not test_details.empty:
#         train_cols = list(set(train_details['var_name']))
#         test_cols = list(set(test_details['var_name']))
#         columns = list(set(columns) & set(train_cols) & set(test_cols))
#         if columns:
#             for col in columns:
#                 train_table = pd.DataFrame()
#                 table_test = pd.DataFrame()
#                 train_table['bins'] = train_details[train_details['var_name'] == col]['bins']

#                 table_test['bins'] = test_details[test_details['var_name'] == col]['bins']
#                 table_test['B%'] = test_details[test_details['var_name'] == col]['total'] / test_len
#                 train_table = pd.merge(train_table, table_test, on=['bins'], how='left', sort=False)
#                 train_table['A%-B%'] = train_table['A%'] - train_table['B%']
#                 train_table['ln(A%/B%)'] = train_table['A%'] / train_table['B%']
#                 train_table['EXP'] = train_table['A%-B%'] * train_table['ln(A%/B%)']
#                 train_table = train_table.fillna(0)
#                 train_table.loc[-1] = ['All', round(train_table['A%'].sum()),
#                                        round(train_table['B%'].sum()),
#                                        np.nan, np.nan,
#                                        train_table['EXP'].sum()]
#                 train_table['sort'] = train_table['bins'].apply(lambda x: sort_(x))
#                 train_table = train_table.sort_values(by=['sort']).reset_index(drop=True)
#                 train_table.drop('sort', axis=1, inplace=True)
#                 train_table['var_name'] = col
#                 lst_cross.append(train_table)
#     if lst_cross:
#         crosstable = pd.concat(lst_cross, axis=0, ignore_index=True)
#         excelAddSheet(save_path, crosstable, sheetName=sheet_name)
#     else:
#         warnings.warn('calculate_variable_psi 计算失败！！')
#     return crosstable


