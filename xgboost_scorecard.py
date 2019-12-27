import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from xgboost_utils import xgb_utils
from dataLoading import dataFrame2sheet
from iv_ks import scorebins,bin_iv_ks_inone
from evaluationFuncs import calculate_auc, calculate_psi, calculate_ks, prob2Score

class xgb_scorecard(xgb_utils):
    '''把预测的结果转换为分数'''
    def __init__(self,detail_path, result_save_path, force_keep_col):
        super(xgb_scorecard,self).__init__(detail_path, result_save_path, force_keep_col)
    def xgb_scores(self, train, test, oot):
        '''
        调用xbgoost_utils中已经训练好的模型来预测
        :param train:训练集
        :param test:测试集合
        :param oot:oot
        :return:保存分数以及结果
        '''
        bins,ivs = bin_iv_ks_inone(train, how='inner', dropcol=self.force_keep_col, verbose=False)
        pred_train, pred_test, pred_val, model = self.predicts(train, test, oot)
        auc_train, gini_train = calculate_auc(pred_train,
                                              targetName='target',
                                              title='train_auc',
                                              figure_save_path=self.result_save_path)
        auc_test, gini_test = calculate_auc(pred_test,
                                            targetName='target',
                                            title='test_auc',
                                            figure_save_path=self.result_save_path)

        auc_val, gini_val = calculate_auc(pred_val,
                                          targetName='target',
                                          title='test_auc',
                                          figure_save_path=self.result_save_path)

        ks_train = calculate_ks(pred_train,
                                targetName='target',
                                n=10,
                                title='train_ks',
                                figure_save_path=self.result_save_path)
        ks_test = calculate_ks(pred_test,
                               targetName='target',
                               n=10,
                               title='test_ks',
                               figure_save_path=self.result_save_path)
        ks_val = calculate_ks(pred_val,
                              targetName='target',
                              n=10,
                              title='val_ks',
                              figure_save_path=self.result_save_path)

        thea = (len(train) - train.target.sum()) / train.target.sum()
        score_train = prob2Score(pred_train,
                                 thea=thea,
                                 basescore=600,
                                 PDO=50,
                                 targetName='target')
        score_test = prob2Score(pred_test,
                                thea=thea,
                                basescore=600,
                                PDO=50,
                                targetName='target')
        score_val = prob2Score(pred_val,
                               thea=thea,
                               basescore=600,
                               PDO=50,
                               targetName='target')

        cardscorebins = scorebins(score_train, score_test, score_val, prob_score='score',
                                  figure_save_path=self.result_save_path)
        cardscorebins_prob = scorebins(score_train, score_test, score_val, prob_score='prob', cutway='cut1',
                                  figure_save_path=self.result_save_path)
        psitable = calculate_psi(score_train.score, score_test.score, n_cut=10, varN='train_test')
        psitable_val = calculate_psi(score_test.score, score_val.score, n_cut=10, varN='test_oot')
        psitable_p = calculate_psi(score_train.prob, score_test.prob, n_cut=10, varN='prob_train_test')
        psitable_valp = calculate_psi(score_test.prob, score_val.prob, n_cut=10, varN='prob_test_oot')
        psitable = psitable.append(psitable_val)
        psitable = psitable.append(psitable_p)
        psitable = psitable.append(psitable_valp)
        model_feature = pd.DataFrame(pickle.load(open(self.result_save_path + '/final_feature.pkl', 'rb')),
                                     columns=['入模标签'])
        try:
            model_performance = pd.read_excel(self.result_save_path+'/model_performance.xlsx').reset_index()
        except Exception  as e:
            print(" **maybe the model_performance.xlsx file not found** ")
        dataFrame2sheet(self.detail_path+'/summary.xlsx',[bins,ivs.reset_index(),model_performance,psitable,cardscorebins,cardscorebins_prob],
                        ['入模标签分箱','入模标签表现','模型表现','psi表现','分数分布','概率分布'])

        scores = score_train.append(score_test)
        scores = scores.append(score_val)
        scores.to_excel(self.result_save_path + '/scores.xlsx',index=False)
        results = pd.DataFrame()
        results = results.append(pd.DataFrame({'kstrain': ks_train, 'kstest': ks_test, 'ksval': ks_val, \
                                               'auctrain': auc_train, 'auc_test': auc_test, 'auc_val': auc_val},
                                              index=[0]))
        return results