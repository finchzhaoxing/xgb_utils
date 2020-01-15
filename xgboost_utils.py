
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.externals import joblib
from multiprocessing import cpu_count
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

class xgb_utils(object):
    def __init__(self, detail_path, result_save_path, force_keep_col, maxpct=0.8,importancethea=0 ,train=False, grid_search=False):
        '''
        :param detail_path: 保存细节的地址
        :param result_save_path: 模型和最终结果保存的地址
        :param grid_search: 是否做网格搜索,默认是不做
        :param force_keep_col: 不参与剔除计算的列
        :param maxpct: 单列中最大单值占比,默认是做剔除,赋值False就不做
        :param importancethea: 筛选的阈值,默认做变量初筛,根据相关性来剔除高相关性的,赋值False不做
        :param train: 是否重新训练xgb模型
        :param delvar: 删除的变量
        '''
        self.detail_path = detail_path
        self.result_save_path = result_save_path
        self.grid_search = grid_search
        self.force_keep_col = force_keep_col
        self.maxpct = maxpct
        self.importancethea = importancethea
        self.train = train
        self.delvar ={}
        if os.path.exists(self.detail_path+"/"):
            pass
        else:
            os.makedirs(self.detail_path+"/")
        if os.path.exists(self.result_save_path+"/"):
            pass
        else:
            os.makedirs(self.result_save_path+"/")
    def transformdata(self, train, test, oot, keepcol):
        '''
        根据挑选好的变量对train, test, oot的数据做对应数据的剔除
        :param train: 训练集
        :param test: 测试集
        :param oot: 测试集
        :param keepcol: 要保存的标签
        :return: 处理好的训练集和测试集oot
        '''
        train = train[keepcol + self.force_keep_col]
        test = test[keepcol + self.force_keep_col]
        oot = oot[keepcol + self.force_keep_col]
        return train, test, oot

    def Maxpct_Check(self, train, test, oot):
        '''
        单列中最大的单值占比校验,这里的data需要把train,test,oot拼接起来
        :param data: train,test,oot拼接起来的值
        :return: 经过处理后的三个数据源和删除的变量
        '''
        data = train.append(test)
        if oot.empty:
            data = data.append(oot)
        maxpct = data.drop(self.force_keep_col, axis=1).apply(lambda x: max(x.astype(float).value_counts()) / data.shape[0])
        after_maxpcnt_col = list(maxpct[maxpct <= self.maxpct].index)
        print("***after maxpct check {} features left".format(len(after_maxpcnt_col)))
        delvars = list(set(train.drop(self.force_keep_col, axis=1)) - set(after_maxpcnt_col))
        self.delvar = {k: '最大单值占比超过80%' for k in delvars}
        train, test, oot = self.transformdata(train, test, oot, after_maxpcnt_col)
        return train, test, oot

    def Roungh_select(self, train, test, oot):
        '''
        利用变量重要度初筛部分变量
        :param train: 输入三部分数据
        :return: 三部分处理后的数据
        '''
        data = train.append(test)
        if oot.empty:
            data = data.append(oot)
        xgb_filter = xgb.XGBClassifier(n_jobs=-1)
        after_maxpcnt_col = list(data.drop(self.force_keep_col, axis=1))
        xgb_filter.fit(data[after_maxpcnt_col], data['target'])
        importances = pd.DataFrame({'col': after_maxpcnt_col, 'importance': xgb_filter.feature_importances_})
        thea = 0
        while importances[importances.importance > thea].shape[0] > 200:
            thea += 0.001
        after_importance_col = list(importances[importances.importance > thea].col)
        print("***after importances filter {} features left".format(len(after_importance_col)))
        delvars = list(set(train.drop(self.force_keep_col, axis=1)) - set(after_importance_col))
        self.delvar = {k: '重要度初筛' for k in delvars}
        train, test, oot = self.transformdata(train, test, oot, after_importance_col)
        return train, test, oot

    def auc_ks_cal(self, model, train, test, oot, verbose=True, returns=False):
        '''
        计算auc和ks值,观察模型效果
        :param model: 训练好的模型
        :param train: 三部分数据
        :return: 三部分数据的auc和ks
        '''
        if oot.empty:
            oot = test
        results = pd.DataFrame(columns = ['auc','ks'], index= ['train','test','oot'])
        trainX, testX, ootX = train.drop(self.force_keep_col,axis=1), test.drop(self.force_keep_col,axis=1), \
                              oot.drop(self.force_keep_col,axis=1)
        train_pred = model.predict_proba(trainX)[:,1]
        test_pred = model.predict_proba(testX)[:,1]
        oot_pred = model.predict_proba(ootX)[:,1]
        fpr, tpr, thresholds= metrics.roc_curve(np.array(train['target']), train_pred)
        fpr1, tpr1, thresholds1= metrics.roc_curve(np.array(test['target']), test_pred)
        fpr2, tpr2, thresholds2= metrics.roc_curve(np.array(oot['target']), oot_pred)
        results.loc['train','auc'] = metrics.roc_auc_score(train['target'].values,train_pred)
        results.loc['test', 'auc'] = metrics.roc_auc_score(test['target'].values, test_pred)
        results.loc['oot', 'auc'] = metrics.roc_auc_score(oot['target'].values, oot_pred)
        results.loc[:,'ks'] = [max(tpr-fpr),max(tpr1-fpr1),max(tpr2-fpr2)]
        trainkey, testkey, ootkey = train[self.force_keep_col], test[self.force_keep_col], oot[self.force_keep_col]
        if verbose:
            print(results)
        if returns:
            trainkey['prob'] = train_pred
            testkey['prob'] = test_pred
            ootkey['prob'] = oot_pred
            return trainkey, testkey, ootkey, results
        else:
            return results

    def GridSearch(self,train, test, oot):
        '''
        网格搜索
        :param train: 输入三个变量
        :return: 合适的网格参数
        '''
        trainX = train.drop(self.force_keep_col,axis=1)
        trainY = train['target']
        param_test1 = {'max_depth': np.arange(2, 5, 1), 'min_child_weight': np.arange(0, 6.0, 0.5)}
        gsearch1 = GridSearchCV(
            estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=20, scale_pos_weight=1, seed=27, verbose=1),
            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=3, verbose=1)
        gsearch1.fit(trainX, trainY)
        best_max_depth, best_min_child_weight = gsearch1.best_params_['max_depth'], gsearch1.best_params_[
            'min_child_weight']

        param_test2 = {'gamma': [i / 10.0 for i in np.arange(0, 5)]}
        gsearch2 = GridSearchCV(
            estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8,
                                    max_depth=best_max_depth,
                                    min_child_weight=best_min_child_weight, objective='binary:logistic', nthread=20,
                                    scale_pos_weight=1, seed=27, verbose=1),
            param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=3, verbose=1)
        gsearch2.fit(trainX, trainY)
        best_gamma = gsearch2.best_params_['gamma']

        param_test3 = {'subsample': [i / 10.0 for i in np.arange(6, 10)],
                       'colsample_bytree': [i / 10.0 for i in np.arange(6, 10)]}
        gsearch3 = GridSearchCV(
            estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=best_max_depth, gamma=best_gamma,
                                    min_child_weight=best_min_child_weight, objective='binary:logistic', nthread=20,
                                    scale_pos_weight=1, seed=27, verbose=1),
            param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=3, verbose=1)
        gsearch3.fit(trainX, trainY)
        best_colsample_bytree, best_subsample = gsearch3.best_params_['colsample_bytree'], \
                                                gsearch3.best_params_['subsample']

        param_test4 = {'reg_lambda': [0.01, 0.1, 0.05, 1, 5, 10, 50, 100, 200, 500]}
        gsearch4 = GridSearchCV(
            estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=best_max_depth, gamma=best_gamma,
                                    colsample_bytree=best_colsample_bytree, subsample=best_subsample,
                                    min_child_weight=best_min_child_weight, objective='binary:logistic', nthread=20,
                                    scale_pos_weight=1, seed=27, verbose=1),
            param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=2, verbose=1)
        gsearch4.fit(trainX, trainY)
        best_reg_alpha = gsearch4.best_params_['reg_lambda']

        param_test5 = {'n_estimators': np.arange(100, 401, 30)}
        gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, max_depth=best_max_depth, gamma=best_gamma,
                                                        colsample_bytree=best_colsample_bytree, subsample=best_subsample,
                                                        reg_alpha=best_reg_alpha,
                                                        min_child_weight=best_min_child_weight,
                                                        objective='binary:logistic',
                                                        nthread=20, scale_pos_weight=1, seed=27, verbose=1),
                                param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=2, verbose=1)
        gsearch5.fit(trainX, trainY)
        best_n_estimators = gsearch5.best_params_['n_estimators']

        best_xgb = xgb.XGBClassifier(learning_rate=0.08, n_estimators=best_n_estimators, max_depth=4, gamma=best_gamma,
                                     reg_lambda=best_reg_alpha, colsample_bytree=best_colsample_bytree,
                                     subsample=best_subsample, min_child_weight=best_min_child_weight,
                                     objective='binary:logistic', nthread=20, scale_pos_weight=1, seed=27, verbose=1)
        best_xgb.fit(trainX,trainY, eval_set=[(test.drop(self.force_keep_col, axis=1),test['target']),
                                              (oot.drop(self.force_keep_col, axis=1),oot['target'])],
                     eval_metric=['roc_auc'],early_stopping_rounds =20)
        pickle.dump(best_xgb, open(self.result_save_path+'/gridsearch_best_xgb.pkl','wb'))
        results = self.auc_ks_cal(best_xgb,train, test, oot)
        results.to_excel(self.result_save_path+'/gridsearch_ksauc.xlsx',index=False)
        return best_xgb

    def manul_finetune(self, train, test, oot, model, use_presele, check=False, roughsele=False, **parm):
        '''
        手动调整xgb的参数,但是一次只能对一个参数做范围搜索,
        如果parm不赋值,会使用传入的xgb模型或者默认的xgb模型,赋值的顺序是单值的在前面,多值的在最后
        :param model: 预定义参数的模型
        :param train: 三个数据集
        :param parm: 需要调整的参数字典{'参数':'参数范围'}
        :return: 调整好参数的模型
        '''
        if oot.empty:
            oot = test
        if check:
            train, test, oot = self.Maxpct_Check(train, test, oot)
            delvars = self.delvar
        if roughsele:
            train, test, oot = self.Roungh_select(train, test, oot)
            if len(delvars)==0:
                delvars = self.delvar
            else:
                delvars.update(self.delvar)
        if roughsele or check:
            pd.Series(delvars).to_csv(self.detail_path +'/剔除变量细节.csv', encoding='utf8')
        if use_presele:
            cols = pickle.load(open(self.result_save_path + '/final_feature.pkl', 'rb'))
            train = train[cols + self.force_keep_col]
            test = test[cols + self.force_keep_col]
            oot = oot[cols + self.force_keep_col]
            self.final_col = cols
        else:
            pickle.dump(list(train.drop(self.force_keep_col, axis=1).columns),
                        open(self.result_save_path + '/final_feature.pkl', 'wb'))
        trainX = train.drop(self.force_keep_col,axis=1)
        trainY = train['target']
        if not model:
            model = xgb.XGBClassifier(learning_rate =0.08, n_estimators=500, max_depth= 4, gamma=20,
                         colsample_bytree = 0.8, subsample = 0.7, reg_lambda=75.5, min_child_weight=10,
                         objective= 'binary:logistic',nthread=40,scale_pos_weight=0.8,seed=27,verbose=1)
        ksresult = pd.DataFrame()
        aucresult = pd.DataFrame()
        if len(parm) != 0:
            for i in parm.keys():
                if type(parm[i]) == int or type(parm[i]) == float:
                    exec("model." + i + "=" + str(parm[i]))
                    best_iters = []
                else:
                    for j in parm[i]:
                        exec("model."+i+"="+str(j))
                        print(model)
                        model.fit(trainX, trainY, eval_set=[(test.drop(self.force_keep_col, axis=1), test['target']),
                                                               (oot.drop(self.force_keep_col, axis=1), oot['target'])],
                                                               eval_metric=['auc'], verbose=True
                                                               , early_stopping_rounds=50)
                        best_iters.append(model.best_iteration)
                        results = self.auc_ks_cal(model, train, test, oot, verbose=False)
                        ksresult = ksresult.append(results.T.loc['ks',:])
                        aucresult = aucresult.append(results.T.loc['auc', :])
                    aucresult.rename(columns={'train':'trainauc', 'test':'testauc', 'oot':'ootauc'},inplace=True)
                    print(aucresult.reset_index(drop=True))
                    ksresult = pd.concat([ksresult.reset_index(drop=True),aucresult.reset_index(drop=True)],axis=1 )
                    ksresult['ks差'] =ksresult['train'] - ksresult['test']
                    ksresult['j'] = list(parm[i])
                    ksresult['best_iters'] = best_iters
                    ksresult = ksresult.sort_values(by=['ks差','trainauc'],ascending=[True,False])
                    ksresult.to_excel(self.detail_path+'/手动调整'+i+'模型结果细节.xlsx',index=False)
                    print(ksresult)
            exec("model." + i + " = " + str(ksresult.head(1).get('j').values[0]))
            pickle.dump(model, open(self.result_save_path + '/finetune'+i+'='+str(ksresult.head(1).get('j').values[0])+'_best_xgb.pkl', 'wb'))
        print(model)
        pickle.dump(model, open(self.result_save_path + '/final_best_xgb.pkl', 'wb'))
        self.model_performance(model)
        model.fit(trainX, trainY, eval_set=[(test.drop(self.force_keep_col, axis=1), test['target']),
                                            (oot.drop(self.force_keep_col, axis=1), oot['target'])],
                  eval_metric=['auc']
                  , early_stopping_rounds=50, verbose=True)
        results = self.auc_ks_cal(model, train, test, oot, verbose=False)
        print("final feature:",len(list(trainX.columns)))
        print(results)
        return model

    def model_performance(self,model):
        '''
        输出模型的cover,weight,gain
        :param model: 训练好的模型
        '''
        x1 = pd.Series(model.get_booster().get_score(importance_type='cover'))
        x2 = pd.Series(model.get_booster().get_score(importance_type='weight'))
        x3 = pd.Series(model.get_booster().get_score(importance_type='gain'))
        x = pd.concat([x1,x2,x3], axis=1)
        x.rename(columns={0: 'cover', 1: 'weight', 2: 'gain'},inplace=True)
        x.sort_values(by='weight', ascending=False).to_excel(self.result_save_path+'/model_performance.xlsx')

    def models(self, model, train, test, oot, grid_search= False):
        '''
        train、test、oot都是带force_keep_col的，grid_search的调用预测函数
        self.manul_finetune来输出model
        models方法适合给
        :param model: 传入自定义的模型,或者为None使用默认自定义的模型参数
        :param train: 输入数据
        :return: 返回模型和预测的三部分数据
        '''
        delvars = {}
        # 变量筛选预处理
        self.grid_search = grid_search
        if oot.empty:
            oot = test
        # 模型选择部分
        if self.grid_search:
            model = self.GridSearch(train, test, oot)
        # # 假如不做grid_search和fine_tune，传入默认的model
        if not grid_search and not model:
            model = xgb.XGBClassifier(learning_rate =0.08, n_estimators=500, max_depth= 4, gamma=20,
                         colsample_bytree = 0.8, subsample = 0.7, reg_lambda=75.5, min_child_weight=10,
                         objective= 'binary:logistic',nthread=40,scale_pos_weight=0.8,seed=27,verbose=1)
        trainX, testX, ootX = train.drop(self.force_keep_col,axis=1), test.drop(self.force_keep_col,axis=1), oot.drop(
            self.force_keep_col,axis=1)
        try:
            temp = model.predict_proba(trainX)
        except:
            model.fit(trainX, train['target'], eval_set=[(testX, test['target'])], eval_metric="auc",
                      early_stopping_rounds=20, verbose=True)
        self.model_performance(model)
        pickle.dump(model, open(
            self.result_save_path + '/final_best_xgb.pkl', 'wb'))
        pickle.dump(list(trainX.columns), open(self.result_save_path + '/final_feature.pkl', 'wb'))
        train_pred, test_pred, oot_pred, results = self.auc_ks_cal(model, train, test, oot, verbose=True, returns=True)
        return model, train_pred, test_pred, oot_pred, results

    def predicts(self,train, test, oot):
        '''
        加载已经保存好的final_best_xgb.pkl和final_feature.pkl,预测新的样本
        :param train: 待预测的样本,需要包含forcekeepcol
        :return: ,如果包含target,返回aucks预测值否则只返回预测值
        '''
        print("Using model from " + self.result_save_path + '/final_best_xgb.pkl')
        final_col = pickle.load(open(self.result_save_path + '/final_feature.pkl', 'rb'))
        model = pickle.load(open(self.result_save_path + '/final_best_xgb.pkl', 'rb'))
        train = train[final_col+self.force_keep_col]
        if 'target' in list(train.columns) and not train.empty and not oot.empty:
            test = test[final_col + self.force_keep_col]
            oot = oot[final_col + self.force_keep_col]
            train_pred, test_pred, oot_pred, results = self.auc_ks_cal(model, train, test, oot, verbose=True,
                                                                       returns=True)
            return train_pred, test_pred, oot_pred, model
        else:
            train['prob'] = model.predict_proba(train[final_col])[:,1]
            return train['prob'],None



if __name__=='__main__':
    # ph1 预处理特征筛选，如果做finetune可以不用先筛选
    train, test, oot, _ = xbgoost_utilss.Roungh_select(train, test, oot)
    train, test, oot, _ = xbgoost_utilss.Maxpct_Check(train, test, oot)
    # ph2: 新数据需要微调参,train,test,oot都是包含force_keep_col的dataframe,可以选择先对数据做预筛选，设置check和roughsele,
    # 每次可以对一个微调的参数做微调，返回一个训练好的参数,第一次使用先设置use_presele=False做预先筛选,后面设为True
    models = xgboost_utilss.manul_finetune(train, test, oot, use_presele=True, check=False, roughsele=False, model=None,
                                           scale_pos_weight=2, min_child_weight=5, reg_lambda=75.5,
                                           gamma=[21])
    # ph3: 微调后的模型做预测,默认是用已经经过预先筛选的变量，
    train_pred, test_pred, oot_pred, model = xgboost_utilss.predicts(train, test, oot)
    # ph4: 转换分数
    xgb_scorecards = xgb_scorecard(detail_path='./data/details',
                                   result_save_path='./data/modelResults',
                                   force_keep_col=['dataSet', 'target'])
    result = xgb_scorecards.xgb_scores(train, test, oot)

