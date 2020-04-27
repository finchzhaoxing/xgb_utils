# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:59:37 2019

@author: finch
自定义模块，主要是输出结果分数的分布和画图20190924
"""

import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from dataLoading import dumps
import numpy as np

def quantile_cut(df,col='score', bins=10):
    '''
    df 当前数据列dataframe
    返回分好的Bins
    '''
    df = df.to_frame()
    cutpoints = df[col].quantile(np.arange(0,1,1/bins))
    bucket = np.digitize(df[col], list(cutpoints)[1:])    
    df['bucketnum'] = bucket
    grouped = df.groupby('bucketnum')[col].min().to_dict()
#    print(grouped,bucket)
    df['min'] = df['bucketnum'].apply(lambda x:round(grouped.get(x),6))
    grouped = df.groupby('bucketnum')[col].max().to_dict()
    df['max'] = df['bucketnum'].apply(lambda x:round(grouped.get(x),6))
    df['bins'] = df['min'].astype(str) + '|' + df['max'].astype(str)
    return df['bins'], cutpoints

def bin_iv_ks_inone(data1,how='cut',dropcol=['id_card_no','card_name','target'], verbose=True):
    data=data1.copy()
    data.target=data.target.astype(int)
    bins=pd.DataFrame()
    col=list(data.copy().drop(dropcol,axis=1).columns)
    ivs=pd.DataFrame(index=col,columns=['iv','ks','覆盖率'])
    for i in col:
        data[i]=data[i].astype(float)
        # if len(data[i].unique()) <=2:
        #     print("删除 "+i+" :唯一值太少了")
        #     continue
        if verbose:
            print(i)
        if len(data[i].unique())<=2 or sum(data[i]>0)/data.shape[0]<0.20:
            if verbose:
                print("删除 "+i+" :唯一值少于3个或者缺失率大于0.2")
            continue
        if how=='cut':
            a,b=pd.cut(data[i],20, retbins=True)
        elif how=='cut1':
            a,b = quantile_cut(data[i], col=i, bins=25)
        else:
            a,b=pd.qcut(data[i],20,duplicates='drop',retbins=True)
        data['bins']=a
        temp=data[['bins','target']]
        binwoe=pd.pivot_table(temp,index=['bins'],columns=['target'],aggfunc=len,fill_value=1)
        binwoe.columns=list(binwoe.columns)
        binwoe=binwoe.reset_index()
        binwoe['badrate']=binwoe[1]/binwoe[1].sum()
        binwoe['goodrate']=binwoe[0]/binwoe[0].sum()
        binwoe['total_pct']=(binwoe[1]+binwoe[0])/data[i].shape[0]
        binwoe['odds']=binwoe['badrate']/binwoe['goodrate']
        binwoe['woe']=(binwoe['badrate']/binwoe['goodrate']).apply(lambda x:math.log(x))
        binwoe['badprob_cum']=binwoe['badrate'].cumsum()
        binwoe['goodprob_cum']=binwoe['goodrate'].cumsum()
        binwoe['iv']=(binwoe['badrate']-binwoe['goodrate'])*binwoe['woe']
        binwoe['ks']=binwoe['badprob_cum']-binwoe['goodprob_cum']
        binwoe['feature']=i
        binwoe=pd.DataFrame(binwoe)
        iv=binwoe.iv.sum()
        ks=abs(binwoe.ks).max()
        bins=bins.append(binwoe.sort_values('bins'))
        ivs.loc[i,'iv'] = iv
        ivs.loc[i,'ks'] = ks
        ivs.loc[i,'覆盖率']=sum(data[i]>0)/data.shape[0]
    ivs=ivs.sort_values(by='iv',ascending=False)
    return bins,ivs

def scoreplot(trainscore,title,figure_save_path=None):
    plt.figure(num=1, figsize=(8, 4))
    trainscore.hist(bins=10,label='train',histtype = 'bar',align = 'mid',density =True)
    trainscore.plot(kind='kde',style='r')
    plt.title(title)
    plt.xlabel('score')
    plt.xlim(trainscore.min()-trainscore.min()/10,trainscore.max()+trainscore.max()/10)
    if figure_save_path is not None:
        dumps(figure_save_path, plt, adds='/' + title + '.png')
    plt.show()
    plt.close()

def scorebins(trainscore,testscore,valscore=None,prob_score='score',cutway='cut1',figure_save_path=None):
    '''实现分数的划分区间+出分数的分布图
    -trainscore 三个分数都是score,target两列组成
    '''
    try:
        trainscore=trainscore[['target',prob_score]]
        testscore=testscore[['target',prob_score]]
    except:
        print("**maybe the X donot contain columns target and score**")
    binall=pd.DataFrame()
#    bintrain = toad.metrics.KS_bucket(trainscore[prob_score],trainscore['target'],bucket=10,method='quantile')
#    bintest = toad.metrics.KS_bucket(testscore[prob_score],testscore['target'],bucket=10,method='quantile')
    bintrain,ivs2=bin_iv_ks_inone(data1=trainscore,how=cutway,dropcol=['target'])
    bintest,ivs2=bin_iv_ks_inone(data1=testscore,how=cutway,dropcol=['target'])
    bintrain['part']='train'
    bintest['part']='test'
    binall=binall.append(bintrain)
    binall=binall.append(bintest)
    scoreplot(trainscore[prob_score],figure_save_path=figure_save_path,title='trainscore')
    scoreplot(testscore[prob_score],figure_save_path=figure_save_path,title='testscore')
    if valscore is None:
        pass
    else:
        binval,ivs2=bin_iv_ks_inone(data1=valscore[['target',prob_score]],how=cutway,dropcol=['target'])
#        binval = toad.metrics.KS_bucket(valscore[prob_score],valscore['target'],bucket=10,method='quantile')
        binval['part']='val'
        print(ivs2)
        binall=binall.append(binval)
        scoreplot(valscore[prob_score],figure_save_path=figure_save_path,title='ootscore')
    return binall

def psi_cal(train,test,dropcol):
    '''
    计算变量的psi,计算train和oot
    '''
    cols = list(train.drop(dropcol,axis=1).columns)
    cat_cols = list(train.drop(dropcol,axis=1).select_dtypes('object').columns)
    for col in cat_cols:
        try:
            train[col] = train.astype(float)
            test[col] = test.astype(float)
            cat_cols.remove(col)
        except:
            pass
    psitable = pd.DataFrame()
    for i in cols:
        if i not in cat_cols:
            if len(train[i].unique())==1:
                continue
            print(i)
            a0,b0 = pd.qcut(train[i],10,duplicates='drop',retbins=True)
            a,b = pd.cut(train[i],b0,duplicates='drop',retbins=True)
            temp = a.value_counts().sort_index().reset_index()
            temp.rename(columns={i:'A'}, inplace=True)
            temp['A_pct'] = temp['A']/train.shape[0]
            a1,b1 =pd.cut(test[i],b0,duplicates='drop',retbins=True)
            tempdict = a1.value_counts().sort_index().to_dict()
            temp['B'] = temp['index'].apply(lambda x:tempdict.get(x,1)).astype(float)
            temp['B_pct'] = temp['index'].apply(lambda x:tempdict.get(x,1)/test.shape[0]).astype(float)
            temp['A_Blog'] = np.log(temp['A_pct']/temp['B_pct'])
            temp['exp'] = (temp['A_pct'] - temp['B_pct'])*temp['A_Blog']
            temp['psi'] = temp['exp'].sum()
            temp['var'] = i 
            psitable = psitable.append(temp)
        else:
            print("no float dtypes pass:"+i)
    return psitable

def corr_filter(data01,testcol1,ivs1):
    deletefea=[]
    k=0
    for i in testcol1:
        if i in deletefea:
            continue
        for j in testcol1:
            if j==i or j in deletefea:
                continue
            corri=abs(data01[[i,j]].corr().replace(1,0)).max().max()
            if corri>0.8:
                if ivs1.loc[i,'iv']>ivs1.loc[j,'iv']:
                    deletefea.append(j)
                else:
                    deletefea.append(i)
        k=k+1
    var_tovif=[item for item in testcol1 if item not in set(deletefea)]
    return var_tovif

# 看分布差异
def filter_extreme_percentile(series,min = 0.025,max = 0.975): #百分位法
    series = series.sort_values()
    q = series.quantile([min,max])
    return np.clip(series,q.iloc[0],q.iloc[1])

def plots(x1,x2,title):
    plt.figure()
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40) #alpha是透明度
    plt.hist(x1,**kwargs)
    plt.hist(x2,**kwargs)
    plt.title(title)
    plt.legend(['asssit','target'])


if __name__=='main':
    '''使用样例'''
    trainscore=score_train[['target','score']]
    bintrain,ivs2=bin_iv_ks_inone(data1=trainscore,how='cut',dropcol=['target'])
    scoreplot(trainscore['score'],figure_save_path=figure_save_path,title='trainscore')
    for i in feature_lst:
        plots(filter_extreme_percentile(dfTrainAssist[i][dfTrainAssist[i]>=0],min = 0.025,max = 0.85),\
    filter_extreme_percentile(dfTrainTarget[i][dfTrainTarget[i]>=0],min = 0.025,max = 0.85),i)





