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
from ScoreCard_new.dataLoading import load, excelAddSheet
from ScoreCard_new.validationUtils import get_ivs, load_csv, dumps, loads
import numpy as np
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
        if len(data[i].unique())<=2 or sum(data[i]>0)/data.shape[0]<0.30:
            if verbose:
                print("删除 "+i+" :唯一值少于3个或者缺失率大于0.3")
            continue
        if how=='cut':
            a,b=pd.cut(data[i],15, retbins=True)
        else:
            a,b=pd.qcut(data[i],10,duplicates='drop',retbins=True)
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
        bins=bins.append(binwoe,sort=False)
        ivs.loc[i,'iv']=iv
        ivs.loc[i,'ks']=ks
        ivs.loc[i,'覆盖率']=sum(data[i]>0)/data.shape[0]
    ivs=ivs.sort_values(by='iv',ascending=False)
    return bins,ivs

def scoreplot(trainscore,title,figure_save_path=None):
    plt.figure(num=1, figsize=(8, 4))
    trainscore.hist(bins=20,label='train',histtype = 'bar',align = 'mid',density =True)
    trainscore.plot(kind='kde',style='r')
    plt.title(title)
    plt.xlabel('score')
    plt.xlim(trainscore.min()-10,trainscore.max()+10)
    if figure_save_path is not None:
        dumps(figure_save_path, plt, adds='/png' + '/' + title + '.png')
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
    bintrain,ivs2=bin_iv_ks_inone(data1=trainscore,how=cutway,dropcol=['target'])
    bintest,ivs2=bin_iv_ks_inone(data1=testscore,how=cutway,dropcol=['target'])
    bintrain['part']='train'
    bintest['part']='test'
    binall=binall.append(bintrain,sort=False)
    binall=binall.append(bintest,sort=False)
    scoreplot(trainscore[prob_score],figure_save_path=figure_save_path,title='trainscore')
    scoreplot(testscore[prob_score],figure_save_path=figure_save_path,title='testscore')
    if valscore is None:
        pass
    else:
        binval,ivs2=bin_iv_ks_inone(data1=valscore[['target',prob_score]],how=cutway,dropcol=['target'])
        binval['part']='val'
        binall=binall.append(binval,sort=False)
        scoreplot(valscore[prob_score],figure_save_path=figure_save_path,title='ootscore')
    return binall



if __name__=='main':    
    trainscore=score_train[['target','score']]
    bintrain,ivs2=bin_iv_ks_inone(data1=trainscore,how='cut',dropcol=['target'])
    scoreplot(trainscore['score'],figure_save_path=figure_save_path,title='trainscore')
