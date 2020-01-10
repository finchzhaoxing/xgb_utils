# xgb_utils
xgb在风控模型中的应用封装

train=data[data.dataSet=='trainSet']
test=data[data.dataSet=='devSet']
oot=data[data.dataSet=='ootSet']

from xgboost_utils import xgb_utils
from xgboost_scorecard import xgb_scorecard
xgboost_utilss = xgb_utils(detail_path='./data/details_hds_xgb_xj',
                               result_save_path = './data/modelResults_hds_xgb_xj0',
                               grid_search=False,
                               force_keep_col=['target', 'dataSet']
        )
xgb_scorecards = xgb_scorecard(detail_path='./data/details_hds_xgb_xj',
                                   result_save_path='./data/modelResults_hds_xgb_xj0',
                                   force_keep_col=['target', 'dataSet'])
## 使用方法
输入数据中force_keep_col不参与标签计算的变量；
如果第一次使用，先设置use_presele=False, check=True, roughsele=False做初步的特征筛选；
后续使用，微调时设置use_presele=True, check=False, roughsele=False,最终最好的模型和入模变量会保存到result_save_path中；
每次可以对一个参数做微调的搜索，选择train_auc和test_差距最小的为最终结果
models = xgboost_utilss.manul_finetune(train, test, oot, use_presele=False, check=True, roughsele=False, model=None, scale_pos_weight=15,subsample = 0.7,min_child_weight=1,max_depth=5,gamma=20,
                                       reg_lambda=[5])
##最终模型转换分数
result = xgb_scorecards.xgb_scores(train, test, oot)
最终输出分数分布和模型结果
##仅预测结果
pred_train, pred_test, pred_val, model = xgboost_utilss.predicts(train, test, oot)