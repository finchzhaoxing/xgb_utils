
import collections
import os
import traceback
import warnings
import pandas as pd


def typeDetec(file_dir):
    fpath, ftype = os.path.splitext(file_dir)
    return fpath, ftype


def dataFrame2sheet(filePath, save_lst, sheetName_lst):
    '''
    将多个dataframe保存成一个excel文件的不同sheet
    :param filePath: str, 文件存储路径
    :param save_lst: list, 需要保存到不同sheet的dataframe组成的集合
    :param sheetName_lst: 不同sheet的名称

    '''
    fPath, fileType = typeDetec(filePath)
    try:
        if fileType == '.xls' or fileType == '.xlsx':
            excelWriter = pd.ExcelWriter(filePath, engine='openpyxl')
        else:
            warnings.warn("Only support '.xls', 'xlsx', got{}. was reset to default '.xlsx'".format(fileType))
            excelWriter = pd.ExcelWriter(fPath + '.xlsx', engine='openpyxl')
        if not isinstance(save_lst, list):
            save_lst = [save_lst]
        if not isinstance(sheetName_lst, list):
            sheetName_lst = [sheetName_lst]
        for i in range(len(save_lst)):
            if save_lst[i].empty or save_lst is None:
                pass
            else:
                save_lst[i].to_excel(excel_writer=excelWriter, sheet_name=sheetName_lst[i], index=None)
        excelWriter.save()
        excelWriter.close()
    except Exception as e:
        traceback.format_exc(e)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def upload(data, filePath, encodeType='utf-8'):
    '''
    数据上传到指定地址，支持'.csv', '.txt', '.xls' 和 '.xlsx'

    :param data: dataframe, 需要保存的数据集
    :param filePath: str, 文件上传路径
    :param encodeType: str, 保存文件编码方式, 建议'utf-8'

    :return:

    '''
    fPath, fileType = typeDetec(filePath)
    try:
        if fileType == '.csv' or not fileType:
            data.to_csv(filePath, index=False, encoding=encodeType)
        elif fileType == 'txt':
            data.to_csv(filePath, index=False, encoding=encodeType)
        elif fileType == '.xls':
            data.to_excel(filePath, index=False, encoding=encodeType)
        elif fileType == '.xlsx':
            data.to_excel(filePath, index=False, encoding=encodeType)
        else:
            warnings.warn(
                " only support '.csv','.txt','.xls' and '.xlsx' or none type!, got{}. was reset to defalut '.csv'".format(
                    fileType))
            data.to_csv(fPath + '.csv', index=False, encoding=encodeType)
    except Exception as e:
        traceback.print_exc("upload failed cause by: {}".format(str(e)))


def dumps(path, f, adds):
    '''
    保存: 
    
    1. 绘制的ks、auc曲线为.jpg格式, adds= '/pictures/ks.jpg或者/pictures/auc.jpg'
    2. 训练模型.pkl格式, adds= '/model/lr_model.pkl'
    3. 模型参数.csv格式, adds= '/params/param.csv'

    :param path: str, 路径
    :param f: model/plt/dataframe
    :param adds: 增添保存路径
    :return:
    '''
    try:
        fPath, _ = typeDetec(path)
        f1, t1 = typeDetec(adds)
        if t1 == '.png':
            mkdir(fPath + '/png')
            strlist = f1.split('/')
            f1 = '/' + strlist[-1] + '.png'
            f.savefig(fPath + '/png' + f1)
        elif t1 == '.pkl':
            mkdir(fPath + '/model')
            joblib.dump(f, fPath + '/model' + adds)
        elif t1 == '.xlsx':
            # mkdir(fPath + '/scorecard')
            upload(f, fPath + adds)
        elif t1 == '.csv':
            mkdir(fPath + '/params')
            upload(f, fPath + '/params' + adds)
    except Exception as e:
        traceback.format_exc(e)