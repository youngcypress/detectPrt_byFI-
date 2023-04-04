
from pathlib import Path
from tqdm import tqdm
import os, gc
from detect import Detect
import pandas as pd
import numpy as np
import exchange_calendars as xcals
from function import _qstart
from WindPy import w
import datetime

# 取数据
w.start()



def Days_reg(c, _w, alpha, info, start_time, end_time, path):
    """
    :param c: 上期持仓二次规划法正则系数
    :param _w: 加权方式
    :param alpha: 岭回归和Lasso回归正则系数
    :param info: 基金信息元组
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param path: 项目根路径
    :return: 得到指定区间下固收+基金的仓位探测，并保存在一个csv文件中
    """
    detect_prt = pd.DataFrame()
    Days = list(x.strftime('%Y-%m-%d') for x in xcals.get_calendar("XSHG").schedule.loc[start_time: end_time].index)
    for i in tqdm(info):
        detect = Detect(c, _w, alpha, i[0], i[1], start_time, end_time, path)
        prt_i = detect.anyDay_reg('window_reg', 'WLScons_amendQP_wreg', Days)
        print(f'{i[0]}全部仓位和波动率计算完成')
        print(prt_i)
        detect_prt = pd.concat([detect_prt, prt_i])
        detect_prt['code'] = i[0]
        # 垃圾回收，释放内存
        del prt_i, detect
        gc.collect()
    detect_prt.to_csv(f'{path}/result/data/所有固收+基金{start_time}to{end_time}仓位探测结果.csv')

def all_reg(c, _w, alpha, info, start_time, end_time, path):
    """
    :param c: 上期持仓二次规划法正则系数
    :param _w: 加权方式
    :param alpha: 岭回归和Lasso回归正则系数
    :param info: 基金信息元组
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param path: 项目根路径
    :return: 不同模型下完整探测区间的探测仓位结果和仓位的波动水平，这个函数就第一次使用，后期无需再完整探测
    """
    dfstd = pd.DataFrame()
    for i in tqdm(info):
        detect = Detect(c, _w, alpha, i[0], i[1], start_time, end_time, path)
        prt1, std1 = detect.all_reg('window_reg', 'WLScons_amendQP_wreg')

        std_dict = {'std1': std1}
        print(f'{i}全部仓位和波动率计算完成')
        dfstd_i = pd.DataFrame(index=[i[0]], data=std_dict)
        dfstd = pd.concat([dfstd, dfstd_i], axis=0)
        print(dfstd)
        path_data = f'{path}/result/data/{i[1]}/{i[0]}/{start_time}to{end_time}'
        if os.path.isdir(path_data):
            pass
        else:
            os.makedirs(path_data)
        with pd.ExcelWriter(f'{path_data}/仓位探测结果.xlsx') as xlsx:
            prt1.to_excel(xlsx, 'WLScons_amendQP_wreg')

        # 垃圾回收，释放内存
        del prt1, std1, dfstd_i, detect
        gc.collect()

    dfstd.loc['mean'] = dfstd.mean(axis=0)  # 默认为对列求均值
    dfstd.to_csv(f"{path}/result/所有基金探测仓位的平均标准差.csv")

def get_RMSE(c, _w, alpha, info, start_time, end_time, path, weighting_method):
    """
    :param c: 上期持仓二次规划法正则系数
    :param _w: 加权方式
    :param alpha: 岭回归和Lasso回归正则系数
    :param info: 基金信息元组
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param path: 项目根路径
    :param weighting_method: 加权方式，用以文件名的保存
    :return: 返回不同模型下完整探测区间内的所有固收+基金的平均RMSE，并保存在一个csv文件中
    """
    df_R = pd.DataFrame()
    for i in tqdm(info):
        detect = Detect(c, _w, alpha, i[0], i[1], start_time, end_time, path)
        R1 = detect.cal_RMSE('window_reg', 'WLScons_amendQP_wreg')  # 加权、c
        Rdict = {'R1': R1,}
        df_R_i = pd.DataFrame(index=[i[0]], data=Rdict)
        df_R = pd.concat([df_R, df_R_i], axis=0)
        print(f'{i[0]}RMSE计算完成')
        # 垃圾回收，释放内存
        del R1, Rdict, df_R_i, detect
        gc.collect()

    df_R.loc['mean'] = df_R.mean(axis=0)  # 默认为对列求均值
    df_R.to_csv(f'{path}/result/data/c={c},{weighting_method}加权,所有基金探测仓位的平均RMSE.csv')

def get_bestIndex(c, _w, alpha, info, start_time, end_time, path):
    """
    :param c: 上期持仓二次规划法正则系数
    :param _w: 加权方式
    :param alpha: 岭回归和Lasso回归正则系数
    :param info: 基金信息元组
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param path: 项目根路径
    :return: 得到所有固收+基金最优指数，并保存在一个csv文件中
    """
    df_bestIndex = pd.DataFrame()
    for i in tqdm(info):
        Test = Detect(c, _w, alpha, i[0], i[1], start_time, end_time, path)
        lasttime_period = Test.dfBound[Test.dfBound['lasttime_prt'].notnull()]['date'].to_list()[:760]
        BestIndex = pd.DataFrame()
        for d in lasttime_period:
            best_index = Test.filter_index(d)
            BestIndex = pd.concat([BestIndex, best_index])
        BestIndex['code'] = i[0]
        df_bestIndex = pd.concat([df_bestIndex, BestIndex])
    df_bestIndex.to_csv('best_index.csv')
    return df_bestIndex

def update_prtFund(info, start_time, end_time):
    """
    季度更新各个基金的报告期仓位，无需每天运行，请在所有基金报告仓位都出之后再更新
    :param info:
    :param start_time:上一个季度的交易日
    :param end_time: 当前最新交易日
    :return: 并将其添加到prt_fund.csv文件中
    """
    for i in tqdm(info):
        df_prt_fund = w.wsd(i[0],
                            "prt_bondtoasset,prt_cashtoasset,mmf_reverserepotoasset,prt_othertoasset,prt_stocktoasset,prt_governmentbondtoasset,prt_centralbankbilltoasset,prt_financialbondtoasset,prt_convertiblebondtoasset", start_time, end_time, "Period=Q;Days=Alldays", usedf=True)[1] /100
        df_prt_fund.fillna(0, inplace=True)
        df_prt_fund.columns = ['prt_rate_bond', 'prt_cash', 'prt_mmf', 'prt_other', 'prt_stock', 'prt_GB', 'prt_CBB', 'prt_FB', 'prt_conv_b']
        # 合并国债、中央票据、金融债、占比为利率债占比(占总资产比例)
        df_prt_fund['prt_rate_b'] = df_prt_fund['prt_GB'] + df_prt_fund['prt_CBB'] + df_prt_fund['prt_FB']
        # 用债券占比减去利率债占比和转债占比得到信用债占比
        df_prt_fund['prt_cred_b'] = df_prt_fund['prt_rate_bond'] - df_prt_fund['prt_rate_b'] - df_prt_fund['prt_conv_b']
        df_prt_fund = df_prt_fund.drop(['prt_rate_bond', 'prt_GB', 'prt_CBB', 'prt_FB'], axis=1)
        df_prt_fund = df_prt_fund.reset_index().rename(columns={'index': 'date'})
        df_prt_fund['date'] = df_prt_fund['date'].apply(lambda x: _qstart(-1, x))
        df_prt_fund['code'] = i[0]
        df_prt_fund.to_csv(f'{path}/data/prt_fund.csv', mode='a', header=False, encoding='utf_8_sig')

def update_dfnw(path):
    """
    每天更新基金收益率
    :param path: 当前文件根路径
    :return: 更新or不更新
    """
    df_nw = pd.read_csv(f"{path}/data/基金日收益率.csv", parse_dates=['date'], index_col='date')
    file_end = df_nw.index[-1]  # 旧文件的结束时间
    amend_file_end = _qstart(1, file_end)  # 旧文件的结束时间
    now_time = datetime.datetime.now().strftime('%Y-%m-%d')
    amend_end = _qstart(0, now_time)  # 修正当前日期，避免出现非交易日的情况
    # 不要重复运行
    if file_end < amend_end:
        fundCodes = df_nw.columns.to_list()
        df = w.wsd(','.join(fundCodes), "NAV_adj_return1", amend_file_end, now_time, usedf=True)[1]
        df.to_csv(f"{path}/data/基金日收益率.csv", mode='a', header=False)
        print('基金收益率更新成功')
    else:
        print('基金收益率无变化')

def update_dfIndex(path):
    """
    每天更新指数收益率
    :param path: 当前文件根路径
    :return: 更新or不更新
    """
    df_index = pd.read_csv(f"{path}/data/参考指数收益率.csv", parse_dates=['date'], index_col='date')
    file_end = df_index.index[-1]
    amend_file_end = _qstart(1, file_end)  # 旧文件的结束时间
    now_time = datetime.datetime.now().strftime('%Y-%m-%d')
    amend_end = _qstart(0, now_time)  # 修正结束时间，避免出现非交易日的情况
    # 不要重复运行
    if file_end < amend_end:
        indexCodes = df_index.columns.to_list()
        df = w.wsd(','.join(indexCodes), "pct_chg", amend_file_end, now_time, usedf=True)[1]
        df.to_csv(f"{path}/data/参考指数收益率.csv", mode='a', header=False)
        print('指数收益率更新成功')
    else:
        print('指数收益率无变化')



if __name__ == '__main__':

    start_time = '2022-12-31'
    # end_time = '2022-12-31'
    end_time = datetime.datetime.now().strftime("%Y-%m-%d")

    halflife, window = 30, 55
    _w = np.diag(list(map(lambda m: (0.5 ** (1 / halflife)) ** (window - m), list(range(window + 1))[1:])))  # 半衰期加权
    c = 100
    alpha = 10**(-5)
    path = Path(__file__).resolve().parent.parent
    df_new_info = pd.read_excel(f'{path}/data/固收+基金名单(回溯12个季度)_to_2022-12-31.xlsx')
    fund_class = df_new_info['投资类型(二级)'].to_list()
    fund_list = df_new_info['基金代码'].to_list()
    info = list(zip(fund_list, fund_class))

    # 首先每日更新基金和指数收益，需在每日6点以后运行
    update_dfnw(path)
    update_dfIndex(path)

    # 其次季度更新报告期仓位，需要再特定日期运行，每日更新请勿打开
    # update_prtFund(info, start_time, end_time)

    # 接着对所有基金进仓位探测，并将结果添加到仓位探测文件中
    Days_reg(c, _w, alpha, info, start_time, end_time, path)   # 指定区间探测所有基金寸尺至本地
    # get_bestIndex(c, _w, alpha, info, start_time, end_time, path)   # 计算最优指数存储至本地
    # get_RMSE(c, _w, alpha, info, start_time, end_time, path, '平方根加权')   # 计算RMSE均值并存储至本地

