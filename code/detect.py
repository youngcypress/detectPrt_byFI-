# -*- coding: utf-8 -*-

"""
@Time    : 30/9/2022 下午8:00
@Author  : SongBai Li
@FileName: detect6.py
@Software: PyCharm
@version: level6
@brief:
自动探测固收+仓位第六代软件，所有原始数据已提前保存在本地excel中
探测的开始时间 = max{同类型基金净值公布最晚的那天，同类型基金仓位公布最晚的那天的第一个交易日}，结束时间为当前最新交易日
探测模型和各类函数位于function.py中
总开关位于main6.py中，只需运行main6.py即可，本文件仅供单只基金进行测试。
预测误差使用均方根误差来衡量，并且能够绘制探测后的趋势图
预测趋势波动水平用标准差来衡量
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import exchange_calendars as xcals
from function import _qstart, cal_RMSE, sort_tuples, updown_cons_dreg, cons_amendQP_dreg, WLScons_amendQP_wreg, WLSmaxmin_cons_wreg, WLSupdown_cons_wreg, OLScons_amendQP_wreg, WLSlasso_reg, WLSridge_reg, WLScons_linear_reg
import datetime


#显示所有列
pd.set_option('display.max_columns',None)
#显示所有行
pd.set_option('display.max_rows',None)
#设置value的显示长度
pd.set_option('max_colwidth', 100)
#设置1000列时才换行
pd.set_option('display.width', 1000)



""" 创建一个探测基金仓位的通用类 """
class Detect(object):
    def __init__(self, c, _w, alpha, fund_code, fund_class, start_time, end_time, path, window=55, rol=8):
        # 超参数λ(c)、加权方式(_w:平方根加权or半衰期加权)、基金代码、基金类别、结束日，回归窗口区间、滚动长度

        self.c = c
        self._w = _w
        self.alpha = alpha
        self.rol = rol
        self.window = window
        self.fund_code = fund_code
        self.fund_class = fund_class
        self.start_time = start_time
        self.end_time = end_time
        self.amend_start_time = _qstart(-(self.rol + self.window - 4), self.start_time)
        self.now = datetime.datetime.now()  # datetime.datetime格式
        self.path = f'{path}/data/'
        self.CredIndex = list(self._indexDict()[0].values())
        self.RateIndex = list(self._indexDict()[1].values())
        self.ConvIndex = list(self._indexDict()[2].values())
        self.StockIndex = list(self._indexDict()[3].values())
        self.df_nw = pd.read_csv(f"{self.path}/基金日收益率.csv", parse_dates=['date'], index_col='date').fillna(0)[self.fund_code]
        self.df_index = pd.read_csv(f"{self.path}/参考指数收益率.csv", parse_dates=['date'], index_col='date')
        self.df_nwroll = self.df_nw.rolling(rol).sum().dropna()
        self.df_indexroll = self.df_index.rolling(rol).sum().dropna()
        self.dfBound = self.get_dfBound()[0]
        self.df_prt = self.get_dfBound()[1][['date', 'prt_stock', 'prt_conv_b', 'prt_rate_b', 'prt_cred_b']]
        self.index_codes = self.df_index.columns

    def _indexDict(self):
        index_cred_b_code_dict = {
            '中证信用': 'H11073.CSI',
            "信用债综合": "CBA02701.CS",
            '中债信用债1年以下': 'CBA02711.CS',
            '中债信用债1-3年': 'CBA02721.CS',
            '中债信用债3-5年': 'CBA02731.CS',
            '中债信用债5-7年': 'CBA02741.CS',
            '中债信用债7-10年': 'CBA02751.CS',
            '中债信用债10年以上': 'CBA02761.CS'
        }
        index_rate_b_code_dict = {
            '中证国债': 'H11006.CSI',
            "利率债综合": "CBA05801.CS",
            '中债利率债1-3年': 'CBA05821.CS',
            '中债利率债3-5年': 'CBA05831.CS',
            '中债利率债5-7年': 'CBA05841.CS',
            '中债利率债7-10年': 'CBA05851.CS',
            '中债利率债10年以上': 'CBA05861.CS'
        }
        index_conv_b_code_dict = {'中证可转债': '000832.CSI'}
        index_share_code_dict = {'中证800': '000906.SH', '中证1000': '000852.SH', '中证500': '000905.SH',
                                 '沪深300': '000300.SH'}
        return index_cred_b_code_dict, index_rate_b_code_dict, index_conv_b_code_dict, index_share_code_dict

    def reg_bnds(self):
        bnds = {
            '混合债券型一级基金': [(0, 0.2), (0, 1), (0, 1), (0, 1)],
            '混合债券型二级基金': [(0, 0.2), (0, 1), (0, 1), (0, 1)],
            '灵活配置型基金': [(0, 0.3), (0, 1), (0, 1), (0, 1)],
            '中长期纯债型基金': [(0, 0), (0, 1), (0, 1), (0, 1)],
            '偏债混合型基金': [(0, 0.3), (0, 1), (0, 1), (0, 1)]
        }
        if self.fund_class == "混合债券型一级基金":
            bnds = bnds['混合债券型一级基金']
        elif self.fund_class == "混合债券型二级基金":
            bnds = bnds['混合债券型二级基金']
        elif self.fund_class == "灵活配置型基金":
            bnds = bnds['灵活配置型基金']
        elif self.fund_class == "中长期纯债型基金":
            bnds = bnds['中长期纯债型基金']
        elif self.fund_class == "偏债混合型基金":
            bnds = bnds['偏债混合型基金']
        return bnds

    def get_dfBound(self):
        df_prt_all = pd.read_csv(f'{self.path}/prt_fund.csv')
        df_prt = df_prt_all[df_prt_all['code'].isin([self.fund_code])]
        columns = ['prt_stock', 'prt_rate_b', 'prt_cred_b', 'prt_conv_b']
        EOQ_list = df_prt['date'].to_list()  # EOQ:end of quarter
        amend_EOQ_list = list(_qstart(1, day) for day in EOQ_list)
        EOQ_list.append(self.now.date())  # 加上现在的时间，然后将datetime.datetime转为datetime.date
        xshg = xcals.get_calendar("XSHG")
        date_tuple = list(zip(amend_EOQ_list, EOQ_list[1:]))
        df_bound = pd.DataFrame()
        for date in date_tuple:
            lasttime_prt = df_prt[columns].iloc[date_tuple.index(date)].to_list()
            if date_tuple.index(date) < len(date_tuple) - 1:
                updown_bnds = list(sort_tuples(i) for i in list(tuple(x) for x in df_prt[columns].iloc[date_tuple.index(
                    date): date_tuple.index(date) + 2].to_dict(orient='list').values()))
                if date_tuple.index(date) == 0:
                    maxmin_bnds = list(zip([0] * 4, df_prt[columns].iloc[0].to_list()))
                else:
                    maxmin_bnds = list((df_prt.iloc[:date_tuple.index(date) + 1][n].min(),
                                        df_prt.iloc[:date_tuple.index(date) + 1][n].max()) for n in columns)
            else:
                maxmin_bnds = list((df_prt.iloc[:date_tuple.index(date) + 1][n].min(),
                                    df_prt.iloc[:date_tuple.index(date) + 1][n].max()) for n in columns)
                updown_bnds = None
            df_bound_i = pd.DataFrame(
                {'date': list(x.strftime('%Y-%m-%d') for x in xshg.schedule.loc[date[0]: date[1]].index),
                 'updown_bnds': str(updown_bnds),
                 'maxmin_bnds': str(maxmin_bnds),
                 'lasttime_prt': str(lasttime_prt)}
            )
            df_bound = pd.concat([df_bound, df_bound_i])
        return df_bound, df_prt

    def get_slice(self, date):
        i = self.df_indexroll.index.get_loc(date)  # df.index.get_loc函数返回某指定索引在行上的隐式索引，即位置
        srsNav = self.df_nwroll.iloc[i - self.window + 1: i + 1]

        dfIndex = self.df_indexroll.iloc[i - self.window + 1: i + 1]
        # 筛选正常值，剔除掉非正常指数波动，即筛选那些净值涨跌幅小于等于三倍指数涨跌幅
        _t = srsNav.apply(np.abs) <= dfIndex.applymap(np.abs).max(axis=1) * 3  # df.max()默认为axis=0，代表列上的最大值
        ids = _t[_t].index
        srsNav = srsNav.loc[ids] * 100
        dfIndex = dfIndex.loc[ids] * 100
        return srsNav, dfIndex

    def doGroup(self, level1, lstGroup0, lstGroup1, dfIndex, lastIndex):
        # km = KMeans(n_clusters=2)
        # km.fit(t.loc[:, indexCodes].values.transpose())
        # srsRet = pd.Series(km.labels_, index=indexCodes)
        srsCorr = dfIndex.corr()[level1]
        select0 = pd.to_numeric(srsCorr[lstGroup0]).idxmin()
        select1 = pd.to_numeric(srsCorr[lstGroup1]).idxmin()
        Index_dict = {'ConvIndex': self.ConvIndex, 'CredIndex': self.CredIndex, 'RateIndex': self.RateIndex,
                      'StockIndex': self.StockIndex}
        test_df = pd.DataFrame(Index_dict.values(), index=Index_dict.keys())
        index0 = test_df.stack()[test_df.stack() == select0].index[0][0]
        index1 = test_df.stack()[test_df.stack() == select1].index[0][0]
        index2 = test_df.stack()[test_df.stack() == level1].index[0][0]
        index3 = test_df.stack()[test_df.stack() == lastIndex].index[0][0]
        indexdict = {index0: select0, index1: select1, index2: level1, index3: lastIndex}
        return indexdict

    def filter_index(self, date):
        # datelist = self.df_nwroll.iloc[self.window - 1:].index
        Index = pd.DataFrame({'StockIndex': '', 'RateIndex': '', 'CredIndex': '', 'ConvIndex': ''}, index=[0])
        # for date in datelist:
        srsNav, dfIndex = self.get_slice(date)
        # 用回归的方式寻找出对srsNav回归效果最好的指数，返回指数代码，一阶线性回归下的alpha与beta
        L = LinearRegression(fit_intercept=True)
        score = 0
        indexCodes = list(self.df_index.columns)
        for idCode in indexCodes:
            srsIndex = dfIndex[idCode]
            L.fit(srsIndex.values.reshape(-1, 1), srsNav.values.reshape(-1, 1))
            if L.score(srsIndex.values.reshape(-1, 1), srsNav.values.reshape(-1, 1)) >= score:
                score = L.score(srsIndex.values.reshape(-1, 1), srsNav.values.reshape(-1, 1))
                level1 = idCode
        # t = dfIndex / dfIndex.std()  # df.std()默认是列，返回每一列的标准差，默认除以n-1
        indexdict = ['ConvIndex', 'CredIndex', 'RateIndex', 'StockIndex']
        if any(x == level1 for x in self.CredIndex):
            # indexCodes = list(set(indexCodes) - set(self.CredIndex + self.ConvIndex))
            lstGroup0, lstGroup1 = self.RateIndex, self.StockIndex
            indexdict = self.doGroup(level1, lstGroup0, lstGroup1, dfIndex, self.ConvIndex[0])

        elif any(x == level1 for x in self.RateIndex):
            # indexCodes = list(set(indexCodes) - set(self.RateIndex + self.ConvIndex))
            lstGroup0, lstGroup1 = self.CredIndex, self.StockIndex
            indexdict = self.doGroup(level1, lstGroup0, lstGroup1, dfIndex, self.ConvIndex[0])

        elif any(x == level1 for x in self.StockIndex):
            # indexCodes = list(set(indexCodes) - set(self.StockIndex + self.ConvIndex))
            lstGroup0, lstGroup1 = self.RateIndex, self.CredIndex
            indexdict = self.doGroup(level1, lstGroup0, lstGroup1, dfIndex, self.ConvIndex[0])

        elif any(x == level1 for x in self.ConvIndex):
            # indexCodes = list(set(indexCodes) - set(self.StockIndex + self.ConvIndex))
            lstGroup0, lstGroup1 = self.RateIndex, self.CredIndex
            indexdict = self.doGroup(level1, lstGroup0, lstGroup1, dfIndex, '000300.SH')
        Index_i = pd.DataFrame(indexdict, index=[date])
        Index_i = Index_i[Index.columns]  # 确保顺序相同
        # Index = pd.concat([Index, Index_i], axis=0)
        # Index['code'] = self.fund_code
        return Index_i

        # 执行回归模型，返回模型在每天上的仓位结果和其准确度评分

    def do_reg(self, date, reg_way, reg_method):

        # _w为WLS加权矩阵，w为上期持仓，c为修正系数，C=1、10、100、1000
        window, df_nwroll, dfBound, c, _w, bnds = self.window, self.df_nwroll, self.dfBound.set_index('date'), self.c, self._w, self.reg_bnds()


        df_index = self.df_index[self.filter_index(date).loc[date]]
        i = df_nwroll.index.get_loc(date)  # df.index.get_loc函数返回某指定索引在行上的隐式索引，即位置
        if reg_way == "day_reg":
            y = df_nwroll.iloc[i]
            x = pd.DataFrame(df_index.iloc[i]).T
            if reg_method == 'updown_cons_dreg':
                bnds = eval(dfBound.loc[date]['updown_bnds'])
                result = updown_cons_dreg(x, y, bnds)
                prt_score = np.append(result.x, result.fun)
            if reg_method == 'cons_amendQP_dreg':
                w = np.array(eval(dfBound.loc[date]['lasttime_prt']))
                result = cons_amendQP_dreg(x, y, bnds, c, w)
                prt_score = np.append(result.x, result.fun)
        elif reg_way == 'window_reg':
            y = np.array(df_nwroll.iloc[i - self.window + 1: i + 1]).reshape(55, )
            x = np.array(df_index.iloc[i - self.window + 1: i + 1])
            if reg_method == 'WLScons_amendQP_wreg':
                w = np.array(eval(dfBound.loc[date]['lasttime_prt']))
                result = WLScons_amendQP_wreg(x, y, bnds, c, w, _w)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'WLSmaxmin_cons_wreg':
                bnds = eval(dfBound.loc[date]['maxmin_bnds'])
                result = WLSmaxmin_cons_wreg(x, y, bnds, _w)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'WLSupdown_cons_wreg':
                bnds = eval(dfBound.loc[date]['updown_bnds'])
                result = WLSupdown_cons_wreg(x, y, bnds, _w)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'OLScons_amendQP_wreg':
                w = np.array(eval(dfBound.loc[date]['lasttime_prt']))
                result = OLScons_amendQP_wreg(x, y, bnds, c, w)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'WLSlasso_reg':
                result = WLSlasso_reg(x, y, _w, bnds, self.alpha)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'WLSridge_reg':
                result = WLSridge_reg(x, y, _w, bnds, self.alpha)
                prt_score = np.append(result.x, result.fun)
            elif reg_method == 'WLScons_linear_reg':
                result = WLScons_linear_reg(x, y, _w, bnds)
                prt_score = np.append(result.x, result.fun)
        return prt_score

        # 对各个季度报告期计算均方根误，汇总计算平均值

    def cal_RMSE(self, reg_way, reg_method):
        date = self.df_prt['date'].to_list()

        amend_date = list(_qstart(0, d).strftime('%Y-%m-%d') for d in date)[1:]
        reg = pd.DataFrame()
        for d in amend_date:
            reg_d = self.do_reg(d, reg_way, reg_method)
            reg_d = pd.DataFrame(index=[d], data=[reg_d], columns=['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x', 'score'])
            reg = pd.concat([reg, reg_d], axis=0)

        reg['new_date'] = date[1:]
        reg_prt = reg.set_index('new_date')[['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x']]
        df_prt = self.df_prt.set_index('date')[['prt_stock', 'prt_rate_b', 'prt_cred_b', 'prt_conv_b']].loc[date[1:]]
        prt_all = pd.concat([reg_prt, df_prt], axis=1)*100
        RMSE = cal_RMSE(prt_all)
        return RMSE

        # 计算探测区间内所有交易日的仓位结果

    def all_reg(self, reg_way, reg_method):

        lasttime_period = self.dfBound[self.dfBound['lasttime_prt'].notnull()]['date'].to_list()[:760]
        updown_period = list(x.strftime('%Y-%m-%d') for x in xcals.get_calendar("XSHG").schedule.loc["2020-01-02": self.end_time].index)
        all_reg = pd.DataFrame()
        if reg_method == 'WLScons_amendQP_wreg' or reg_method == 'OLScons_amendQP_wreg':
            for d in lasttime_period:
                reg_d = self.do_reg(d, reg_way, reg_method)
                reg_d = pd.DataFrame(index=[d], data=[reg_d], columns=['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x', 'score'])
                all_reg = pd.concat([all_reg, reg_d], axis=0)
            all_reg = (all_reg[['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x']] * 100).round(2)
        elif reg_method == 'WLSupdown_cons_wreg' or reg_method == 'updown_cons_dreg':
            for d in updown_period:
                reg_d = self.do_reg(d, reg_way, reg_method)
                reg_d = pd.DataFrame(index=[d], data=[reg_d],columns=['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x', 'score'])
                all_reg = pd.concat([all_reg, reg_d], axis=0)
            all_reg = (all_reg[['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x']] * 100).round(2)
        return all_reg, all_reg.std().mean()

    def anyDay_reg(self, reg_way, reg_method, Days):
        """
        任意时间点的回归探测，但是最早时间不能超过start_time的时间
        :param reg_way: 窗口回归还是单日回归
        :param reg_method: 回归方式
        :param day: 时刻
        :return: 返回仓位探测结果
        """
        Days_reg = pd.DataFrame()
        if reg_method == 'WLScons_amendQP_wreg' or reg_method == 'OLScons_amendQP_wreg':
            for d in Days:
                reg_d = self.do_reg(d, reg_way, reg_method)
                reg_d = pd.DataFrame(index=[d], data=[reg_d], columns=['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x', 'score'])
                Days_reg = pd.concat([Days_reg, reg_d], axis=0)
        elif reg_method == 'WLSupdown_cons_wreg' or reg_method == 'updown_cons_dreg':
            for d in Days:
                reg_d = self.do_reg(d, reg_way, reg_method)
                reg_d = pd.DataFrame(index=[d], data=[reg_d],columns=['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x', 'score'])
                Days_reg = pd.concat([Days_reg, reg_d], axis=0)
        Days_reg = (Days_reg[['prt_stock_x', 'prt_rate_b_x', 'prt_cred_b_x', 'prt_conv_b_x']] * 100).round(2)
        return Days_reg





