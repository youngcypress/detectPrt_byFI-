# -*- coding: utf-8 -*-

"""
@Time    : 20/8/2022 下午8:00
@Author  : SongBai Li
@FileName: function.py
@Software: PyCharm
@brief：固收+仓位探测用到的函数包
"""


# from sklearn.model_selection import train_test_split as TTS
# from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from scipy.optimize import shgo
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import exchange_calendars as xcals


color_set = [
    "#be002e", "#c0c0c0", "#FFA500", "#72001c", "#1c335b", "#ffaa96",
    "#dc3400", "#5b9bd5", "#f32e00", "#01b8aa", "#f2c80f", "#016e66"
]


def _qstart(n, day):
    # 由于频繁用到日期回跳，这里做一个私有函数，返回的是Timestamp格式a
    # 注意，这里如果最终的n=0，如果当天不是交易日，则默认返回前一个交易日，如果当前是交易日，则默认返回当前交易日
    xshg = xcals.get_calendar("XSHG")
    new_date = xshg.date_to_session(day, direction="previous")
    if n > 0:
        n += 1
        new_day = xshg.sessions_window(new_date, n)[-1]
    else:
        n -= 1
        new_day = xshg.sessions_window(new_date, n)[0]
    return new_day


#  建立一个带限制的通用多元线性回归模型
#  不根据报告期的仓位进行限制
#  X为自变量矩阵，格式为dateframe, y为因变量列向量，格式为dateframe
#  用法：scipy.optimize.shgo(func, bounds, args=(), constraints=None, n=None, iters=1, callback=None, minimizer_kwargs=None, options=None, sampling_method='simplicial')
#  bounds指决策变量的范围，也可以转化为constriants
#  shgo是scipy.optimize中的minimaze演化而来，只支持计算方式method=COBYlA，也就是constraints只支持ineq

def OLSreg(x, y):
    result = np.linalg.inv(x.T.dot(_w).dot(x)).dot(x.T).dot(_w).dot(y)
    score = 1 - np.sum((y - x.dot(result)) ** 2) / np.sum((y - y.mean()) ** 2)
    return result, score

def WLScons_linear_reg(X, y, _w, bnds):
    def my_func(x):
        #  很显然，若因变量为0（即基金涨跌幅为0），则使函数取最小值的系数全为0，此时所有仓位都是0
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(_w).dot(X.dot(x) - y)
        return result

    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

def WLSlasso_reg(X, y, _w, bnds, alpha):
    def my_func(x):
        #  很显然，若因变量为0（即基金涨跌幅为0），则使函数取最小值的系数全为0，此时所有仓位都是0
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(_w).dot(X.dot(x) - y) + alpha * np.sum(abs(x))
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

def WLSridge_reg(X, y, _w, bnds, alpha):
    def my_func(x):
        #  很显然，若因变量为0（即基金涨跌幅为0），则使函数取最小值的系数全为0，此时所有仓位都是0
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(_w).dot(X.dot(x) - y) + alpha * (x.T.dot(x))
        return result

    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res


# 建立一个无限制的通用多元线性回归模型
def linear_regression(X, y):
    OLS = LinearRegression(fit_intercept=False)  # 建立多元线性回归模型
    OLS.fit(X, y)  # 使用训练数据绘制模型
    prt = OLS.coef_.reshape(4,)
    print(prt)
    score = OLS.score()
    return prt, score

# WLS将仓位限制在历史最大最小值内的二次规划法
def WLSmaxmin_cons_wreg(X, y, bnds, _w):
    def my_func(x):
        n = X.shape[0]
        result = 0.5 * (1 / n) * (X.dot(x) - y).T.dot(_w).dot(_w).dot(X.dot(x) - y)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2},)
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

# WLS将仓位限制在历史前后期内的二次规划法
def WLSupdown_cons_wreg(X, y, bnds, _w):
    def my_func(x):
        n = X.shape[0]
        result = 0.5 * 1 / n * (X.dot(x) - y).T.dot(_w).dot(X.dot(x) - y)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res


# WLS基于上期持仓修正的二次规划法
def WLScons_amendQP_wreg(X, y, bnds, c, w, _w):
    def my_func(x):
        #  很显然，若因变量为0（即基金涨跌幅为0），则使函数取最小值的系数全为0，此时所有仓位都是0
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(_w).dot(X.dot(x) - y) + c/2 * (x - w).T.dot(x - w)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

# OLS基于上期持仓修正的二次规划法
def OLScons_amendQP_wreg(X, y, bnds, c, w):
    def my_func(x):
        #  很显然，若因变量为0（即基金涨跌幅为0），则使函数取最小值的系数全为0，此时所有仓位都是0
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(X.dot(x) - y) + c/2 * (x - w).T.dot(x - w)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res


def _lasso(x,y):
    alpharange = np.logspace(-10,-3, 200, base=10)
    lasso = Lasso(alpha=0.00000001, positive=True).fit(x,y)
    prt = lasso.coef_
    score = lasso.score(x,y)
    return prt, score

def ridgecv(x,y):
    ridgecv = RidgeCV(alphas=np.arange(0.000001, 0.0001, 0.000001), store_cv_values=True).fit(x, y)
    score = ridgecv.score(x, y)
    prt = ridgecv.coef_
    alpha = ridgecv.alpha_
    return prt, score, alpha

def lassocv(x,y):
    alpharange = np.logspace(-8, -5, 100, base=10)
    lassocv = LassoCV(alphas=alpharange, cv=5, positive=True).fit(x, y)
    score = lassocv.score(x, y)
    prt = lassocv.coef_
    alpha = lassocv.alpha_
    return prt, score, alpha

# 单日回归下基于上期持仓修正的二次规划法
def cons_amendQP_dreg(X, y, bnds, c, w):
    def my_func(x):
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(X.dot(x) - y) + c/2 * (x - w).T.dot(x - w)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

# 单日回归下将仓位限制在历史前后期内的二次规划法
def updown_cons_dreg(X, y, bnds):
    def my_func(x):
        n = X.shape[0]
        result = 0.5 * 1/n * (X.dot(x) - y).T.dot(X.dot(x) - y)
        return result
    def g1(x):
        return np.sum(x)  # 回归系数之和>=0
    def g2(x):
        return 1 - np.sum(x)  # 1-sum(x)>=0,即回归系数之和小于等于1
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2})
    res = shgo(my_func, bounds=bnds, constraints=cons)
    return res

# 计算均方根误差
def cal_RMSE(df):
    prt_reality = np.array(df[['prt_stock', 'prt_cred_b', 'prt_rate_b', 'prt_conv_b']])
    prt_detect = np.array(df[['prt_stock_x', 'prt_cred_b_x', 'prt_rate_b_x', 'prt_conv_b_x']])
    RMSE = []
    for i in range(prt_reality.shape[0]):
        RMSE_i = math.sqrt((np.sum((prt_reality[i] - prt_detect[i])**2)) / 4)
        RMSE.append(RMSE_i)
    RMSE = np.array(RMSE).mean()
    return RMSE.round(2)

# 元组排序
def sort_tuples(tuples):
    L = list(tuples)
    L.sort()
    T = tuple(L)
    return T

# 建立一个绘图函数
def draw_hot_corr(df, path):
    a = df.corr()
    plt.subplots(figsize=(9, 9), dpi=120)
    sns.heatmap(a, annot=True, vmax=1, square=True, cmap='Blues')
    plt.savefig(f'{path}/仓位相关系数矩阵热力图.png')
    plt.show()

# 绘制仓位分布
def draw_prt_season(df, x_name, y_name, fill_name, width, path, position=''):
    p = (ggplot(df, aes(x=x_name, y=y_name, fill=fill_name))
        + geom_col(width=width, position=position)
        + theme_classic()
        + scale_fill_manual(values=color_set)
        + theme(  # 对绘图的表现进行调整
            text=element_text(family="SimHei"),  # 设置黑体，可以显示中文
            axis_title_x=element_blank(),  # X轴标题为空
            axis_title_y=element_blank(),  # Y轴标题为空
            axis_text_x=element_text(angle=60),
            legend_title=element_blank(),
            legend_key_size=10,
            axis_line=element_line(size=0.5),
            axis_ticks_major=element_line(size=0.5),
            figure_size=(15, 10),
            legend_position='top'
        )
        + scale_y_continuous(expand=(0,0))
    )
    p.save(filename=f'{path}/{fill_name}图.png', units='in', dpi=1000)

# 绘制仓位趋势
def draw_prt_all(df, title, tick_name, path):
    # df = df.resample("8D").bfill()
    x_num = df.shape[0]
    x_detect = [i.strftime("%Y-%m-%d") for i in df[f'{tick_name}_x'].index]
    x_reality = [i.strftime("%Y-%m-%d") for i in df[f'{tick_name}'].index]
    y_detect = df[f'{tick_name}_x'].values
    y_reality = df[f'{tick_name}_y'].values
    plt.figure(figsize=(20, 8), dpi=300)
    plt.plot(range(len(x_detect)), y_detect, label=f"{tick_name}_detect", color=color_set[0])
    # plt.bar(x=x_reality, height=y_reality, label=f"{tick_name}_reality", color=color_set[1])
    plt.scatter(x_reality, y_reality, s=15, label=f"{tick_name}_reality", color=color_set[3])
    plt.xticks(range(0, len(x_detect), 50), list(x_detect)[::50], rotation=45)  # rotation刻度显示的名称的角度，以45度显示
    plt.legend(loc="best")
    plt.savefig(f'{path}/{title}历史趋势.png')
    plt.show()

