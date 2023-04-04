# detectPrt_byFI-
基于市场上的固收+基金，本文基于二次规划法回归，以每日的基金收益为因变量，各资产指数收益为自变量，拟合回归各资产仓位，并通过历史报告期的真实仓位数据判定准确度，最后在测试的最优模型下进行仓位预测工作。
# 准备工作
## 前置库(如果安装了anaconda或使用anaconda的python解释器进行编译则只需要安装exchange_calendars)
- pandas、numpy、matplotlib：传统数据处理三件套
- tqdm：添加循环进度条
- pathlib：读取文件本地绝对路径
- scipy：python强大的数学计算库
- sklearn：机器学习
- plotnine：三大绘图库之一
- exchange_calendars：一款开源的获取交易日的库
- Windpy：国内金融数据软件商Wind的api，需要有wind账号才可开启，这个库非必须，如果实在没有，可以试着用其他的数据库重写更新基金收益率、指数收益率、真实仓位的代码，或者直接在excel中手动更新数据
# 文件说明
## function.py：包含了处理该项目的大部分模型函数
## detect.py：探测回归的主要文件，是一个面向对象（每只基金）的类
## main.py：主要运行函数，包含了处理数据的函数，以及每日更新数据的函数
