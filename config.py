import os
import pandas as pd
import datetime
import numpy as np
import pickle
target = "1M"
date_col = "datetime"
stock_col = "instrument"
industry_col = '行业'
market_value_col = '市值'
date_stock_lst = [date_col, stock_col]
industry_lst = [industry_col]
market_value_lst = [market_value_col]
target_lst = ["1M", "3M", "5M", "10M"]
bt_indicators = ["Cumulative Return", "Volatility (ann.)", "Sharpe", "Max Drawdown", "Sortino", "Omega"]
# factors_lst = [i for i in df.columns if i not in date_stock_lst+industry_lst+target_lst]
factors_lst = ['换手率相对波动率',
            '市净率',
            '市现率',
            '市盈率',
            '市销率',
            '涨跌幅',
            '净利润增速',
            '净资产收益率',
            '营业利润增速',
            '营业收入增速',
            '累计振动升降指标技术',
            '股东权益比率',
            '资金现金回收率',
            '交易量波动率',
            '成交量',
            '振幅',
            '换手率',
            '市值']

# prefix = 'csi300_'
prefix = 'data/'
fig_path = './figure/'
pic_postfix = '.png'
generateFilePath = lambda filename:os.path.join(fig_path, f"{filename}{pic_postfix}")
now_time = lambda:datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")

def check_type(df):
    if isinstance(df, np.ndarray):
        df = df.flatten()
    if isinstance(df, (pd.Series, pd.DataFrame)):
        df = df.values.flatten()
    return df

def _msfe_metrics(y_true, y_pred, weight=1):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    msfe = np.mean(weight * np.square((y_true -  y_true_mean) - (y_pred - y_pred_mean)))
    return msfe
    
def _r2cs_metrics(y_true, y_pred, weight=1):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    a = np.sum(weight * np.square((y_true -  y_true_mean) - (y_pred - y_pred_mean)))
    b = np.sum(weight * np.square(y_true - y_true_mean))
    r2cs = 1 - a/b
    return r2cs

def _biascs_metrics(y_true, y_pred, weight=1):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    a = np.sum(weight * (y_true - y_true_mean) * (y_pred - y_pred_mean))
    b = np.sum(weight * np.square((y_pred - y_pred_mean)))
    biascs = a/b
    return biascs

def cs_metrics(y_true, y_pred, date=None, metric="msfe", weight=1):
    y_true = check_type(y_true)
    y_pred = check_type(y_pred)
    date = check_type(date)
    
    metrics_dict = {"msfe": _msfe_metrics, "r2cs":_r2cs_metrics, "biascs":_biascs_metrics, 'all':[]}
    metrics_func = metrics_dict.get(metric)
    if date is not None:
        metrics_lst = []
        date_unique = np.unique(date)
        for _date in date_unique:
            tmp_y_true = np.array([y_true[i] for i, _bool in enumerate((date == _date)) if _bool])
            tmp_y_pred = np.array([y_pred[i] for i, _bool in enumerate((date == _date)) if _bool])
            if metrics_func is not None:
                metrics = metrics_func(tmp_y_true, tmp_y_pred, weight)
                metrics_lst.append(metrics)
        metric_result = np.mean(metrics_lst)
    else:
        metric_result = metrics_func(y_true, y_pred, weight)
    return metric_result
    
def evaluation(y_true, y_score, **kwargs):
    date = kwargs.get('date', None)
    metric = kwargs.get("metric", "msfe")
    weight = kwargs.get("weight", 1)
    metric_result = cs_metrics(y_true, y_score, date, metric, weight)
    return {f"{metric}": metric_result}

def save_obj(obj, file_name, save_path="./"):
    with open(os.path.join(save_path, f"{file_name}"), "wb") as f:
        pickle.dump(obj, f)
        
def load_obj(file_name, save_path="./"):
    with open(os.path.join(save_path, f"{file_name}"), "rb") as f:
        obj = pickle.load(f)
    return obj

def get_idxs(date, test_start_year, train_year_span=None, valid_year_span=2, test_year_span=2):
    '''数据集划分：
    train_year_span:训练集时间跨度，取None时默认从最小日期开始
    val_year_span:验证集时间跨度，取None或不大于0时，默认不划分验证集
    test_start_year:测试集开始日期
    test_year_span:测试集时间跨度
    '''
    date = pd.to_datetime(date)
    valid_year_span = 0 if valid_year_span is None else valid_year_span
    start_date = f"{test_start_year-valid_year_span-min(train_year_span, 16)}-01-01" if train_year_span else datetime.datetime.strftime(min(date), "%Y-%m-%d")
    
    # 数据集日期范围
    train_date = date[(date>=start_date)&(date<f"{test_start_year-valid_year_span}-01-01")]
    val_date = date[(date>=f"{test_start_year-valid_year_span}-01-01") & (date<f"{test_start_year}-01-01")] 
    test_date = date[(date>=f"{test_start_year}-01-01") & (date<f"{test_start_year+test_year_span}-01-01")]
    
    # 数据集位置范围
    train_idxs  = date.index[np.where(date.isin(train_date))[0]];
    val_idxs = date.index[np.where(date.isin(val_date))[0]]
    test_idxs = date.index[np.where(date.isin(test_date))[0]]
    return train_idxs, val_idxs, test_idxs

def filter_extreme(df, type="Percentile", only_data=True, truncated=True, **kwargs):
    '''
    args
    type: Percentile, MAD, Sigma
    kwargs: n_mad, n_sigma, _min, _max
    '''
    n_mad, n_sigma, _min, _max = kwargs.get("n_mad",5), kwargs.get("n_sigma",3), kwargs.get("_min", 0.1), kwargs.get("_max",0.9)
    train_df = kwargs.get("train_df", df)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        train_df = pd.DataFrame(train_df)
    columns = df.columns
    
    if type == "MAD":
        min_range, max_range = filter_extreme_MAD(train_df, n_mad)
    if type == "Sigma":
        min_range, max_range = filter_extreme_3sigma(train_df, n_sigma)
    if type == "Percentile":
        min_range, max_range = filter_extreme_percentile(train_df, _min, _max)
    if truncated:
        data = np.clip(df, min_range, max_range, axis=1)
        df = pd.DataFrame(data, columns=columns)
    else:
        df = {col:df[(df>min_range)[col] & (df<max_range)[col]][[col]] for col in df.columns}
    if only_data:
        return df
    else:
        return df, min_range, max_range

def filter_extreme_MAD(train_df, n=5): #MAD:中位数去极值
    median = train_df.quantile(0.5)
    new_median = ((train_df - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return min_range, max_range

def filter_extreme_3sigma(train_df, n=3): #3 sigma
    mean = train_df.mean()
    std = train_df.std()
    max_range = mean + n*std
    min_range = mean - n*std
    return min_range, max_range

def filter_extreme_percentile(train_df, min=0.05, max=0.95): #百分位法
    q = train_df.quantile([min, max])
    min_range, max_range = q.iloc[0], q.iloc[1]
    return min_range, max_range

import calendar
from dateutil.relativedelta import relativedelta


class DateTimeUtil():

    def get_cur_month(self):
        # 获取当前月
        return datetime.now().strftime("%Y-%m")

    def get_last_month(self, number=1):
        # 获取前几个月
        month_date = datetime.now().date() - relativedelta(months=number)
        return month_date.strftime("%Y-%m")

    def get_next_month(self, number=1):
        # 获取后几个月
        month_date = datetime.now().date() + relativedelta(months=number)
        return month_date.strftime("%Y-%m")

    def get_cur_month_start(self):
        # 获取当前月的第一天
        month_str = datetime.now().strftime('%Y-%m')
        return '{}-01'.format(month_str)

    def get_cur_month_end(self):
        # 获取当前月的最后一天
        '''
        param: month_str 月份，2021-04
        '''
        # return: 格式 %Y-%m-%d

        month_str = datetime.now().strftime('%Y-%m')
        year, month = int(month_str.split('-')[0]), int(month_str.split('-')[1])
        end = calendar.monthrange(year, month)[1]
        return '{}-{}-{}'.format(year, month, end)

    def get_last_month_start(self, month_str=None):
        # 获取上一个月的第一天
        '''
        param: month_str 月份，2021-04
        '''
        # return: 格式 %Y-%m-%d
        if not month_str:
            month_str = datetime.now().strftime('%Y-%m')
        year, month = int(month_str.split('-')[0]), int(month_str.split('-')[1])
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
        return '{}-{}-01'.format(year, month)

    def get_next_month_start(self, month_str=None):
        # 获取下一个月的第一天
        '''
        param: month_str 月份，2021-04
        '''
        # return: 格式 %Y-%m-%d
        if not month_str:
            month_str = datetime.now().strftime('%Y-%m')
        year, month = int(month_str.split('-')[0]), int(month_str.split('-')[1])
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        return '{}-{}-01'.format(year, month)

    def get_last_month_end(self, month_str=None):
        # 获取上一个月的最后一天
        '''
        param: month_str 月份，2021-04
        '''
        # return: 格式 %Y-%m-%d
        if not month_str:
            month_str = datetime.now().strftime('%Y-%m')
        year, month = int(month_str.split('-')[0]), int(month_str.split('-')[1])
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
        end = calendar.monthrange(year, month)[1]
        return '{}-{}-{}'.format(year, month, end)

    def get_next_month_end(self, month_str=None):
        # 获取下一个月的最后一天
        '''
        param: month_str 月份，2021-04
        '''
        # return: 格式 %Y-%m-%d
        if not month_str:
            month_str = datetime.now().strftime('%Y-%m')
        year, month = int(month_str.split('-')[0]), int(month_str.split('-')[1])
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        end = calendar.monthrange(year, month)[1]
        return '{}-{}-{}'.format(year, month, end)



# if __name__ == '__main__':
#     # 获取当前月
#     print('当前月', DateTimeUtil().get_cur_month())
#     # 获取上一个月
#     print('上一个月', DateTimeUtil().get_last_month())
#     # 获取上两个月
#     print('上两个月', DateTimeUtil().get_last_month(number=2))
#     # 获取下一个月
#     print('下一个月', DateTimeUtil().get_next_month())
#     # 获取下两个月
#     print('下两个月', DateTimeUtil().get_next_month(number=2))
#     # 获取当前月的第一天
#     print('当前月的第一天', DateTimeUtil().get_cur_month_start())
#     # 获取当前月的最后一天
#     print('当前月的最后一天', DateTimeUtil().get_cur_month_end())
#     # 获取上个月的第一天
#     print('上个月的第一天', DateTimeUtil().get_last_month_start())
#     # 获取下个月的第一天
#     print('下个月的第一天', DateTimeUtil().get_next_month_start())
#     # 获取上个月的最后一天
#     print('上个月的最后一天', DateTimeUtil().get_last_month_end())
#     # 获取下个月的最后一天
#     print('下个月的最后一天', DateTimeUtil().get_next_month_end())

