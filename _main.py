import pandas as pd
import matplotlib.pyplot as plt
import quandl
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from ta import *
from sklearn import preprocessing, neural_network, linear_model

import math

yf.pdr_override()

stock_code = 'MU'
start_date = '2011-01-01'
train_day_after = 10

filter_columns = [
    "open",
    "high",
    "low",
    "close",
    "adj close",
    "volume",
    "pe",
    "pb",
    "ps",
    "roe",
    "evebit",
    "evebitda",
    "fcf",
    "ncf",
    "de",
    "marketcap",
    "netmargin",
    "ebitdamargin",
    "divyield"
]

quandl.ApiConfig.api_key = 'mxDCJM-_R2uJWRX2KEBg'

technical = pdr.get_data_yahoo(stock_code, start=start_date)
technical.index.names = ["date"]
technical.columns = [c.lower() for c in technical.columns]

fundamental = quandl.get_table('SHARADAR/SF1', ticker=stock_code)
fundamental = fundamental.set_index('reportperiod')
fundamental.index.names = ["date"]
fundamental.sort_index(inplace=True)
fundamental = fundamental.drop(
    columns=[
        "ticker",
        "dimension",
        "calendardate",
        "datekey",
        "lastupdated"
    ]
)

daily_fundamental = quandl.get_table("SHARADAR/DAILY", ticker=stock_code)
daily_fundamental = daily_fundamental.set_index('date')
daily_fundamental = daily_fundamental[daily_fundamental.index >= start_date]
daily_fundamental.sort_index(inplace=True)
daily_fundamental = daily_fundamental.drop(
    columns=[
        "ticker",
        "lastupdated"
    ]
)

master = pd.DataFrame(
    technical,
    index=technical.index,
    columns=list(technical.columns) + list(fundamental.columns)
)

count = 0
for f in fundamental.index:
    date = master[master.index >= f].index[0]
    if count is 0:
        start_date = date
    for c in fundamental.columns:
        master.loc[master.index == date, c] = fundamental.loc[f, c]
    count += 1

for c in fundamental.columns:
    master[c] = master[c].interpolate(method='quadratic')
    master[c].fillna(method='ffill', inplace=True)

master = master[master.index >= start_date]
master.update(daily_fundamental)

master = master[filter_columns]

correlation = master.corr()['adj close']
correlation.transpose()

high_corr_columns = list(correlation[abs(correlation) >= 0.5].index)

# master.drop(
#     columns=list(correlation[abs(correlation) < 0.5].index)
# )

master = add_all_ta_features(
    master,
    "open",
    "high",
    "low",
    "adj close",
    "volume",
    fillna=True
)

# master.drop(
#     columns=["open", "high", "low", "close"]
# )
master['target_diff'] = pd.Series([None] * len(master.index), index=master.index)
print(master['target_diff'])
master.loc['target_diff', train_day_after:] = master.loc['adj close', :train_day_after]

master.to_csv('test.csv', sep=',', encoding='utf-8')

min_max_scaler = preprocessing.MinMaxScaler()
master_values_scaled = min_max_scaler.fit_transform(master.values)
master_normalized = pd.DataFrame(
    master_values_scaled, index=master.index, columns=master.columns)

training_x = master_normalized.values[0: math.floor(
    master_normalized.shape[0] / 3 * 2) - train_day_after]

training_y = master_normalized["adj close"].values[train_day_after: math.floor(
    master_normalized.shape[0] / 3 * 2)]


accuracy = 0

while(accuracy < 0.9):
    # model = neural_network.MLPRegressor(
    #     activation='tanh',
    #     solver='adam',
    #     max_iter=1000,
    #     learning_rate='invscaling',
    #     verbose=True,
    #     early_stopping=False,
    #     n_iter_no_change=10,
    #     # random_state=0
    # )
    model = linear_model.BayesianRidge()

    model.fit(training_x, training_y)

    accuracy = model.score(training_x, training_y)

    # print(training_x.shape, training_y[0])

predict_x = master_normalized.values[math.floor(
    master_normalized.shape[0] / 3 * 2) - train_day_after: -train_day_after]

predict_y = model.predict(predict_x)

actual_y = master_normalized["adj close"].values[math.floor(
    master_normalized.shape[0] / 3 * 2):]

predict_df = pd.DataFrame({"num": predict_y})

predict_df["num"] = predict_df["num"].rolling(5).mean()

accuracy = model.score(predict_x, actual_y)

print(accuracy)

plt.plot(predict_df["num"])
plt.plot(actual_y)
# plt.plot(master_normalized["adj close"].values[math.floor(
#     master_normalized.shape[0] / 3 * 2) - train_day_after: -train_day_after])

# plt.matshow(master.corr())
plt.show()
