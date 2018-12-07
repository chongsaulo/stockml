import json
import quandl
import fix_yahoo_finance as yf
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime
from ta import *


class StockScraperService:

    earliest_date = '2001-01-03'

    def __init__(self):
        yf.pdr_override()
        with open('config.json') as file:
            data = json.load(file)
            self.quandl_api_key = data['quandl']['api_key']
            quandl.ApiConfig.api_key = self.quandl_api_key

    def get_historical_prices(self, stock_code, start_date=None, end_date=None):

        if start_date is None:
            start_date = StockScraperService.earliest_date

        if end_date is None:
            end_date = datetime.today()

        data = pdr.get_data_yahoo(
            stock_code,
            start=start_date,
            end=end_date
        )
        data.index.names = ['date']
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]

        return data

    def get_historical_fundamentals(self, stock_code, start_date=None, end_date=None):
        data = quandl.get_table(
            'SHARADAR/SF1',
            ticker=stock_code
        )
        data = data.set_index('reportperiod')
        data.index.names = ['date']

        if start_date is not None and end_date is not None:
            data = data[(data.index >= start_date) & (data.index < end_date)]
        elif start_date is not None:
            data = data[data.index >= start_date]
        elif end_date is not None:
            data = data[data.index < end_date]

        data.sort_index(inplace=True)

        data = data.drop(
            columns=[
                'ticker',
                'dimension',
                'calendardate',
                'datekey',
                'lastupdated'
            ]
        )
        return data

    def get_daily_fundamentals(self, stock_code, start_date=None, end_date=None):
        data = quandl.get_table(
            'SHARADAR/DAILY',
            ticker=stock_code
        )

        data = data.set_index('date')

        if start_date is not None and end_date is not None:
            data = data[(data.index >= start_date) & (data.index < end_date)]
        elif start_date is not None:
            data = data[data.index >= start_date]
        elif end_date is not None:
            data = data[data.index < end_date]

        data.sort_index(inplace=True)
        data = data.drop(
            columns=[
                'ticker',
                'lastupdated'
            ]
        )
        return data

    def apply_technical_indicators(self, df):

        # df['ma10'] = df['adj_close'].rolling(10).mean()
        # df['ma20'] = df['adj_close'].rolling(20).mean()
        # df['ma50'] = df['adj_close'].rolling(50).mean()
        # df['ma100'] = df['adj_close'].rolling(100).mean()
        # df['ma250'] = df['adj_close'].rolling(250).mean()

        # df['newma_fast'] = df['low'].rolling(50).mean()
        # df['newma_slow'] = df['low'].rolling(50).mean().shift(-8)

        # df['ma40'] = df['adj_close'].rolling(40).mean()
        # df['ma60'] = df['adj_close'].rolling(60).mean()

        # df = df.dropna(subset=[
        #     'ma10',
        #     'ma20',
        #     'ma40',
        #     'ma50',
        #     'ma60', 
        #     'ma100',
        #     'ma250',
        #     'newma_fast',
        #     'newma_slow',
        # ])

        # df['high50'] =  df['high'].rolling(50).max()
        # df['low50'] =  df['low'].rolling(50).max()

        df = add_all_ta_features(
            df,
            "open",
            "high",
            "low",
            "adj_close",
            "volume",
            fillna=True
        )

        return df

    def get_historical_merged_data(self, stock_code, start_date=None, end_date=None, concurrent=False, remove_empty=True, technical_only=True):

        if not technical_only:
            fundamental = self.get_historical_fundamentals(
                stock_code,
                start_date,
                end_date
            )
            daily_fundamental = self.get_daily_fundamentals(
                stock_code,
                start_date,
                end_date
            )

        technical = self.get_historical_prices(
            stock_code,
            start_date,
            end_date
        )

        master = pd.DataFrame(
            technical,
            index=technical.index,
            columns=list(technical.columns) if technical_only else list(
                technical.columns) + list(fundamental.columns)
        )

        if not technical_only:
            for f in fundamental.index:
                date = master[master.index >= f].index[0]
                for c in fundamental.columns:
                    master.loc[master.index == date, c] = fundamental.loc[f, c]

            if concurrent:
                for c in fundamental.columns:
                    master[c] = master[c].interpolate(method='quadratic')
                    master[c].fillna(method='ffill', inplace=True)

            if len(daily_fundamental.index) is not 0:
                master.update(daily_fundamental)

            master = master[['open', 'high', 'low', 'adj_close', 'volume'] + list(daily_fundamental.columns)]

        master = self.apply_technical_indicators(master)

        if remove_empty:
            if not technical_only:
                master = master.dropna(subset=['adj_close', 'pe'])
            master = master.dropna(axis='columns', thresh=10)

        return master
