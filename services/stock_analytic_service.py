from .stock_scraper_service import StockScraperService
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class StockAnalyticService:

    def __init__(self, stock_code, past_day=250, predict_future_day=250, technical_only=True, dropout=0.2):
        self.stock_code = stock_code
        self.technical_only = technical_only
        self.past_day = past_day
        self.predict_future_day = predict_future_day
        self.dropout = dropout

        self.scaler = MinMaxScaler()
        self.resultScaler = MinMaxScaler()
        self.stock_scraper_service = StockScraperService()
        self.fetchData().augFeatures().normalize()

    def fetchData(self):
        self.stock_data = self.stock_scraper_service.get_historical_merged_data(
            self.stock_code, concurrent=True, technical_only=self.technical_only)
        return self

    def augFeatures(self):
        self.stock_data.index = pd.to_datetime(self.stock_data.index)
        self.stock_data["year"] = self.stock_data.index.year
        self.stock_data["month"] = self.stock_data.index.month
        self.stock_data["day"] = self.stock_data.index.day
        self.stock_data["day_of_week"] = self.stock_data.index.dayofweek
        self.stock_data["pct_change"] = self.stock_data["adj_close"].pct_change(
            periods=self.past_day)
        return self

    def exportStockData(self):
        self.stock_data.to_csv(self.stock_code + '.csv',
                               sep=',', encoding='utf-8')
        return self

    def exportNormalizedData(self):
        self.normalized_data.to_csv(
            self.stock_code + ' (Normalized).csv', sep=',', encoding='utf-8')
        return self

    def normalize(self):

        # excludePctChange = [x for x in list(self.stock_data.columns) if x not in ['pct_change']]

        self.normalized_data = self.scaler.fit_transform(self.stock_data)
        self.normalized_adj_close = self.resultScaler.fit_transform(self.stock_data[['adj_close']])

        # self.normalized_data = self.stock_data.apply(
        #     lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

        self.normalized_data = pd.DataFrame(self.normalized_data, index=self.stock_data.index, columns=self.stock_data.columns)
        self.normalized_adj_close = pd.DataFrame(self.normalized_adj_close, index=self.stock_data.index, columns=['adj_close'])

        # pct_change = self.normalized_data['pct_change']

        self.normalized_data = self.normalized_data.dropna(
            axis='columns',
            thresh=10
        )

        # self.normalized_data['pct_change'] = pct_change

        return self

    def buildTrain(self):

        self.X_train = []
        self.Y_train = []
        self.Real_Predict = []

        # excludePctChange = [x for x in list(self.normalized_data.columns) if x not in ['pct_change']]

        for i in range(self.past_day, self.normalized_data.shape[0]-self.predict_future_day-self.past_day):
            
            self.X_train.append(np.array(
                self.normalized_data.iloc[i:i+self.past_day] #[excludePctChange]
            ))

            self.Y_train.append(np.array(
                self.normalized_adj_close.iloc[i+self.past_day:i+self.past_day+self.predict_future_day]["adj_close"]))
            
            self.Real_Predict.append(np.array(
                self.normalized_data.iloc[i+self.past_day:i+self.past_day+self.predict_future_day])) #[excludePctChange]))

        # self.X_train = np.array(self.normalized_data.values[0: -predict_future_day])
        # self.Y_train = np.array(self.normalized_data["adj_close"].values[predict_future_day:])
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        self.Real_Predict = np.array(self.Real_Predict)

        return self

    def shuffleTrainData(self, data):
        np.random.seed(10)
        randomList = np.arange(data.shape[0])
        np.random.shuffle(randomList)
        return data[randomList]

    def shapeTrainData(self, valuation_rate):
        
        self.valuation_rate = valuation_rate
        
        X_train = self.X_train[:int(
            self.X_train.shape[0] * (1 - valuation_rate))]
        Y_train = self.Y_train[:int(
            self.Y_train.shape[0] * (1 - valuation_rate))]

        X_val = self.X_train[int(
            self.X_train.shape[0] * (1 - valuation_rate)):]
        Y_val = self.Y_train[int(
            self.Y_train.shape[0] * (1 - valuation_rate)):]

        # self.X_val = self.shuffleTrainData(self.X_val)
        # self.Y_val = self.shuffleTrainData(self.Y_val)

        Y_train = Y_train[:, :, np.newaxis]
        Y_val = Y_val[:, :, np.newaxis]

        self.X_train = X_train
        self.Y_train = Y_train

        self.X_val = X_val
        self.Y_val = Y_val

        return self

    def buildModel(self):
        shape = self.X_train.shape
        self.model = Sequential()

        self.model.add(LSTM(shape[2], input_length=shape[1], input_dim=shape[2], return_sequences=True))

        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(loss="mse", optimizer="adam")
        self.model.summary()

        return self

    def train(self):
        callback = EarlyStopping(
            monitor="loss", patience=10, verbose=1, mode="auto")
        self.train_history = self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=100,
            batch_size=128,
            # validation_split=self.valuation_rate,
            shuffle=True,
            verbose=1,
            validation_data=(self.X_val, self.Y_val),
            callbacks=[callback]
        )

    def evaluate(self):
        trainScore = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' %
              (trainScore, math.sqrt(trainScore)))

        testScore = self.model.evaluate(self.X_val, self.Y_val, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' %
              (testScore, math.sqrt(testScore)))

        # trainPredict = self.model.predict(self.X_train)
        testPredict = self.model.predict(self.X_val)
        realPredict = self.model.predict(self.Real_Predict)

        # plt.plot(trainPredict[0])
        # plt.plot(testPredict[-1])
        # plt.plot(self.Y_val[-1])
        # plt.plot(realPredict[-1])
        
        # plt.plot(trainPredict[-1])
        plt.plot(self.resultScaler.inverse_transform(testPredict[-1]))
        plt.plot(self.resultScaler.inverse_transform(self.Y_val[-1]))
        plt.plot(self.resultScaler.inverse_transform(realPredict[-1]))

        plt.show()