import pandas as pd
import numpy as np
import time
import os
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import parameter

BASE_DIR = parameter.BASE_DIR
DEBUG = False

# Min-Max Scaling
scaler_minmax = MinMaxScaler()

## z 정규화 해야할 cols + 평균 0으로 가정
z_cols = ['openp', 'highp', 'lowp', 'closep',
          'sma5p', 'sma10p', 'sma20p', 'sma40p', 'sma60p','sma90p','sma120p', 'ema5p', 'ema20p','ema60p','ema120p','volp',
          'upperbandp', 'lowerbandp', 'atr', 'cci', 'adx']
## min_max cols
min_max_cols = ['rsi']

# 04.20 volp는 원래 없어야 해
drop_list = ['open', 'high', 'low', 'close', 'volume', 'sma10p', 'sma40p', 'sma90p', 'ema5p', 'ema20p','ema60p','ema120p']
# 시작연도
START_YEAR = parameter.START_YEAR

class Data:
    def __init__(self):
        # raw data
        self.data_1w = None
        self.data_1d = None
        self.data_4h = None
        self.data_1h = None
        self.data_15m = None
        self.data_5m = None
        self.data_1m = None

        # 정규화 + 관찰 데이터
        self.data_1w_obs = None
        self.data_1d_obs = None
        self.data_4h_obs = None
        self.data_1h_obs = None
        self.data_15m_obs = None
        self.data_5m_obs = None
        self.data_1m_obs = None

        # 정규화용 val 데이터(Test 시)
        self.data_1w_var = None
        self.data_1d_var = None
        self.data_4h_var = None
        self.data_1h_var = None
        self.data_15m_var = None

        # 제공하는 데이터 시간 단위
        self.time_frames = parameter.TIMEFRAMES
        self.data_attributes = [f"data_{tf}" for tf in self.time_frames]

    def load_data(self, ticker, test=False):
        start = time.time()
        path = os.path.join(BASE_DIR, 'candles_sub_data')
        for tf in self.time_frames:
            file_path = os.path.join(path, f"{ticker}_{tf}_sub.csv")
            if os.path.exists(file_path):
                data_frame = pd.read_csv(file_path).drop(columns='Unnamed: 0').dropna()
                setattr(self, f"data_{tf}", data_frame)
            else:
                print(f"[Error]: File {file_path} does not exist.")
        print("[Data]: load data completed. time=",time.time()-start)

    def make_var_table(self, ticker): # 0이 variance

        self.load_data(ticker)

        for attr in self.data_attributes:
            data = getattr(self, attr)
            data.replace([np.inf, -np.inf], 0, inplace=True)
            
            # 딕셔너리를 만드는 과정: 그 후 딕->시리즈로 변환된다! 
            var_series = pd.Series({col: data[col].var() for col in z_cols})
            path = os.path.join(BASE_DIR, "var_table")
            file_path = os.path.join(path, f"{ticker}_{attr}_var_table.csv")
            var_series.to_csv(file_path)

    def make_mean_var_table(self, ticker): # 0이 mean, 1이 variance

        self.load_data(ticker)

        for attr in self.data_attributes:
            data = getattr(self, attr)
            data.replace([np.inf, -np.inf], 0, inplace=True)

            mean_list = []
            var_list = []

            for col in z_cols:
                mean = data[col].mean()
                var = data[col].var()
                mean_list.append(mean)
                var_list.append(var)

            mean_var_list = [mean_list, var_list]
            df = pd.DataFrame(mean_var_list, columns=z_cols)
            path = os.path.join(BASE_DIR, "var_table")
            file_path = os.path.join(path, f"{ticker}_{attr}_var_table.csv")
            df.to_csv(file_path)

    # def load_data_1m(ticker):
    #     file_list = []
    #     for y in range(START_YEAR, 2024):
    #         for i in range(0, 4):
    #             file_list.append(f"candle_datas/{ticker}_1m_{y}_{i}_sub.csv")
    #     df = [pd.read_csv(file).drop(columns='Unnamed: 0').dropna() for file in file_list if os.path.exists(file)]
    #     df_combine = pd.concat(df, ignore_index=True)
    #     return df_combine

    # def load_data_5m(ticker):
    #     file_list = []
    #     for y in range(START_YEAR, 2024):
    #         file_list.append(f"candle_datas/{ticker}_5m_{y}_sub.csv")
    #     df = [pd.read_csv(file).drop(columns='Unnamed: 0').dropna() for file in file_list if os.path.exists(file)]
    #     df_combine = pd.concat(df, ignore_index=True)
    #     return df_combine


    # 정규화 전 평균을 0으로 가정한 normalization!
    def z_norm(self):
        start = time.time()
        for attr in self.data_attributes:
            print(attr)
            data = getattr(self, attr)
            data.replace([np.inf, -np.inf], 0, inplace=True)
            
            for col in z_cols:
                std_dev = math.sqrt(data[col].var())
                data[col] = data[col] / std_dev

        print("[Data]: z_norm completed. time=",time.time()-start)

    def download_mv(self, ticker):
        for attr in self.data_attributes:
            path = os.path.join(BASE_DIR, "var_table")
            file_path = os.path.join(path, f"{ticker}_{attr}_var_table.csv")
            if os.path.exists(file_path):
                data = pd.read_csv(file_path).drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
                setattr(self, f"{attr}_var", data)
            else:
                print(f"[Error]: File {file_path} does not exist.")

    def mm_norm(self):
        start = time.time()
        for attr in self.data_attributes:
            data = getattr(self, attr)
            for row in min_max_cols:
                data[row] = scaler_minmax.fit_transform(data[[row]])
        print("[Data]: mm_norm completed. time=",time.time()-start)
                
    def normalization(self):
        self.z_norm()
        self.mm_norm()

    # 미리 저장된 var로 정규화! 평균은 0으로 가정
    def z_norm_with_var(self):
        for attr in self.data_attributes:
            data = getattr(self, attr)
            data_var = getattr(self, attr+"_var")
            data.replace([np.inf, -np.inf], 0, inplace=True)
            for col in z_cols:
                std_dev = math.sqrt(data_var[col].iloc[1])
                data[col] = data[col] / std_dev

    def load_obs_data(self):
        for attr in self.data_attributes:
            data = getattr(self, attr)
            data_obs = data.drop(columns=drop_list)
            setattr(self, f"{attr}_obs", data_obs)

    def get_datas(self):
        # return 값은 list
        return [getattr(self, f"data_{tf}") for tf in self.time_frames]
    
    def get_obs_datas(self):
        # return 값은 list
        return [getattr(self, f"data_{tf}_obs") for tf in self.time_frames]
    
    def get_obs_datas_len(self):
        obs_data_len = 0
        for tf in self.time_frames:
            attr = f"data_{tf}_obs"
            data = getattr(self, attr)
            obs_data_len += data.shape[1]
        return obs_data_len

    def get_data_shape(self):
        if self.data_1d is not None:
            return len(self.data_1w.columns)
        else:
            return 0

    def load_train_data(self, ticker):
        self.load_data(ticker)
        self.normalization()
        self.load_obs_data()

    def load_test_data(self, ticker):
        self.load_data(ticker, test=True)
        self.download_mv(ticker)
        self.z_norm_with_var()
        self.mm_norm()
        self.load_obs_data()


if __name__ == '__main__':
    data = Data()
    for ticker in parameter.test_tickers:
        data.make_mean_var_table(ticker)



