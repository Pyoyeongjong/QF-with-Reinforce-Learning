import gym
from gym import spaces
import numpy as np
import pandas as pd
from data import Data
import time
from collections import deque
from datetime import datetime
import parameter

DEBUG = False
TIMEDEBUG = False

LONG_TRANS_FEE = parameter.LONG_TRANS_FEE
SHORT_TRANS_FEE = parameter.SHORT_TRANS_FEE

test_tickers = parameter.test_tickers
tickers = parameter.tickers

# 파라미터
HOLD_REWARD = parameter.HOLD_REWARD

# Last Row Index -> get_next_obs 함수 최적화를 위해
lri = {}
# lri 초기화
def init_lri():
    lri.clear() # 초기화

class ChartTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, time_steps, tickers, budget):
        self.curr = 0 # 1분봉 기준 현재 행의 위치
        self.curr_ticker = 0 # tickers 리스트에서 현재 사용 중인 index 
        self.tickers = tickers
        self.ticker_done = False # ticker 교체를 해야 하는가?
        self.datas = Data()
        self.datas.load_train_data(self.tickers[self.curr_ticker]) # test용
        self.budget = budget
        self.initial_budget = budget
        super(ChartTradingEnv, self).__init__()
        # 0 = 산다, 1= 판다, 2 = 관망
        self.action_space = spaces.Discrete(3) # 0 = 롱, 1 = 숏, 2 = 관망
        self.long_action_space = spaces.Discrete(2) # 0 = 홀딩, 1 = 정리
        self.short_action_space = spaces.Discrete(2) # 0 = 홀딩, 1 = 정리
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.datas.get_obs_datas_len()-len(self.datas.data_attributes),), dtype=np.float64) # (data.shape[1], ) = (열, 1) # time 은 포함되어있음
        # LSTM 용 여러개 obs 저장용
        self.time_steps = time_steps # 0 이면 DNN 쓰는 거임.
        if time_steps > 0:
            self.lstm_obs = deque(maxlen=time_steps)
        if DEBUG:
            print("[BitcoinTradingEnv]: Env init OK")

        self.init = True
        

    # 현재 ticker가 끝났는지 확인하는 함수
    def ticker_is_done(self):
        if self.curr >= len(self.datas.data_1h): #self.datas.data_1h
            self.ticker_done = True
            return True
        else:
            self.ticker_done = False
            return False
        
    # curr에 대응하는 time frame의 ohlcv 제공. Return : Series
    def get_curr_ohlcv(self, ohlcv):
        return self.datas.data_1h.loc[self.curr, ohlcv] # 1h

    # curr의 다음 행을 가져오는 함수.
    def get_next_row_obs(self):

        self.curr += 1
        # 만약 다음 행이 없다면?
        if self.ticker_is_done():
            return None, None

        start = time.time()
        timestamp = [604800, 86400, 14400, 3600, 900, 300, 60] * 1000
        ohlcv_list = ['open', 'high', 'low', 'close', 'volume'] # OHLCV
        datas = self.datas.get_obs_datas()
        
        curr_time = self.datas.data_1h.loc[self.curr, 'time'] # 1m
        rows_list = []

        for data_loc, data in enumerate(datas):
            if data_loc not in lri:
                lri[data_loc] = -1

            # 1h 에 대해선 안해도 됨.
            if data_loc >= len(datas) - 1:
                lri[data_loc] = self.curr
            else:
                # 만약 초기값 상태라면, 1h time보다 작은 가장 큰 행 위치를 lri에 기록
                if lri[data_loc] < 0:
                    pos = data['time'].searchsorted(curr_time, side='right') - 1 # 이진탐색
                    if pos >= 0:
                        lri[data_loc] = pos
                    else: # 해당하는 row를 찾지 못함.
                        print("Row not find")
                        return None, None

                # row보다 1m time + 시간프레임이 크다면 index += 1
                if (lri[data_loc] < len(data) - 1 and
                    curr_time - data.loc[lri[data_loc], 'time'] >= timestamp[data_loc] * 2):
                    lri[data_loc] += 1

            if lri[data_loc] > 0:
                if DEBUG:
                    print(datetime.fromtimestamp(int(data.iloc[lri[data_loc]-1]['time'])/1000))
                row = data.iloc[lri[data_loc]-1].drop("time").values
            else:
                row = np.zeros(len(data.columns) - 1)  # 기본값으로 채움

            # Debug
            # print(datetime.fromtimestamp(int(row['time'])/1000))

            rows_list.append(row)

        rows = np.concatenate(rows_list)
        
        # row에 NaN이 있는지 검사
        if np.isnan(rows).any():
            print("There is nan data.")
            return None, None
        
        # 1m ohlcv data
        ohlcv = self.get_curr_ohlcv(ohlcv_list)

        if self.time_steps == 0:
            if DEBUG:
                print("")
            if TIMEDEBUG:
                print(f"get_next_row_obs(): takes time{time.time()-start}")
            return ohlcv, rows
        else: # LSTM 용 Next_OBS
            self.lstm_obs.append(rows)
            if len(self.lstm_obs) < self.time_steps:
                ohlcv, state = self.get_next_row_obs()
            if TIMEDEBUG:
                print(f"get_next_row_obs(): takes time{time.time()-start}")
            return ohlcv, self.lstm_obs


    # 손익률 구하는 함수
    def cal_percent(self, before, after):
        return ((after - before) / before)
        
    # 여기서 return하는 done은 현재 포지션의 종료 여부다.
    # return 값 : next_obs, 손익값, 종료 여부, 정보
    def long_or_short_step(self, action, position, short):

        # 행동의 다음 봉
        ohlcv, obs = self.get_next_row_obs()
        # 판단할 수 없을 때 = 보통 ticker_done 일 때이다.
        if obs is None: 
            return None, None, None, ["Obs is None"]
        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = ??? 잘 생각해보자, done = False, info = 대충 없음.
        if action == 0:
            done = False
            # percent = self.cal_percent(position, ohlcv['close']) - self.cal_percent(position, ohlcv['open'])
            percent = 0
            # 홀딩 추가지원금
            percent += HOLD_REWARD
        # action이 정리(1) 이면
        # obs = 다음 행, reward = 포지션에 대한 이득??? 잘 생각해보자, done = True, info = 대충 없음.
        elif action == 1:
            done = True
            # percent = 0
            percent = self.cal_percent(position, ohlcv['open'])
            if short:
                percent = -percent
                percent = percent - SHORT_TRANS_FEE
            else:
                percent = percent - LONG_TRANS_FEE
        # 청산가 계산
        long_sl = parameter.get_long_sl(position)
        short_sl = parameter.get_short_sl(position)
        # 강제 청산
        if ohlcv['low'] < long_sl and short == False: # 롱 청산 
            percent = self.cal_percent(position, long_sl)
            percent = percent - LONG_TRANS_FEE * 2
            done = True
            info = [long_sl]
        elif ohlcv['high'] > short_sl and short == True: # 숏 청산 
            percent = - self.cal_percent(position, short_sl)
            percent = percent - SHORT_TRANS_FEE * 2
            done = True
            info = [short_sl]
        else:
            info = [ohlcv['open']]
        return obs, percent, done, info

    def reset(self, test):

        if self.init == False:
            if self.curr_ticker >= len(tickers) - 1:
                self.curr_ticker = 0
            else:
                self.curr_ticker += 1
        else:
            self.init = False

        if test:
            print("[Env]: Reset. ticker: ",test_tickers[self.curr_ticker])
        else:
            print("[Env]: Reset. ticker: ",tickers[self.curr_ticker])

        self.budget = self.initial_budget
        self.ticker_done = False
        init_lri()

        if test:
            self.datas.load_test_data(test_tickers[self.curr_ticker])
        else:
            self.datas.load_train_data(tickers[self.curr_ticker])

        self.curr = 0
        self.lstm_obs = deque(maxlen=self.time_steps)

        return self.get_next_row_obs()
    
    ### Test용 함수
    def set_curr_to_timestamp(self, timestamp):
        while( self.datas.data_1h.loc[self.curr, 'time'] < timestamp ):
            # print(self.datas.data_1h.loc[self.curr, 'time'])
            self.curr += 1
        return
    

if __name__ == '__main__':
    env = ChartTradingEnv(time_steps=0, tickers=tickers, budget=10000)
    env.reset(test=False)
    env.reset(test=False)
    env.reset(test=False)
    env.get_next_row_obs()
    env.get_next_row_obs()
