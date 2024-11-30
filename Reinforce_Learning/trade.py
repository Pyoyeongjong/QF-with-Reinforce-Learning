# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
import logging
import argparse
import random

# A2C
from a2c import A2Cagent
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

# 경고 무시
import warnings
from sklearn.exceptions import DataConversionWarning

# 텔레그램
import telegram
import asyncio

# 반내림, 루트
from decimal import Decimal, ROUND_DOWN
import math

### Telegram
import talib

# 시간 동기화
import win32api
import time
import concurrent.futures
from datetime import datetime
from datetime import timedelta

# 보조지표 계산/출력 라이브러리
import math
import matplotlib.pyplot as plt

# Numpy / pandas
import numpy as np
import pandas as pd
import pytz

# CSV파일
import os
import csv

# 로그
import logging
import parameter

BASE_DIR = os.path.expanduser("C:\\Users\\pyjon\\repos\\QF-with-Reinforce-Learning\\Reinforce_Learning")
GAMMA = parameter.GAMMA
BATCH_SIZE = parameter.BATCH_SIZE
act_threshold = 0.9

# Setup logging
log_file_path = "trading_bot.log"  # 로그 파일 경로 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),  # 로그를 파일에 기록
    logging.StreamHandler()  # 로그를 콘솔에 출력
])
logger = logging.getLogger(__name__)

# 특정 경고 무시
warnings.filterwarnings(action='ignore', category=UserWarning)

# 상수값
LONG_TYPE = 0
SHORT_TYPE = 1
ALL_TYPE = 2

ticks = ["SOL"]
# Ticker
tick = ["ETH", "SOL", "XRP", "ADA", "AVAX", 
        "DOT", "STX", "ETC", "XLM", "XTZ", "DOGE"]



# 클라이언트 변수
home = BASE_DIR

class Binance_Client:
    def __init__(self, name, api, token, chat_id):
        self.name = name
        self.client = create_client(api)
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id
        get_usdt_balance(self.client)

        self.TradeCenters = []
        self.ticker_center = Ticker_Center()

    def create_tradecenter(self, tick, model, loss, amt):
        center = TradeCenter(tick, model, loss)
        self.TradeCenters.append(center)
        self.ticker_center.set_tick_amt(tick, amt)

# 차트 데이터 다운로드
def get_klines(client, symbol, limit, interval):
    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote', 'trade_num', 'taker_buy_base',
                'taker_buy_quote', 'ignored']
    data = pd.DataFrame(klines, columns=col_name)
    data = data[['time', 'open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    return data

def get_candle_subdatas(candles):

    if candles.empty :
        return pd.DataFrame()

    ### 데이터 분석
    # 문자열 -> 숫자 변환 && Pd Series
    close = candles['close'].apply(pd.to_numeric).to_numpy() # 종가 값 활용
    high = candles['high'].apply(pd.to_numeric).to_numpy()
    low = candles['low'].apply(pd.to_numeric).to_numpy()
    volume = candles['volume'].apply(pd.to_numeric).to_numpy()

    front_close = candles['close'].shift(1).apply(pd.to_numeric).to_numpy()

    highr = ( candles['high'] - front_close ) / front_close * 100
    highr.name = 'highp'
    lowr = ( candles['low'] - front_close ) / front_close * 100
    lowr.name = 'lowp'
    closer = ( candles['close'] - front_close ) / front_close * 100
    closer.name = 'closep'
    openr = ( candles['open'] - front_close ) / front_close * 100
    openr.name = 'openp'

    # Numpy밖에 못 쓴다 -> .to_numpy()
    sma5 = (pd.Series(talib.SMA(close, timeperiod=5), name="sma5") - candles['close']) / candles['close'] * 100
    sma5.name = 'sma5p'
    sma10 = (pd.Series(talib.SMA(close, timeperiod=10), name="sma10") - candles['close']) / candles['close'] * 100
    sma10.name = 'sma10p'
    sma20 = (pd.Series(talib.SMA(close, timeperiod=20), name="sma20") - candles['close']) / candles['close'] * 100
    sma20.name = 'sma20p'
    sma40 = (pd.Series(talib.SMA(close, timeperiod=40), name="sma40") - candles['close']) / candles['close'] * 100
    sma40.name = 'sma40p'
    sma60 = (pd.Series(talib.SMA(close, timeperiod=60), name="sma60") - candles['close']) / candles['close'] * 100
    sma60.name = 'sma60p'
    sma90 = (pd.Series(talib.SMA(close, timeperiod=90), name="sma90") - candles['close']) / candles['close'] * 100
    sma90.name = 'sma90p'
    sma120 = (pd.Series(talib.SMA(close, timeperiod=120), name="sma120") - candles['close']) / candles['close'] * 100
    sma120.name = 'sma120p'

    ema5 = (pd.Series(talib.EMA(close, timeperiod=5), name="ema5") - candles['close']) / candles['close'] * 100
    ema5.name = 'ema5p'
    ema20 = (pd.Series(talib.EMA(close, timeperiod=20), name="ema20") - candles['close']) / candles['close'] * 100
    ema20.name = 'ema20p'
    ema60 = (pd.Series(talib.EMA(close, timeperiod=60), name="ema60") - candles['close']) / candles['close'] * 100
    ema60.name = 'ema60p'
    ema120 = (pd.Series(talib.EMA(close, timeperiod=120), name="ema120") - candles['close']) / candles['close'] * 100
    ema120.name = 'ema120p'

    rsi = pd.Series(talib.RSI(close, timeperiod=14), name="rsi")
    volume_sma = pd.Series(talib.SMA(volume, timeperiod=20), name="vol_sma") / candles['volume']
    volume_sma.name = "volp"

    ### 한국 시간으로 맞춰주기 + DateTime으로 변환
    # korea_tz = pytz.timezone('Asia/Seoul')
    # datetime = pd.to_datetime(candles['time'], unit='ms')
    # candles['time'] = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 볼린저 밴드
    upperband, middleband, lowerband = talib.BBANDS(candles['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    upperband = ( upperband - candles['close'] ) / candles['close'] * 100
    upperband.name = "upperbandp"
    lowerband = ( lowerband - candles['close'] ) / candles['close'] * 100
    lowerband.name = "lowerbandp"
    # atr
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), name="atr")
    # macd
    # macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # macd = pd.Series(macd, name="macd")
    # macdsignal = pd.Series(macdsignal, name="macdsignal")
    # macdhist = pd.Series(macdhist, name="macdhist")
    # cci
    real = pd.Series(talib.CCI(high, low, close, timeperiod=14), name="cci")
    # adx
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14),name="adx")
    # 트렌드
    # inclination = calculate_trends(candles, 0)
    # 연결
    data = pd.concat([candles, openr, highr, lowr, closer, sma5, sma10, sma20, sma40, sma60, sma90, sma120, ema5, ema20, ema60, ema120, rsi, volume_sma, upperband, lowerband, atr, real, adx],
                     axis=1)

    data.fillna(0, inplace=True)
    # inf값 없애기
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    file_path = os.path.join(BASE_DIR, file_path)
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret


# 시스템 시간 동기화
def set_system_time(serv_time):
    gmtime = time.gmtime(int((serv_time["serverTime"])/1000))
    win32api.SetSystemTime(gmtime[0],
                           gmtime[1],
                           0,
                           gmtime[2],
                           gmtime[3],
                           gmtime[4],
                           gmtime[5],
                           0)

# 클라이언트 로그인
def create_client(api_key_file_path):
    client = None
    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        set_system_time(server_time)
    except BinanceAPIException as e:
        logger.error(f"Error creating client: {e}")
        exit()
    logger.info("Log in OK")
    return client

# Future USDT 잔고 출력
def get_usdt_balance(client):
    usdt_balance = None
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            usdt_balance = float(asset['balance'])
            break
    if usdt_balance is not None:
        logger.info(f"Future Total USDT: {usdt_balance}")
    else:
        logger.error("Can't Find Future USDT.")
    return usdt_balance

# client 가용 usdt long
def check_long_usdt(client):
    balance = client.get_asset_balance(asset='USDT')
    amt = balance['free']
    return float(amt)

# client 가용 usdt short
def check_short_usdt(client):
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            amt = asset['availableBalance']
    return float(amt)

# Spot 지갑에서 Futures 지갑으로 USDT 옮기기 또는 그 반대
# type 1이 spot -> future 2가 future->sopt
def transfer_spot_futures(client, amount, type):
    if amount <= 0:
        return
    try:
        # Spot에서 USD-M Futures 지갑으로의 전송
        result = client.futures_account_transfer(asset='USDT', amount=amount, type=type)
        logger.info("Transfer successful:", result)
    except Exception as e:
        logger.error("An error occurred:", str(e))

# client usdt 가능한 전부 옮기기
# type 1이 spot -> future 2가 future->sopt
def transfer_usdt(client, type):
    long_usdt = check_long_usdt(client)
    short_usdt = check_short_usdt(client)

    print(long_usdt, short_usdt)

    # spot ->
    if type==1:
        if long_usdt > 0 :
            transfer_spot_futures(client, long_usdt, type)
        return check_short_usdt(client)
    else:
        if short_usdt > 0:
            transfer_spot_futures(client, short_usdt, type)
        return check_long_usdt(client)
    
    
# ticker 정보찾기
def find_long_precision(client, symbol):

    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']

    for s in symbols:
        if s['symbol'] == symbol:
            filters = {f['filterType']: f for f in s['filters']}
            step_size = Decimal(filters['LOT_SIZE']['stepSize']).normalize()
            tick_size = Decimal(filters['PRICE_FILTER']['tickSize']).normalize()
            min_notional = Decimal(filters['NOTIONAL']['minNotional']).normalize()
            quantity_precision = -step_size.as_tuple().exponent
            price_precision = -tick_size.as_tuple().exponent
            return price_precision, quantity_precision, min_notional
        
def find_short_precision(client, symbol):
    info = client.futures_exchange_info()
    symbols = info['symbols']

    for s in symbols:
        if s['symbol'] == symbol:
            price_precision = s['pricePrecision']
            quantity_precision = s['quantityPrecision']
            for f in s['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = f['notional']
            return price_precision, quantity_precision, float(min_notional)

# 롱 전용 오더
def create_order_market(client, ticker, quantity):
    try:
        order = client.order_market_buy(
            symbol=ticker,
            quantity=quantity,
        )
        return order
    except BinanceAPIException as e:
        logger.error(f"Error creating market buy order for {ticker}: {e}")

def close_order_market(client, ticker, quantity):
    try:
        order = client.order_market_sell(
            symbol=ticker,
            quantity=quantity,
        )
        return order
    except BinanceAPIException as e:
        logger.error(f"Error creating market sell order for {ticker}: {e}")

def cancel_order(client, ticker, order):
    if order == None:
        return
    try:
        result = client.cancel_order(
            symbol=ticker,
            orderId=order['orderId']
        )
        return result
    except BinanceAPIException as e:
        logger.error(f"Error cancelling order for {ticker}: {e}")

def cancel_all_open_order(client, ticker):
    try:
        orders = client.get_open_orders(symbol=ticker)
        for order in orders:
            result = client.cancel_order(
                symbol=ticker,
                orderId = order['orderId']
            )
        return
    except BinanceAPIException as e:
        logger.error(f"Error cancelling all open orders for {ticker}: {e}")

# 숏 전용 오더
def future_create_order_market(client, symbol, quantity):
    try:
        f_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return f_order
    except BinanceAPIException as e:
        logger.error(f"Error creating futures market sell order for {symbol}: {e}")

def future_close_order_market(client, symbol, quantity):
    try:
        f_order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return f_order
    except BinanceAPIException as e:
        logger.error(f"Error creating futures market buy order for {symbol}: {e}")

def future_cancel_order(client, symbol, order):
    if order == None:
        return
    try:
        f_order = client.futures_cancel_order(
            symbol=symbol,
            orderID = order['orderId']
        )
        return f_order
    except BinanceAPIException as e:
        logger.error(f"Error cancelling futures order for {symbol}: {e}")

def future_cancel_all_open_order(client, symbol):
    try:
        f_order = client.futures_cancel_all_open_orders(
            symbol=symbol
        )
        return f_order
    except BinanceAPIException as e:
        logger.error(f"Error cancelling all futures open orders for {symbol}: {e}")

# 수량 확인
def get_balance(client, ticker):
    try:
        return client.get_asset_balance(asset=ticker)
    except BinanceAPIException as e:
        logger.error(f"Error fetching balance for {ticker}: {e}")
        return None

def adjust_precision(precision, number):
    if precision == 0:
        decimal_format = "1"
    else:
        decimal_format = f'0.{"0" * (int(precision)-1)}1'
    decimal_value = Decimal(number).quantize(Decimal(decimal_format), rounding=ROUND_DOWN)
    return float(decimal_value)

# 각 트레이드 센터에서 Ticker에 대한 모든 것들을 관리한다.
class TradeCenter:
    def __init__(self, ticker, agent, lossp):
        self.tick = ticker
        self.ticker = f"{ticker}USDT"
        self.lossp = lossp

        self.stop_order = None
        #self.set_myqt()

        # ticker 정보 모으기 
        # price_precision: 가격 정확도  quantity_precision: 수량 정확도  min_notional: 최소주문금액(USDT)
        self.l_price_precision, self.l_quantity_precision, self.l_min_notional = find_long_precision(main_client, self.ticker)
        self.s_price_precision, self.s_quantity_precision, self.s_min_notional = find_short_precision(main_client, self.ticker)

        self.data_1w = None
        self.data_1d = None
        self.data_4h = None
        self.data_1h = None

        self.data_1w_obs = None
        self.data_1d_obs = None
        self.data_4h_obs = None
        self.data_1h_obs = None
        self.obs_len = None

        self.data_1w_mv = None
        self.data_1d_mv = None
        self.data_4h_mv = None
        self.data_1h_mv = None

        # Z-score Standardization
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.scaler_minmax.data_max_ = 100
        self.scaler_minmax.data_min_ = 0

        self.intervals = {
            '1w': {'interval': Client.KLINE_INTERVAL_1WEEK, 'data': 'data_1w'},
            '1d': {'interval': Client.KLINE_INTERVAL_1DAY, 'data': 'data_1d'},
            '4h': {'interval': Client.KLINE_INTERVAL_4HOUR, 'data': 'data_4h'},
            '1h': {'interval': Client.KLINE_INTERVAL_1HOUR, 'data': 'data_1h'}
        }

        # Ticker의 mean, variance를 불러온다.
        self.download_mv()

        # 데이터 다운로드 일련 속도

        self.download_datas(main_client)
        self.make_obs_datas()
        self.normalize_obs()
        self.get_obs_row()

        self.TradeAgent = A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=3, gamma=GAMMA, batch_size=BATCH_SIZE)
        self.LongAgent= A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=2, gamma=GAMMA, batch_size=BATCH_SIZE)
        self.ShortAgent = A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=2, gamma=GAMMA, batch_size=BATCH_SIZE)

        self.load_weights(agent)

    def print_ticker_info(self):
        logger.info(f"l_price_precision={self.l_price_precision} l_quantity_precision={self.l_quantity_precision} l_min_notional={self.l_min_notional}"+ 
              f" s_price_precision={self.s_price_precision} s_quantity_precision={self.s_quantity_precision} s_min_notional={self.s_min_notional}")

    def check_on_trading(self, client):

        curr_price = self.data_1h['close'].iloc[-1]
        # 롱 
        amt = self.check_long_amount(client)
        amt = adjust_precision(self.l_quantity_precision, amt)
        if amt * curr_price >= self.l_min_notional :
            return "Long"
        # 숏
        amt = self.check_short_amount(client)
        amt = adjust_precision(self.s_quantity_precision, amt)
        if amt * curr_price <= - self.s_min_notional:
            return "Short"
        return "Stay"

    def normalize_obs(self):
        z_cols = ['openp', 'highp', 'lowp', 'closep',
          'sma5p', 'sma20p',  'sma60p', 'sma120p', 'volp', 
          'upperbandp', 'lowerbandp', 'atr', 'cci', 'adx']
        min_max_cols = ['rsi']

        for key, info in self.intervals.items():
            data = getattr(self, info['data']+"_obs")
            data_mv = getattr(self, info['data']+"_mv")
            for col in z_cols:
                self.scaler_standard.mean_ = np.array([data_mv[col][0]])
                self.scaler_standard.scale_ = np.array([math.sqrt(data_mv[col][1])])
                data[col] = self.scaler_standard.transform(data[[col]])
            for col in min_max_cols:
                data[col] = self.scaler_minmax.fit_transform(data[[col]])
                
    def download_mv(self):
        start = time.time()
        path = os.path.join(BASE_DIR, "var_table")
        self.data_1w_mv = pd.read_csv(f"{path}/{self.ticker}_data_1w_var_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1d_mv = pd.read_csv(f"{path}/{self.ticker}_data_1d_var_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_4h_mv = pd.read_csv(f"{path}/{self.ticker}_data_4h_var_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1h_mv = pd.read_csv(f"{path}/{self.ticker}_data_1h_var_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)

    def load_weights(self, path):
        path = os.path.join(BASE_DIR, "save_weights\\a2c_17")
        self.TradeAgent.actor.model.load_weights(f"{path}/Trade_actor.weights.h5")
        self.TradeAgent.critic.model.load_weights(f"{path}/Trade_critic.weights.h5")
        self.LongAgent.actor.model.load_weights(f"{path}/Long_actor.weights.h5")
        self.LongAgent.critic.model.load_weights(f"{path}/Long_critic.weights.h5")
        self.ShortAgent.actor.model.load_weights(f"{path}/Short_actor.weights.h5")
        self.ShortAgent.critic.model.load_weights(f"{path}/Short_critic.weights.h5")
        logger.info(f"{self.tick} Weights Load Okay")
        return
        
    def download_datas(self, client):
        self.data_1w = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1WEEK)
        self.data_1d = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1DAY)
        self.data_4h = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_4HOUR)
        self.data_1h = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1HOUR)

    def update_datas(self, client): # download_datas와 걸리는 시간이 얼마 차이 안난다.
        for key, info in self.intervals.items():
            # 현재 인터벌에 대한 새 캔들 데이터를 가져옴
            line = get_klines(client=client, symbol=self.ticker, limit=1, interval=info['interval'])
            data = getattr(self, info['data'])
            # 새 행이 감지되면 기존 데이터 업데이트
            if line['time'].iloc[0] != data['time'].iloc[-1]:
                info['data'] = pd.concat([data.drop(0), line], ignore_index=True)
                # 인스턴스 변수 업데이트
                setattr(self, f"data_{key}", f"{info['data']}")
        self.make_obs_datas()        

    def make_obs_datas(self):
        drop_list = ['open', 'high', 'low', 'close', 'volume', 'sma10p', 'sma40p', 'sma90p', 'ema5p', 'ema20p','ema60p','ema120p']
        self.data_1w_obs = get_candle_subdatas(self.data_1w).drop(columns=drop_list)
        self.data_1d_obs = get_candle_subdatas(self.data_1d).drop(columns=drop_list)
        self.data_4h_obs = get_candle_subdatas(self.data_4h).drop(columns=drop_list)
        self.data_1h_obs = get_candle_subdatas(self.data_1h).drop(columns=drop_list)

    def get_obs_row(self):
        row_list = []

        for key, info in self.intervals.items():
            data_obs = getattr(self, info['data']+'_obs')
            row = data_obs.iloc[-2].drop('time')
            row_list.append(row)

        obs = np.concatenate(row_list)
        self.obs_len = len(obs)
        return obs

    def check_long_amount(self, client):
        balance = client.get_asset_balance(asset=self.tick)
        amt = float(balance['free']) + float(balance['locked'])
        return float(amt)
    
    # amt output < 0
    def check_short_amount(self, client):
        positions = client.futures_position_information()
        amt = 0
        for position in positions:
            if position['symbol'] == self.ticker:
                amt = float(position['positionAmt'])
        # return adjust_precision(self.smqyt, float(amt))
        return float(amt)


    def trade(self, binance_client, now):
        client = binance_client.client
        bot = binance_client.bot
        chat_id = binance_client.chat_id
        amt_info = binance_client.ticker_center

        self.download_datas(client)
        self.make_obs_datas()
        self.normalize_obs()
        obs = self.get_obs_row()

        curr_price = self.data_1h['close'].iloc[-1]
        tact = self.TradeAgent.get_action(obs)
        lact = self.LongAgent.get_action(obs)
        sact = self.ShortAgent.get_action(obs)

        logger.info(f"{now} {self.ticker}\nTrade_Act = {tact}\nLong_Act = {lact}\nShort_Act = {sact}")
    
        if adjust_precision(self.l_quantity_precision, self.check_long_amount(client)) * curr_price > self.l_min_notional: # if Long
            act = lact
            act = np.argmax(act)

            # limitAct
            if act == 1 and np.argmax(tact) == 0:
                act = 0
                logger.info(f"{now} {self.ticker} Long lack activated")

            if act == 0: # Hold
                # Long 유지 알림
                logger.info(f"{now} {self.ticker} Long Holding")
                bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 유지")
                return
            else: # Sell
                self.close_long(binance_client, now)
                return

        elif abs(adjust_precision(self.s_quantity_precision, self.check_short_amount(client))) * curr_price > self.s_min_notional: # if Short
            act = sact
            act = np.argmax(act)

            # limitAct
            if act == 1 and np.argmax(tact) == 1:
                act = 0
                logger.info(f"{now} {self.ticker} Short lack activated")

            if act == 0: # Hold
                # Short 유지 알림
                logger.info(f"{now} {self.ticker} Short Holding")
                bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 유지")
                return
            else: # Sell
                self.close_short(binance_client, now)
                return
            
        else: # No position
            print("no position")
            act = tact
            raw_act = act
            # bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : Raw_Act = {raw_act}")
            act = np.argmax(act)
            # lact = np.argmax(lact)
            # sact = np.argmax(sact)
            if raw_act[act] >= act_threshold:
                if act == 0: # Long
                    self.open_long(binance_client, amt_info, now)
                    return
                elif act == 1:
                    self.open_short(binance_client, amt_info, now)
                    return
                    
            # 미진입 알림
            logger.info(f"{now} {self.ticker} STAY")
            # bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 관망")
            return
        
    def open_long(self, binance_client, amt_info, now):

        client = binance_client.client
        bot = binance_client.bot
        chat_id = binance_client.chat_id
        centers = binance_client.TradeCenters

        # 가용 usdt 확인 - 근데 이거 1개 ticker 운용 기준이다..
        available_usdt = transfer_usdt(client, 2)
        long_budget, short_budget = printbudget(binance_client, False)
        target_usdt = (long_budget+short_budget) * amt_info.get_tick_amt(self.tick)

        if available_usdt < target_usdt:
            target_usdt = available_usdt

        # 현재 비트코인 시세 확인
        tick_price = self.data_1h['close'].iloc[-1]
        # 포지션 들어가기
        amt = target_usdt / tick_price
        # quantity precision 교정
        amt = adjust_precision(self.l_quantity_precision, amt)
        if float(available_usdt) >= self.l_min_notional and float(amt) > 0 :
            # 롱 진입
            create_order_market(client, self.ticker, amt)
            # 손절가 설정
            self.stop_order = self.create_stop_loss(client, tick_price)
            # Long 진입 알림
            logger.info(f"{now} {self.ticker} Long Entered {tick_price} {amt}")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 진입\n진입가 : {tick_price}\n진입량 : {amt}")
            return
        else:
            logger.info(f"{now} {self.ticker} Long Enter Failed - No Balance")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 진입 자금이 부족합니다.")
            return
        
    def open_short(self, binance_client, amt_info, now):

        client = binance_client.client
        bot = binance_client.bot
        chat_id = binance_client.chat_id
        centers = binance_client.TradeCenters

        # 가용 usdt 확인 - 근데 이거 1개 ticker 운용 기준이다..
        available_usdt = transfer_usdt(client, 1)
        long_budget, short_budget = printbudget(binance_client, False)
        target_usdt = (long_budget+short_budget) * amt_info.get_tick_amt(self.tick)

        if available_usdt < target_usdt:
            target_usdt = available_usdt

        # 현재 비트코인 시세 확인
        tick_price = self.data_1h['close'].iloc[-1]
        # 포지션 들어가기
        client.futures_change_leverage(symbol=self.ticker, leverage=1)
        amt = target_usdt / tick_price
        # quantitiy precision 교정
        print(amt, self.s_quantity_precision)
        amt = adjust_precision(self.s_quantity_precision, amt)
        
        if float(available_usdt) >= self.s_min_notional and float(amt) > 0: # smqyt 는 ticker마다 다른데, 일일이 찾아봐야 하는 것 같다.     
            # 숏 진입       
            future_create_order_market(client, self.ticker, amt)
            # 손절가 설정
            self.stop_order = self.future_create_stop_loss(client, self.ticker, tick_price)
            # Short 진입 알림
            logger.info(f"{now} {self.ticker} Short Entered {tick_price} {amt}")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 진입\n진입가 : {tick_price}\n진입량 : {amt}")
            return
        else:
            logger.error(f"{now} {self.ticker} Short Enter Failed - No Balance")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 진입 자금이 부족합니다.")
            return
        
    def close_long(self, binance_client, now):

        client = binance_client.client
        bot = binance_client.bot
        chat_id = binance_client.chat_id

        cancel_all_open_order(client, self.ticker) # TODO: 여기서 통신 오류로 오더가 취소 안되면 에러 뜬다.
        
        amt = self.check_long_amount(client)
        amt = adjust_precision(self.l_quantity_precision, amt)
        if amt <= 0 :
            return
        
        close_order_market(client, self.ticker, float(amt))
        # Long 거래종료 알림
        logger.info(f"{now} {self.ticker} Long closed {amt}")
        bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 종료\n종료량 : {amt}")
        return

    def close_short(self, binance_client, now):

        client = binance_client.client
        bot = binance_client.bot
        chat_id = binance_client.chat_id

        future_cancel_all_open_order(client, self.ticker)
        amt = self.check_short_amount(client)
        amt = adjust_precision(self.s_quantity_precision, amt)
        if amt >= 0 :
            return
        future_close_order_market(client, self.ticker, float(abs(amt)))
        # Short 거래종료 알림
        logger.info(f"{now} {self.ticker} Short closed {amt}")
        bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 종료\n종료량 : {amt}")
        return
    
    def create_stop_loss(self, client, price):

        # 따로 또 계산해줘야함
        amt = self.check_long_amount(client)
        amt = adjust_precision(self.l_quantity_precision, amt)

        if amt <= 0 :
            return

        decimal_format = f'0.{"0" * (int(self.l_price_precision)-1)}1'
        OVER = 0.9 # 트리거 값

        try:
            order = client.create_order(
                symbol=self.ticker,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT, # 왜 LIMIT밖에 안돼
                timeInForce=TIME_IN_FORCE_GTC, # 주문의 유효시간
                quantity=amt,
                stopPrice = Decimal(price * (1-self.lossp)).quantize(Decimal(decimal_format), rounding=ROUND_DOWN),
                price = Decimal(price * (1-self.lossp) * OVER).quantize(Decimal(decimal_format), rounding=ROUND_DOWN) # LIMIT 값을 stop price 아래로 걸어놓으면 시장가로 팔린다.
            )
            return order
        except BinanceAPIException as e:
            logger.error(f"Error creating stop loss order for {self.ticker}: {e}")
        
    def future_create_stop_loss(self, client, symbol, price):

        amt = self.check_short_amount(client)
        amt = adjust_precision(self.s_quantity_precision, amt)

        if amt >= 0 :
            return
        
        decimal_format = f'0.{"0" * (int(self.s_price_precision)-1)}1'

        try:
            f_order = client.futures_create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                quantity=abs(amt),
                stopPrice= Decimal(price*(1+self.lossp)).quantize(Decimal(decimal_format), rounding=ROUND_DOWN)
            )
            return f_order
        except BinanceAPIException as e:
            logger.error(f"Error creating future stop loss order for {self.ticker}: {e}")
    
    # 솔직히 매우 비효율적이다
class Ticker_Center:
    def __init__(self):
        self.ticker_amt = {}

    def set_tick_amt(self, ticker, amt):
        # TODO: ticker_amt 합이 1이 넘지 않는 로직을 추가해야 함.

        self.ticker_amt[ticker] = amt

    def get_tick_amt(self, ticker):
        return self.ticker_amt[ticker]

    def get_on_trading_ticks(self, client, centers):
        on_trade_long = []
        on_trade_short = []
        for center in centers:
            now = center.check_on_trading(client)
            if now == "Long":
                on_trade_long.append(center.tick)
            elif now == "Short":
                on_trade_short.append(center.tick)
        return on_trade_long, on_trade_short

    def cal_amt(self, ticker, client, centers, long):
        divider = 1
        on_trade_long, on_trade_short = self.get_on_trading_ticks(client, centers)
        if long==True:
            for t in on_trade_long:
                # 이미 진행 중이라면
                if ticker == t:
                    return 0
                divider = divider - self.ticker_amt[t]
        elif long==False:
            for t in on_trade_short:
                # 이미 진행 중이라면
                if ticker == t:
                    return 0
                divider = divider - self.ticker_amt[t]

        amt = float(self.ticker_amt[ticker] / divider)
        if amt >= 1:
            return 0
        else:
            return amt

def process_client(client):
    client.bot.send_message(chat_id=client.chat_id, text="###[딥러닝 자동매매 봇]###\nCOPYRIGHT by 표영종")
    client.bot.send_message(chat_id=client.chat_id, text="###[딥러닝 자동매매 봇]###\n데이터 로드 OK, 매매 시작")

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
    server_time = client.client.get_server_time()
    set_system_time(server_time)
    centers = client.TradeCenters
    # for center in centers:
    #     print(center.tick)
    #     center.trade(client, formatted_now)

def printbudget(binance_client, isprint):

    binance_client.bot.send_message(chat_id=binance_client.chat_id, text=f"###[딥러닝 자동매매 봇] - 현재 자금")

    client = binance_client.client
    # 계좌 정보 가져오기
    account_info = client.get_account()
    budget = 0

    # 잔고가 0보다 큰 자산만 필터링하여 출력
    for balance in account_info['balances']:
        asset = balance['asset']
        
        total_balance = float(balance['free']) + float(balance['locked'])
        if total_balance > 0:
            # BTC/USDT 현재 가격 가져오기
            if asset=="USDT":
                budget += total_balance
                continue
            ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
            price = float(ticker['price'])
            budget += price * total_balance
            if round(price * total_balance, 1) > 0 and isprint:
                binance_client.bot.send_message(chat_id=binance_client.chat_id, text=f"{ticker['symbol']} 롱 진행중... 현재가치:{round(price * total_balance, 1)}")

    future_budget = get_usdt_balance(client)

    # 현재 포지션 정보 얻기 (선물 계정)
    positions = client.futures_account()['positions']

    # 각 포지션의 미실현 PnL 출력
    for position in positions:
        symbol = position['symbol']
        unrealized_pnl = float(position['unrealizedProfit'])

        if unrealized_pnl != 0:
            if isprint:
                binance_client.bot.send_message(chat_id=binance_client.chat_id, text=f"###{symbol} 숏 진행중... 미실현수익: {round(unrealized_pnl), 1}")
            future_budget += float(unrealized_pnl)

    if isprint:
        long_usdt = check_long_usdt(client)
        short_usdt = check_short_usdt(client)
        binance_client.bot.send_message(chat_id=binance_client.chat_id, text=f"현재 총 잔고:{round(budget+future_budget, 1)}\n롱 진입량:{round(budget-long_usdt, 1)}\n숏 진입량:{round(future_budget-short_usdt, 1)}")
    logger.info(f"{binance_client.name} bugdet={budget+future_budget}")
    return float(budget), float(future_budget)


def periodic_trade(client):
    while True:
        now = datetime.now()
        if now.minute == 0 and now.second < 1:
            continue
        else:
            if now.minute == 0 and now.second < 5:
                formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
                server_time = client.client.get_server_time()
                set_system_time(server_time)
                centers = client.TradeCenters
                random.shuffle(centers)
                for center in centers:
                    center.trade(client, formatted_now)
                printbudget(client, True)
                time_to_sleep = 60 - now.second  
            else:
                time_to_sleep = 60 - now.second
            logger.info(f"{client.name} Sleeping...") 
            time.sleep(time_to_sleep)


# 실매매 클라이언트
clients = []

# 정보 업뎃용
main_client = create_client("users/pyjong1999.txt")

def main():
    # 초기 메시지 전송 및 초기 거래 실행
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_client, clients)

    # 각 클라이언트에 대해 주기적인 거래 실행
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(periodic_trade, clients)



if __name__=='__main__':

    # 로그인 Main Client
    clientA = Binance_Client(name="pyjong1999", api="users/ujungsun.txt",token="6332731064:AAEOlgnRBgM8RZxW9CnkUPJHvEo54SZoEH8", chat_id=1735838793)
    # clientB = Binance_Client(name="minsu", api="api2.txt",token="5954588749:AAHh9g-cKaNL90orGxqurLOcn-GE9bwu5mU", chat_id=6886418534)
    for t in tick:
        clientA.create_tradecenter(t, "a2c_17", 0.1, 0.09)
        # clientB.create_tradecenter(t, "a2c_17", 0.1, 0.12)
    clients.append(clientA)
    # clients.append(clientB)

    for client in clients:
        printbudget(client, True)

    main()
