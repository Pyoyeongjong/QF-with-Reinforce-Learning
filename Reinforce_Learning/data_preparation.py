# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
# Time 동기화
import time
# 보조지표 계산/출력 라이브러리
import talib
# Numpy / pandas
import numpy as np
import pandas as pd
# CSV파일
import os
import csv
# minmax scaler
from sklearn.preprocessing import MinMaxScaler
import parameter

BASE_DIR = parameter.BASE_DIR
KLINE_INTERVAL_MAPPING = {
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    # "5m": Client.KLINE_INTERVAL_5MINUTE,
    # "1m": Client.KLINE_INTERVAL_1MINUTE,
}

test_tickers = parameter.test_tickers
tickers = parameter.tickers
# Scaler
scaler = MinMaxScaler()
# API 파일 경로
api_key_file_path = os.path.join(BASE_DIR, "api.txt")
# 클라이언트 변수
_client = None

### Initiation
# row 생략 없이 출력
pd.set_option('display.max_rows', 20)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
# 가져올 분봉 데이터의 개수 (최대 500개까지 가능)
limit = 500

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret

def create_client():
    global _client
    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        _client = Client(binance_access_key, binance_secret_key)
    except BinanceAPIException as e:
        print(e)
        exit()
    return

# USDT 잔고 출력
def get_usdt_balance(client, isprint):
    usdt_balance = None
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            usdt_balance = float(asset['balance'])
            break
    if usdt_balance is not None:
        if isprint:
            print(f"USDT 잔고: {usdt_balance}")
    else:
        print("USDT 잔고를 찾을 수 없습니다.")
    return usdt_balance

# 현재 데이터 가져오는 함수
def get_klines(client, symbol, limit, interval, start_time=None, end_time=None):
    col_name = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote', 'trade_num', 'taker_buy_base',
                'taker_buy_quote', 'ignored']
    if start_time and end_time:
        start_timestamp = int(start_time.timestamp() * 1000)  # 밀리초 단위로 변환
        end_timestamp = int(end_time.timestamp() * 1000)  # 밀리초 단위로 변환
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit,
                                startTime=start_timestamp, endTime=end_timestamp)
    else:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit) # 최근 500개
    return pd.DataFrame(klines, columns=col_name)

# 과거 데이터를 가져와 csv에 저장하는 함수
# Time 작성법 "1 Jan, 2017", "30 DEC, 2023"
def get_candle_data_to_csv(ticker, scale, start_time, end_time):
    # csv 파일 생성
    filename = f"{ticker}_{scale}.csv"
    data_dir = os.path.join(BASE_DIR, "candles_data")
    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])
        print("Open Ok")
        klines = _client.get_historical_klines(ticker, KLINE_INTERVAL_MAPPING.get(scale), start_time, end_time)
        print("Get Candles OK")

        # 첫 6개만 slicing
        writer.writerows([k[:6] for k in klines])
    print("Data fetching and saving completed.")


def get_all_candle_datas_to_csv(ticker, start_time, end_time):
    for key in KLINE_INTERVAL_MAPPING.keys():
        if key == '1m' or key == '5m': # 1m, 5m은 너무 커서 PASS
            continue
        print(f"getting_candle_data {ticker}_{key}.csv")
        get_candle_data_to_csv(ticker, key, start_time, end_time)


def get_candle_subdatas(candles):
    # 비어있다면?
    if candles.empty:
        return

    # 문자열 -> 숫자 변환 && pd Series
    close = candles['close'].apply(pd.to_numeric).to_numpy() # 종가 값 활용
    high = candles['high'].apply(pd.to_numeric).to_numpy()
    low = candles['low'].apply(pd.to_numeric).to_numpy()
    volume = candles['volume'].apply(pd.to_numeric).to_numpy()
    front_close = candles['close'].shift(1).replace(0, np.nan)

    highr = ((candles['high'] - front_close ) / front_close * 100).fillna(0)
    highr.name = 'highp'
    lowr = ((candles['low'] - front_close ) / front_close * 100).fillna(0)
    lowr.name = 'lowp'
    closer = (( candles['close'] - front_close ) / front_close * 100).fillna(0)
    closer.name = 'closep'
    openr = ((candles['open'] - front_close ) / front_close * 100).fillna(0)
    openr.name = 'openp'

    def calculate_percentage(series, base):
        return ((series - base) / base * 100).fillna(0)

    sma_periods = [5, 10, 20, 40, 60, 90, 120]
    sma_cols = {}
    for period in sma_periods:
        sma = talib.SMA(close, timeperiod=period)
        sma_cols[f'sma{period}p'] = calculate_percentage(pd.Series(sma), candles['close'])
        sma_cols[f'sma{period}p'].name = f"sma{period}p"

    ema_periods = [5, 20, 60, 120]
    ema_cols = {}
    for period in ema_periods:
        ema = talib.EMA(close, timeperiod=period)
        ema_cols[f'ema{period}p'] = calculate_percentage(pd.Series(ema), candles['close'])
        ema_cols[f'ema{period}p'].name = f"ema{period}p"

    rsi = pd.Series(talib.RSI(close, timeperiod=14), name="rsi").fillna(0)
    volume_sma = (pd.Series(talib.SMA(volume, timeperiod=20), name="vol_sma") / candles['volume']).replace([np.inf, -np.inf], 0).fillna(0)
    volume_sma.name = "volp"

    ### 한국 시간으로 맞춰주기 + DateTime으로 변환
    # korea_tz = pytz.timezone('Asia/Seoul')
    # datetime = pd.to_datetime(candles['time'], unit='ms')
    # candles['time'] = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 볼린저 밴드
    upperband, _, lowerband = talib.BBANDS(candles['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    upperband = calculate_percentage(pd.Series(upperband), candles['close'])
    upperband.name = "upperbandp"
    lowerband = calculate_percentage(pd.Series(lowerband), candles['close'])
    lowerband.name = "lowerbandp"
    # atr
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), name="atr")
    # cci
    real = pd.Series(talib.CCI(high, low, close, timeperiod=14), name="cci")
    # adx
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14),name="adx")
    # 연결
    data = pd.concat([candles, openr, highr, lowr, closer] + list(sma_cols.values()) +
                     list(ema_cols.values()) + [rsi, volume_sma, upperband, lowerband, atr, real, adx],
                     axis=1)

    data.fillna(0, inplace=True)
    return data

def get_subdatas(tickers):
    for ticker in tickers:
        for key in KLINE_INTERVAL_MAPPING.keys():
            print(f"making {ticker}_{key}..")
            file_path = os.path.join(BASE_DIR, f"candles_data/{ticker}_{key}.csv")
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            df_sub = get_candle_subdatas(df)
            if df_sub is not None:
                new_file_path = os.path.join(BASE_DIR, f"candles_sub_data/{ticker}_{key}_sub.csv")
                df_sub.to_csv(new_file_path)

def get_subdata_one():
    
    df = pd.read_csv("candle_datas/BTCUSDT_1w.csv")
    df_sub = get_candle_subdatas(df)
    if df_sub is not None :
        df_sub.to_csv(f"candle_datas/BTCUSDT_1w_sub.csv")
    

if __name__ == '__main__':
    print("hello")
    create_client()
    get_usdt_balance(_client, True)

    # for ticker in tickers:
    #     get_all_candle_datas_to_csv(ticker, "1 Jan, 2017", "31 Jul, 2024")
    get_subdatas(tickers)


    
