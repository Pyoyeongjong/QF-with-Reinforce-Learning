import os
# Data 시작연도
START_YEAR = 2018
# BASE_DIR
BASE_DIR = os.path.expanduser("~/QF-with-Reinforce-Learning/Reinforce_Learning")
# 보상함수
def cal_reward(percent):
    if percent >= 0:
        return percent
    else:
        return - (1 / (1 + percent) - 1)
# 파라미터
HOLD_REWARD = 0.01
LONG_TRANS_FEE = 0.1 * 0.01
SHORT_TRANS_FEE = 0.04 * 0.01
# Test 날짜
TEST_TIMESTAMP = 1704067200 * 1000
TRAIN_TIMESTAMP = 1527814800 * 1000 
# 관망 Decay
STAY_REWARD = 0.000
STAY_DECAY = 1
INIT_DECAY = 1
# Tickers
test_tickers= ["ETCUSDT", "XLMUSDT"]
tickers = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT","DOTUSDT",
            "LINKUSDT", "TRXUSDT", "MATICUSDT","BCHUSDT", "ICPUSDT",
            "NEARUSDT", "UNIUSDT", "LTCUSDT", "STXUSDT",
            "FILUSDT", "THETAUSDT", "NEOUSDT", "FLOWUSDT", "XTZUSDT"]
# 스탑로스 함수
def get_long_sl(position):
    return position * 0.9
def get_short_sl(position):
    return position * 1.1
# Train Batch Size
BATCH_SIZE_T = 32 # Trade Agent
BATCH_SIZE_LS = 128 # Lond / Short Agent
# Agent Default Size
GAMMA = 0.9999
BATCH_SIZE = 32
# 만약 테스트할 때 일정 확률 이상일 때만 실행시키면?? (수수료 Issue)
THRESHOLD = 0.5 # 0이면 THR 없음
# LA, SA가 TA 눈치보기
TASIGNALON = 1
#사용 중인 TimeFrames
TIMEFRAMES = ['1w', '1d', '4h', '1h', '15m']