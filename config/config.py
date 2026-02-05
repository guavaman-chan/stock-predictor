# 股票隔日漲跌預測模型配置

import os

# 資料目錄設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 建立目錄
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 資料獲取設定
DEFAULT_HISTORY_DAYS = 365  # 預設抓取一年歷史資料
CACHE_EXPIRY_HOURS = 24     # 快取過期時間

# 模型設定
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# 特徵設定
TECHNICAL_FEATURES = [
    'return_1d', 'return_5d', 'return_20d',
    'ma_5_ratio', 'ma_20_ratio', 'ma_60_ratio', 'ma_cross',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'k_value', 'd_value', 'bollinger_position', 'atr'
]

VOLUME_FEATURES = [
    'volume_ratio', 'volume_ma_trend', 'obv_trend',
    'momentum_10', 'roc_10', 'williams_r'
]

VOLATILITY_FEATURES = [
    'volatility_20d', 'volatility_ratio', 'price_range'
]

INSTITUTIONAL_FEATURES = [
    'foreign_buy', 'foreign_buy_5d', 'trust_buy', 
    'dealer_buy', 'institution_total'
]

FUNDAMENTAL_FEATURES = [
    'pe_ratio', 'pb_ratio', 'dividend_yield'
]

MARKET_FEATURES = [
    'taiex_return', 'sector_return', 'day_of_week'
]

ALL_FEATURES = (
    TECHNICAL_FEATURES + 
    VOLUME_FEATURES + 
    VOLATILITY_FEATURES + 
    INSTITUTIONAL_FEATURES + 
    FUNDAMENTAL_FEATURES + 
    MARKET_FEATURES
)

# 台股代號對應
STOCK_SUFFIX = '.TW'
TAIEX_SYMBOL = '^TWII'  # 加權指數
