"""
特徵工程模組
計算技術指標、處理籌碼資料、生成訓練特徵
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ALL_FEATURES


class FeatureEngineer:
    """特徵工程處理器"""
    
    def __init__(self):
        pass
    
    def calculate_all_features(self, df: pd.DataFrame, 
                               taiex_df: Optional[pd.DataFrame] = None,
                               institutional_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        計算所有特徵
        
        Args:
            df: 股票 OHLCV 資料
            taiex_df: 大盤資料 (可選)
            institutional_df: 籌碼資料 (可選)
            
        Returns:
            包含所有特徵的 DataFrame
        """
        df = df.copy()
        
        # 確保欄位名稱為小寫
        df.columns = [col.lower() for col in df.columns]
        
        # 計算各類特徵
        df = self._calculate_return_features(df)
        df = self._calculate_ma_features(df)
        df = self._calculate_momentum_features(df)
        df = self._calculate_volatility_features(df)
        df = self._calculate_volume_features(df)
        
        # 計算籌碼特徵
        if institutional_df is not None:
            df = self._merge_institutional_features(df, institutional_df)
        else:
            df = self._add_default_institutional_features(df)
        
        # 計算市場特徵
        if taiex_df is not None:
            df = self._calculate_market_features(df, taiex_df)
        else:
            df = self._add_default_market_features(df)
        
        # 計算時間特徵
        df = self._calculate_time_features(df)
        
        # 添加基本面特徵 (使用預設值)
        df = self._add_fundamental_features(df)
        
        return df
    
    def _calculate_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算報酬率特徵"""
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        return df
    
    def _calculate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算均線相關特徵"""
        # 計算均線
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()
        
        # 計算價格相對均線位置 (比率)
        df['ma_5_ratio'] = (df['close'] - df['ma_5']) / df['ma_5']
        df['ma_20_ratio'] = (df['close'] - df['ma_20']) / df['ma_20']
        df['ma_60_ratio'] = (df['close'] - df['ma_60']) / df['ma_60']
        
        # 均線交叉訊號 (短期均線相對長期均線位置)
        df['ma_cross'] = (df['ma_5'] > df['ma_20']).astype(int)
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算動能指標"""
        # RSI
        df['rsi_14'] = momentum.RSIIndicator(df['close'], n=14).rsi()
        
        # MACD
        macd_indicator = trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # KD 指標
        stoch = momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], n=14, d_n=3
        )
        df['k_value'] = stoch.stoch()
        df['d_value'] = stoch.stoch_signal()
        
        # 威廉指標
        df['williams_r'] = momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close'], lbp=14
        ).wr()
        
        # ROC
        df['roc_10'] = momentum.ROCIndicator(df['close'], n=10).roc()
        
        # 動能
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算波動率特徵"""
        # 歷史波動率
        df['volatility_20d'] = df['return_1d'].rolling(window=20).std() * np.sqrt(252)
        
        # 波動率變化
        vol_5d = df['return_1d'].rolling(window=5).std()
        vol_20d = df['return_1d'].rolling(window=20).std()
        df['volatility_ratio'] = vol_5d / vol_20d
        
        # 當日振幅
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # 布林通道
        bollinger = volatility.BollingerBands(df['close'], n=20, ndev=2)
        df['bollinger_position'] = (
            (df['close'] - bollinger.bollinger_lband()) / 
            (bollinger.bollinger_hband() - bollinger.bollinger_lband())
        )
        
        # ATR
        df['atr'] = volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], n=14
        ).average_true_range()
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算成交量特徵"""
        # 成交量比率
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 成交量趨勢
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_trend'] = (df['volume_ma_5'] > df['volume_ma_20']).astype(int)
        
        # OBV 趨勢
        obv = volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_ma']).astype(int)
        
        return df
    
    def _merge_institutional_features(self, df: pd.DataFrame, 
                                       institutional_df: pd.DataFrame) -> pd.DataFrame:
        """合併籌碼資料"""
        # 重新索引對齊
        institutional_df = institutional_df.reindex(df.index, method='ffill')
        
        df['foreign_buy'] = institutional_df['foreign_buy']
        df['trust_buy'] = institutional_df['trust_buy']
        df['dealer_buy'] = institutional_df['dealer_buy']
        
        # 計算衍生特徵
        df['foreign_buy_5d'] = df['foreign_buy'].rolling(window=5).sum()
        df['institution_total'] = df['foreign_buy'] + df['trust_buy'] + df['dealer_buy']
        
        return df
    
    def _add_default_institutional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加預設籌碼特徵（當無實際資料時）"""
        df['foreign_buy'] = 0.0
        df['trust_buy'] = 0.0
        df['dealer_buy'] = 0.0
        df['foreign_buy_5d'] = 0.0
        df['institution_total'] = 0.0
        return df
    
    def _calculate_market_features(self, df: pd.DataFrame, 
                                    taiex_df: pd.DataFrame) -> pd.DataFrame:
        """計算市場相關特徵"""
        # 對齊大盤資料
        taiex_df.columns = [col.lower() for col in taiex_df.columns]
        taiex_aligned = taiex_df.reindex(df.index, method='ffill')
        
        # 大盤報酬率
        df['taiex_return'] = taiex_aligned['close'].pct_change(1)
        
        # 類股報酬率 (簡化版本，使用大盤報酬率)
        df['sector_return'] = df['taiex_return']
        
        return df
    
    def _add_default_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加預設市場特徵"""
        df['taiex_return'] = 0.0
        df['sector_return'] = 0.0
        return df
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算時間特徵"""
        df['day_of_week'] = df.index.dayofweek
        return df
    
    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基本面特徵（使用預設值）"""
        # 實際應用中這些應該從財報資料獲取
        df['pe_ratio'] = 15.0  # 預設本益比
        df['pb_ratio'] = 2.0   # 預設股價淨值比
        df['dividend_yield'] = 0.03  # 預設殖利率 3%
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        創建預測標籤
        
        標籤定義：隔日收盤價上漲 = 1，下跌 = 0
        """
        df = df.copy()
        df['next_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = (df['next_return'] > 0).astype(int)
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                         feature_cols: list = None) -> tuple:
        """
        準備訓練特徵
        
        Args:
            df: 包含所有特徵和標籤的 DataFrame
            feature_cols: 要使用的特徵欄位列表
            
        Returns:
            (X, y) 特徵矩陣和標籤向量
        """
        if feature_cols is None:
            feature_cols = ALL_FEATURES
        
        # 過濾存在的欄位
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # 移除含有 NaN 的列
        df_clean = df.dropna(subset=available_cols + ['label'])
        
        X = df_clean[available_cols].values
        y = df_clean['label'].values
        
        return X, y, available_cols


if __name__ == "__main__":
    # 測試
    from data_fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher()
    engineer = FeatureEngineer()
    
    # 獲取資料
    df = fetcher.get_stock_data('2330', days=365)
    taiex = fetcher.get_taiex_data(days=365)
    
    # 計算特徵
    df = engineer.calculate_all_features(df, taiex_df=taiex)
    df = engineer.create_labels(df)
    
    print("特徵數量:", len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]))
    print("\n特徵列表:")
    print(df.columns.tolist())
    
    # 準備訓練資料
    X, y, cols = engineer.prepare_features(df)
    print(f"\n訓練樣本數: {len(y)}")
    print(f"特徵數: {len(cols)}")
    print(f"正樣本比例: {y.mean():.2%}")
