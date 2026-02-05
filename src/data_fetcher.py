"""
股票資料獲取模組
使用 yfinance 獲取股價資料，並從 TWSE 獲取籌碼資料
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    RAW_DATA_DIR, CACHE_EXPIRY_HOURS, 
    DEFAULT_HISTORY_DAYS, STOCK_SUFFIX, TAIEX_SYMBOL
)


class StockDataFetcher:
    """股票資料獲取器"""
    
    def __init__(self):
        self.cache_dir = RAW_DATA_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str) -> str:
        """取得快取檔案路徑"""
        return os.path.join(self.cache_dir, f"{symbol.replace('.', '_')}.csv")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """檢查快取是否有效"""
        if not os.path.exists(cache_path):
            return False
        
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        expiry_time = datetime.now() - timedelta(hours=CACHE_EXPIRY_HOURS)
        return modified_time > expiry_time
    
    def get_stock_data(self, symbol: str, days: int = DEFAULT_HISTORY_DAYS, 
                       force_refresh: bool = False) -> pd.DataFrame:
        """
        獲取股票歷史資料
        
        Args:
            symbol: 股票代號 (如 '2330' 或 '2330.TW')
            days: 歷史天數
            force_refresh: 是否強制重新獲取
            
        Returns:
            包含 OHLCV 資料的 DataFrame
        """
        # 處理股票代號格式
        if not symbol.endswith('.TW') and not symbol.startswith('^'):
            symbol = f"{symbol}{STOCK_SUFFIX}"
        
        cache_path = self._get_cache_path(symbol)
        
        # 檢查快取
        if not force_refresh and self._is_cache_valid(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        
        # 從 yfinance 獲取資料
        try:
            # end_date 設為明天，因為 yfinance 的 end 是不包含的 (exclusive)
            end_date = datetime.now() + timedelta(days=1)
            start_date = end_date - timedelta(days=days+1)
            
            # 使用預設行為，讓 yfinance 自動處理 Session (新版本會使用 curl_cffi)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"無法獲取 {symbol} 的資料")
            
            # 重命名欄位為小寫
            df.columns = [col.lower() for col in df.columns]
            
            # 移除時區資訊（避免 pandas 相容性問題）
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 儲存快取
            df.to_csv(cache_path)
            
            return df
            
        except Exception as e:
            print(f"獲取 {symbol} 資料時發生錯誤: {e}")
            raise
    
    def get_taiex_data(self, days: int = DEFAULT_HISTORY_DAYS) -> pd.DataFrame:
        """獲取台灣加權指數資料"""
        return self.get_stock_data(TAIEX_SYMBOL, days)
    
    def get_institutional_data(self, symbol: str, date: str = None) -> dict:
        """
        獲取三大法人買賣超資料
        
        Args:
            symbol: 股票代號 (純數字，如 '2330')
            date: 日期 (YYYYMMDD 格式)，預設為今天
            
        Returns:
            包含三大法人買賣超的字典
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        # 移除 .TW 後綴
        symbol = symbol.replace('.TW', '').replace('.tw', '')
        
        try:
            # 使用 TWSE API
            url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date}&selectType=ALL&response=json"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' not in data:
                return self._get_default_institutional_data()
            
            # 搜尋目標股票
            for row in data['data']:
                if row[0].strip() == symbol:
                    return {
                        'foreign_buy': self._parse_number(row[4]),  # 外資買賣超
                        'trust_buy': self._parse_number(row[10]),   # 投信買賣超
                        'dealer_buy': self._parse_number(row[11]),  # 自營商買賣超
                    }
            
            return self._get_default_institutional_data()
            
        except Exception as e:
            print(f"獲取籌碼資料時發生錯誤: {e}")
            return self._get_default_institutional_data()
    
    def _parse_number(self, value: str) -> float:
        """解析數字字串"""
        try:
            return float(value.replace(',', '').replace(' ', ''))
        except:
            return 0.0
    
    def _get_default_institutional_data(self) -> dict:
        """返回預設的籌碼資料"""
        return {
            'foreign_buy': 0.0,
            'trust_buy': 0.0,
            'dealer_buy': 0.0,
        }
    
    def get_historical_institutional_data(self, symbol: str, 
                                          days: int = 30) -> pd.DataFrame:
        """
        獲取歷史籌碼資料
        
        由於 TWSE API 限制，這裡使用模擬資料作為示範
        實際應用中可以建立資料庫累積每日資料
        """
        # 簡化版本：返回隨機模擬資料
        # 實際應用中應該從資料庫讀取累積的歷史資料
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        np.random.seed(42)
        data = {
            'date': dates,
            'foreign_buy': np.random.randn(days) * 10000,
            'trust_buy': np.random.randn(days) * 5000,
            'dealer_buy': np.random.randn(days) * 3000,
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df


if __name__ == "__main__":
    # 測試
    fetcher = StockDataFetcher()
    
    # 測試獲取股價
    print("測試獲取台積電股價...")
    df = fetcher.get_stock_data('2330', days=10)
    print(df.tail())
    
    # 測試獲取大盤
    print("\n測試獲取加權指數...")
    taiex = fetcher.get_taiex_data(days=10)
    print(taiex.tail())
