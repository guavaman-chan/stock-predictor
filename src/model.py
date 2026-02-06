"""
機器學習模型模組
使用 XGBoost 進行股票漲跌預測
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_DIR, MODEL_PARAMS, ALL_FEATURES
from src.data_fetcher import StockDataFetcher
from src.feature_engineer import FeatureEngineer
from src.cloud_storage import get_cloud_storage


class StockPredictor:
    """股票漲跌預測模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.fetcher = StockDataFetcher()
        self.engineer = FeatureEngineer()
        
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def _get_model_path(self, symbol: str) -> str:
        """取得模型檔案路徑"""
        symbol_clean = symbol.replace('.', '_').replace('^', '')
        return os.path.join(MODEL_DIR, f"model_{symbol_clean}.joblib")
    
    def _get_scaler_path(self, symbol: str) -> str:
        """取得 scaler 檔案路徑"""
        symbol_clean = symbol.replace('.', '_').replace('^', '')
        return os.path.join(MODEL_DIR, f"scaler_{symbol_clean}.joblib")
    
    def prepare_data(self, symbol: str, days: int = 730) -> tuple:
        """
        準備訓練資料
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            
        Returns:
            (X, y, feature_cols) 特徵矩陣、標籤、特徵欄位名
        """
        # 獲取資料
        df = self.fetcher.get_stock_data(symbol, days=days)
        taiex_df = self.fetcher.get_taiex_data(days=days)
        
        # 計算特徵
        df = self.engineer.calculate_all_features(df, taiex_df=taiex_df)
        df = self.engineer.create_labels(df)
        
        # 準備特徵
        X, y, feature_cols = self.engineer.prepare_features(df)
        
        return X, y, feature_cols
    
    def train(self, symbol: str, days: int = 730, 
              test_size: float = 0.2, save_model: bool = True) -> dict:
        """
        訓練模型
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            test_size: 測試集比例
            save_model: 是否儲存模型
            
        Returns:
            訓練結果統計
        """
        print(f"開始訓練 {symbol} 預測模型...")
        
        # 準備資料
        X, y, feature_cols = self.prepare_data(symbol, days)
        self.feature_cols = feature_cols
        
        print(f"總樣本數: {len(y)}")
        print(f"特徵數: {len(feature_cols)}")
        
        # 時間序列分割 (不打亂順序)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"訓練集: {len(y_train)}, 測試集: {len(y_test)}")
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 訓練模型
        self.model = XGBClassifier(**MODEL_PARAMS)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # 評估
        y_pred = self.model.predict(X_test_scaled)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
        }
        
        print("\n===== 模型評估結果 =====")
        print(f"準確率 (Accuracy): {results['accuracy']:.4f}")
        print(f"精確率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1 分數: {results['f1']:.4f}")
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n===== Top 10 重要特徵 =====")
        print(feature_importance.head(10).to_string(index=False))
        
        results['feature_importance'] = feature_importance
        
        # 儲存模型
        if save_model:
            model_path = self._get_model_path(symbol)
            scaler_path = self._get_scaler_path(symbol)
            
            joblib.dump({
                'model': self.model,
                'feature_cols': self.feature_cols
            }, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # 同步上傳到雲端
            cloud = get_cloud_storage()
            model_filename = os.path.basename(model_path)
            scaler_filename = os.path.basename(scaler_path)
            cloud.upload_file('models', model_path, model_filename)
            cloud.upload_file('models', scaler_path, scaler_filename)
            
            print(f"\n模型已儲存至: {model_path}")
        
        return results
    
    def load_model(self, symbol: str) -> bool:
        """
        載入已訓練的模型
        
        Args:
            symbol: 股票代號
            
        Returns:
            是否載入成功
        """
        model_path = self._get_model_path(symbol)
        scaler_path = self._get_scaler_path(symbol)
        
        # 如果本地沒有模型，嘗試從雲端下載
        if not os.path.exists(model_path):
            cloud = get_cloud_storage()
            model_filename = os.path.basename(model_path)
            scaler_filename = os.path.basename(scaler_path)
            cloud.download_file('models', model_filename, model_path)
            cloud.download_file('models', scaler_filename, scaler_path)
        
        if not os.path.exists(model_path):
            return False
        
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.feature_cols = data['feature_cols']
            self.scaler = joblib.load(scaler_path)
            return True
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return False
    
    def predict(self, symbol: str, retrain_if_missing: bool = True) -> dict:
        """
        預測隔日漲跌
        
        Args:
            symbol: 股票代號
            retrain_if_missing: 如果模型不存在是否重新訓練
            
        Returns:
            預測結果
        """
        # 嘗試載入模型
        if self.model is None:
            if not self.load_model(symbol):
                if retrain_if_missing:
                    print("模型不存在，開始訓練...")
                    self.train(symbol)
                else:
                    raise ValueError(f"找不到 {symbol} 的預測模型")
        
        # 獲取最新資料
        df = self.fetcher.get_stock_data(symbol, days=100)
        taiex_df = self.fetcher.get_taiex_data(days=100)
        
        # 計算特徵
        df = self.engineer.calculate_all_features(df, taiex_df=taiex_df)
        
        # 取最後一筆資料進行預測
        available_cols = [col for col in self.feature_cols if col in df.columns]
        X_latest = df[available_cols].iloc[-1:].values
        
        # 處理 NaN
        X_latest = np.nan_to_num(X_latest, nan=0.0)
        
        # 標準化
        X_scaled = self.scaler.transform(X_latest)
        
        # 預測
        pred = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        
        result = {
            'symbol': symbol,
            'prediction': '上漲' if pred == 1 else '下跌',
            'prediction_code': int(pred),
            'up_probability': float(pred_proba[1]),
            'down_probability': float(pred_proba[0]),
            'confidence': float(max(pred_proba)),
            'latest_close': float(df['close'].iloc[-1]),
            'latest_date': str(df.index[-1].date()),
        }
        
        return result
    
    # ============ 風險指標計算方法 ============
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """計算最大回撤"""
        if len(returns) == 0:
            return 0.0
        
        equity_curve = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / running_max - 1
        return float(np.min(drawdowns))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, annualization: int = 252) -> float:
        """計算年化 Sharpe Ratio（無風險利率視為 0）"""
        if len(returns) == 0:
            return 0.0
        
        std = np.std(returns)
        if std == 0:
            return 0.0
        
        return float(np.sqrt(annualization) * np.mean(returns) / std)
    
    def _build_risk_metrics(self, strategy_returns: np.ndarray) -> dict:
        """彙整策略風險與報酬指標"""
        if len(strategy_returns) == 0:
            return {
                'cumulative_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_daily_return': 0.0,
            }
        
        return {
            'cumulative_return': float(np.prod(1 + strategy_returns) - 1),
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(strategy_returns),
            'win_rate': float(np.mean(strategy_returns > 0)),
            'avg_daily_return': float(np.mean(strategy_returns)),
        }
    
    # ============ 回測方法 ============
    
    def backtest(self, symbol: str, days: int = 365,
                 test_ratio: float = 0.3) -> dict:
        """
        回測模型表現（含風險指標）
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            test_ratio: 測試集比例（保留參數以相容舊版介面）
            
        Returns:
            回測結果，包含準確率與風險指標
        """
        # 保留 test_ratio 參數以避免破壞既有呼叫介面
        _ = test_ratio
        
        # 準備包含 next_return 的資料，便於計算策略績效
        df = self.fetcher.get_stock_data(symbol, days=days)
        taiex_df = self.fetcher.get_taiex_data(days=days)
        df = self.engineer.calculate_all_features(df, taiex_df=taiex_df)
        df = self.engineer.create_labels(df)
        
        available_cols = [col for col in ALL_FEATURES if col in df.columns]
        df_clean = df.dropna(subset=available_cols + ['label', 'next_return'])
        
        X = df_clean[available_cols].values
        y = df_clean['label'].values
        next_returns = df_clean['next_return'].values
        
        # 時間序列交叉驗證
        tscv = TimeSeriesSplit(n_splits=5)
        
        accuracies = []
        fold_results = []
        all_strategy_returns = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_next_returns = next_returns[test_idx]
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 訓練
            model = XGBClassifier(**MODEL_PARAMS)
            model.fit(X_train_scaled, y_train, verbose=False)
            
            # 評估
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            
            # 策略報酬：預測上漲做多，預測下跌做空
            strategy_returns = np.where(y_pred == 1, test_next_returns, -test_next_returns)
            all_strategy_returns.extend(strategy_returns.tolist())
            
            fold_metrics = self._build_risk_metrics(strategy_returns)
            fold_metrics['fold'] = fold
            fold_metrics['accuracy'] = float(acc)
            fold_results.append(fold_metrics)
        
        overall_risk_metrics = self._build_risk_metrics(np.array(all_strategy_returns))
        
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'fold_accuracies': [float(x) for x in accuracies],
            'fold_results': fold_results,
            'overall_risk_metrics': overall_risk_metrics,
        }
    
    def backtest_with_risk(self, symbol: str, days: int = 365) -> dict:
        """
        整合風險指標的完整回測（使用 risk_metrics 模組）
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            
        Returns:
            包含風險指標的回測結果
        """
        from src.risk_metrics import Backtester
        
        print(f"執行 {symbol} 風險回測...")
        
        # 準備資料
        df = self.fetcher.get_stock_data(symbol, days=days)
        taiex_df = self.fetcher.get_taiex_data(days=days)
        
        df = self.engineer.calculate_all_features(df, taiex_df=taiex_df)
        df = self.engineer.create_labels(df)
        
        # 載入或訓練模型
        if not self.load_model(symbol):
            self.train(symbol, days=days)
        
        # 取得特徵
        X, y, _ = self.engineer.prepare_features(df)
        
        # 標準化並預測
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # 對齊資料
        pred_series = pd.Series(predictions, index=df.index[-len(predictions):])
        price_series = df['close'].loc[pred_series.index]
        
        # 執行回測
        backtester = Backtester()
        results = backtester.run_backtest(price_series, pred_series)
        
        # 加入回測報告
        results['report'] = backtester.generate_report(results)
        
        print(results['report'])
        
        return results
    
    def walk_forward_train(self, symbol: str, days: int = 730,
                           train_ratio: float = 0.7,
                           validation_ratio: float = 0.15,
                           n_iterations: int = 3) -> dict:
        """
        滾動式訓練與自動修正
        
        將資料分成多個區段，每輪用前面資料訓練、後面資料驗證，
        再用驗證結果作為回饋來優化下一輪訓練
        
        Args:
            symbol: 股票代號
            days: 歷史天數
            train_ratio: 訓練集比例
            validation_ratio: 驗證集比例
            n_iterations: 迭代次數
            
        Returns:
            訓練結果與各輪表現
        """
        print(f"開始 {symbol} 滾動式訓練 ({n_iterations} 輪)...")
        
        # 準備完整資料
        X, y, feature_cols = self.prepare_data(symbol, days)
        self.feature_cols = feature_cols
        
        total_samples = len(y)
        iteration_results = []
        best_model = None
        best_scaler = None
        best_accuracy = 0
        
        # 累積的錯誤樣本，用於加強學習
        error_samples_X = []
        error_samples_y = []
        
        for i in range(n_iterations):
            print(f"\n=== 第 {i+1}/{n_iterations} 輪訓練 ===")
            
            # 計算每輪使用的資料範圍（滾動窗口）
            start_idx = int(i * total_samples * 0.1)
            end_idx = min(start_idx + int(total_samples * 0.9), total_samples)
            
            X_iter = X[start_idx:end_idx]
            y_iter = y[start_idx:end_idx]
            
            # 分割訓練、驗證、測試集
            n_samples = len(y_iter)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + validation_ratio))
            
            X_train = X_iter[:train_end]
            y_train = y_iter[:train_end]
            X_val = X_iter[train_end:val_end]
            y_val = y_iter[train_end:val_end]
            X_test = X_iter[val_end:]
            y_test = y_iter[val_end:]
            
            # 加入之前錯誤的樣本（增強學習）
            if len(error_samples_X) > 0:
                X_train = np.vstack([X_train] + error_samples_X)
                y_train = np.concatenate([y_train] + error_samples_y)
                print(f"  加入 {sum(len(e) for e in error_samples_X)} 個錯誤樣本進行強化學習")
            
            print(f"  訓練: {len(y_train)}, 驗證: {len(y_val)}, 測試: {len(y_test)}")
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # 訓練模型
            model = XGBClassifier(**MODEL_PARAMS)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # 驗證集評估
            y_val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            # 找出驗證集中預測錯誤的樣本
            val_errors_mask = y_val_pred != y_val
            if np.any(val_errors_mask):
                error_X = X_val[val_errors_mask]
                error_y = y_val[val_errors_mask]
                max_errors = min(len(error_y), 50)
                error_samples_X.append(error_X[:max_errors])
                error_samples_y.append(error_y[:max_errors])
            
            # 測試集評估
            y_test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            print(f"  驗證準確率: {val_accuracy:.4f}")
            print(f"  測試準確率: {test_accuracy:.4f}, F1: {test_f1:.4f}")
            
            iter_result = {
                'iteration': i + 1,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'train_samples': len(y_train),
                'error_samples_added': len(error_samples_X[-1]) if error_samples_X else 0
            }
            iteration_results.append(iter_result)
            
            # 保留最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model = model
                best_scaler = scaler
        
        # 使用最佳模型
        self.model = best_model
        self.scaler = best_scaler
        
        # 儲存最佳模型
        model_path = self._get_model_path(symbol)
        scaler_path = self._get_scaler_path(symbol)
        
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols
        }, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # 計算改進幅度
        first_acc = iteration_results[0]['test_accuracy']
        last_acc = iteration_results[-1]['test_accuracy']
        improvement = last_acc - first_acc
        
        print(f"\n=== 滾動式訓練完成 ===")
        print(f"最佳準確率: {best_accuracy:.4f}")
        print(f"準確率變化: {first_acc:.4f} → {last_acc:.4f} ({improvement:+.4f})")
        
        # 計算特徵重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'iterations': iteration_results,
            'best_accuracy': best_accuracy,
            'improvement': improvement,
            'feature_importance': feature_importance,
            'train_samples': len(y),
            'test_samples': iteration_results[-1]['train_samples']
        }


if __name__ == "__main__":
    predictor = StockPredictor()
    
    # 訓練模型
    results = predictor.train('2330', days=730)
    
    # 預測
    print("\n===== 預測結果 =====")
    prediction = predictor.predict('2330')
    print(f"股票: {prediction['symbol']}")
    print(f"最新收盤價: {prediction['latest_close']:.2f}")
    print(f"預測: {prediction['prediction']}")
    print(f"上漲機率: {prediction['up_probability']:.2%}")
    print(f"信心度: {prediction['confidence']:.2%}")
