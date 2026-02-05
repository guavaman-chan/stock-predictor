"""
模型回饋與增量學習模組
允許用戶輸入實際結果來修正模型
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR


class FeedbackManager:
    """
    回饋管理器
    記錄預測結果與實際結果，用於模型修正
    """
    
    def __init__(self):
        self.feedback_dir = os.path.join(DATA_DIR, 'feedback')
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def _get_feedback_path(self, symbol: str) -> str:
        """取得回饋資料檔案路徑"""
        symbol_clean = symbol.replace('.', '_').replace('^', '')
        return os.path.join(self.feedback_dir, f"feedback_{symbol_clean}.csv")
    
    def _get_prediction_log_path(self, symbol: str) -> str:
        """取得預測記錄檔案路徑"""
        symbol_clean = symbol.replace('.', '_').replace('^', '')
        return os.path.join(self.feedback_dir, f"predictions_{symbol_clean}.json")
    
    def save_prediction(self, symbol: str, prediction: dict, features: dict = None):
        """
        儲存預測記錄
        
        Args:
            symbol: 股票代號
            prediction: 預測結果
            features: 預測時使用的特徵值
        """
        log_path = self._get_prediction_log_path(symbol)
        
        # 讀取現有記錄
        predictions = []
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        
        # 新增記錄
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'prediction_date': prediction.get('latest_date'),
            'predicted_direction': prediction.get('prediction_code'),
            'up_probability': prediction.get('up_probability'),
            'confidence': prediction.get('confidence'),
            'latest_close': prediction.get('latest_close'),
            'actual_direction': None,  # 待用戶填入
            'actual_close': None,      # 待用戶填入
            'features': features,
        }
        
        predictions.append(record)
        
        # 儲存
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        return record
    
    def get_pending_feedbacks(self, symbol: str) -> list:
        """
        取得待回饋的預測記錄
        
        Args:
            symbol: 股票代號
            
        Returns:
            待回饋的預測列表
        """
        log_path = self._get_prediction_log_path(symbol)
        
        if not os.path.exists(log_path):
            return []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 篩選未填入實際結果的記錄
        pending = [p for p in predictions if p.get('actual_direction') is None]
        
        return pending
    
    def submit_feedback(self, symbol: str, prediction_date: str, 
                        actual_direction: int, actual_close: float = None) -> bool:
        """
        提交實際結果回饋
        
        Args:
            symbol: 股票代號
            prediction_date: 預測日期
            actual_direction: 實際漲跌 (1=上漲, 0=下跌)
            actual_close: 實際收盤價
            
        Returns:
            是否成功
        """
        log_path = self._get_prediction_log_path(symbol)
        
        if not os.path.exists(log_path):
            return False
        
        with open(log_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 找到對應的預測記錄
        updated = False
        for pred in predictions:
            if pred.get('prediction_date') == prediction_date:
                pred['actual_direction'] = actual_direction
                pred['actual_close'] = actual_close
                pred['feedback_time'] = datetime.now().isoformat()
                pred['is_correct'] = (pred['predicted_direction'] == actual_direction)
                updated = True
                break
        
        if updated:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            
            # 同時更新回饋資料集
            self._update_feedback_dataset(symbol, predictions)
        
        return updated
    
    def _update_feedback_dataset(self, symbol: str, predictions: list):
        """
        更新回饋資料集（用於重新訓練）
        """
        feedback_path = self._get_feedback_path(symbol)
        
        # 篩選已有實際結果的記錄
        feedbacks = [p for p in predictions if p.get('actual_direction') is not None]
        
        if not feedbacks:
            return
        
        # 轉換為 DataFrame
        records = []
        for fb in feedbacks:
            if fb.get('features'):
                record = fb['features'].copy()
                record['label'] = fb['actual_direction']
                record['prediction_date'] = fb['prediction_date']
                record['is_correct'] = fb.get('is_correct', False)
                records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(feedback_path, index=False)
    
    def get_feedback_stats(self, symbol: str) -> dict:
        """
        取得回饋統計
        
        Args:
            symbol: 股票代號
            
        Returns:
            統計資訊
        """
        log_path = self._get_prediction_log_path(symbol)
        
        if not os.path.exists(log_path):
            return {'total': 0, 'with_feedback': 0, 'correct': 0, 'accuracy': 0.0}
        
        with open(log_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        total = len(predictions)
        with_feedback = [p for p in predictions if p.get('actual_direction') is not None]
        correct = [p for p in with_feedback if p.get('is_correct', False)]
        
        return {
            'total': total,
            'with_feedback': len(with_feedback),
            'correct': len(correct),
            'accuracy': len(correct) / len(with_feedback) if with_feedback else 0.0,
            'pending': total - len(with_feedback),
        }
    
    def get_feedback_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        取得可用於重新訓練的回饋資料
        
        Args:
            symbol: 股票代號
            
        Returns:
            回饋資料 DataFrame
        """
        feedback_path = self._get_feedback_path(symbol)
        
        if not os.path.exists(feedback_path):
            return None
        
        return pd.read_csv(feedback_path)
    
    def get_prediction_history(self, symbol: str, limit: int = 30) -> list:
        """
        取得預測歷史記錄
        
        Args:
            symbol: 股票代號
            limit: 最多返回筆數
            
        Returns:
            預測歷史列表
        """
        log_path = self._get_prediction_log_path(symbol)
        
        if not os.path.exists(log_path):
            return []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 返回最近的記錄
        return predictions[-limit:]


class IncrementalLearner:
    """
    增量學習器
    使用回饋資料來微調模型
    """
    
    def __init__(self, predictor):
        """
        Args:
            predictor: StockPredictor 實例
        """
        self.predictor = predictor
        self.feedback_manager = FeedbackManager()
    
    def retrain_with_feedback(self, symbol: str, 
                               min_feedback_samples: int = 10) -> dict:
        """
        使用回饋資料重新訓練模型
        
        Args:
            symbol: 股票代號
            min_feedback_samples: 最少需要的回饋樣本數
            
        Returns:
            訓練結果
        """
        # 取得回饋資料
        feedback_df = self.feedback_manager.get_feedback_data(symbol)
        
        if feedback_df is None or len(feedback_df) < min_feedback_samples:
            return {
                'success': False,
                'message': f'回饋樣本不足，需要至少 {min_feedback_samples} 筆，目前有 {len(feedback_df) if feedback_df is not None else 0} 筆'
            }
        
        # 準備原始訓練資料
        X_original, y_original, feature_cols = self.predictor.prepare_data(symbol)
        
        # 準備回饋資料
        available_cols = [col for col in feature_cols if col in feedback_df.columns]
        X_feedback = feedback_df[available_cols].values
        y_feedback = feedback_df['label'].values
        
        # 合併資料 (給予回饋資料更高權重)
        # 將回饋資料重複 3 次，增加權重
        X_combined = np.vstack([X_original, X_feedback, X_feedback, X_feedback])
        y_combined = np.concatenate([y_original, y_feedback, y_feedback, y_feedback])
        
        # 重新訓練
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score
        from config.config import MODEL_PARAMS
        
        # 時間序列分割
        split_idx = int(len(X_combined) * 0.8)
        X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
        y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 訓練
        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # 評估
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 更新模型
        self.predictor.model = model
        self.predictor.scaler = scaler
        self.predictor.feature_cols = feature_cols
        
        # 儲存
        import joblib
        model_path = self.predictor._get_model_path(symbol)
        scaler_path = self.predictor._get_scaler_path(symbol)
        
        joblib.dump({
            'model': model,
            'feature_cols': feature_cols
        }, model_path)
        joblib.dump(scaler, scaler_path)
        
        return {
            'success': True,
            'message': '模型已使用回饋資料重新訓練',
            'accuracy': accuracy,
            'feedback_samples': len(feedback_df),
            'total_samples': len(y_combined),
        }


if __name__ == "__main__":
    # 測試
    fm = FeedbackManager()
    
    # 模擬預測記錄
    test_prediction = {
        'symbol': '2330',
        'prediction': '上漲',
        'prediction_code': 1,
        'up_probability': 0.65,
        'confidence': 0.65,
        'latest_close': 580.0,
        'latest_date': '2024-01-15',
    }
    
    # 儲存預測
    fm.save_prediction('2330', test_prediction, {'return_1d': 0.01, 'rsi_14': 55})
    
    # 查看待回饋記錄
    pending = fm.get_pending_feedbacks('2330')
    print(f"待回饋記錄: {len(pending)} 筆")
    
    # 提交回饋
    fm.submit_feedback('2330', '2024-01-15', actual_direction=1, actual_close=585.0)
    
    # 查看統計
    stats = fm.get_feedback_stats('2330')
    print(f"預測準確率: {stats['accuracy']:.2%}")
