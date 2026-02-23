"""
AI 深度分析模組
整合 Google Gemini API 自動產生針對股票預測的文字解讀報告
"""

import os
import streamlit as st
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai package not installed, AI analyzer disabled.")

class AIAnalyzer:
    """股票預測 AI 分析器"""
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        初始化分析器
        
        Args:
            model_name: 要使用的 Gemini 模型名稱（預設：gemini-1.5-flash）
        """
        self.enabled = False
        self.model_name = model_name
        self.client = None
        
        # 覆寫 model_name (如果 secrets 中有設定)
        try:
            if hasattr(st, 'secrets') and 'gemini' in st.secrets:
                if 'model' in st.secrets['gemini']:
                    self.model_name = st.secrets['gemini']['model']
        except:
            pass
            
        self._init_client()
        
    def _init_client(self):
        """初始化 Gemini API 客戶端"""
        if not GENAI_AVAILABLE:
            return
            
        api_key = None
        
        # 嘗試從 Streamlit secrets 獲取 API 金鑰
        try:
            if hasattr(st, 'secrets') and 'gemini' in st.secrets:
                api_key = st.secrets['gemini'].get('api_key')
        except:
            pass
            
        # 備用：從環境變數讀取
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
            
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.enabled = True
            except Exception as e:
                print(f"⚠️ Gemini API 初始化失敗: {e}")
                
    def generate_analysis(self, symbol: str, prediction_data: Dict[str, Any], features: Dict[str, float]) -> Optional[str]:
        """
        產生股票分析報告
        
        Args:
            symbol: 股票代號
            prediction_data: 機器學習模型的預測結果
            features: 最新一天的特徵值字典
            
        Returns:
            AI 分析報告文字 (Markdown 格式) 或是 None
        """
        if not self.enabled or not self.client:
            return None
            
        # 將技術指標轉換成容易閱讀的格式
        metrics_text = []
        
        # 常見技術指標映射
        key_metrics = {
            'close': '最新收盤價',
            'volume': '成交量',
            'return_1d': '單日報酬率',
            'return_5d': '5日報酬率',
            'rsi_14': 'RSI (14日)',
            'macd': 'MACD 值',
            'macdsignal': 'MACD 訊號線',
            'macdhist': 'MACD 柱狀圖',
            'bb_high_ind': '布林通道上軌突破',
            'bb_low_ind': '布林通道下軌突破',
            'kd_k': 'KD指標 K值',
            'kd_d': 'KD指標 D值',
            'sma_5': '5日均線',
            'sma_20': '20日均線'
        }
        
        for k, v in features.items():
            name = key_metrics.get(k, k)
            # 簡化數值顯示
            if isinstance(v, float):
                metrics_text.append(f"- {name}: {v:.4f}")
            else:
                metrics_text.append(f"- {name}: {v}")
                
        metrics_str = "\n".join(metrics_text[:20]) # 取前 20 個重要指標避免 context 過長
        
        # 組裝 Prompt
        prompt = f"""
你是一位專業的股票市場分析師。請根據以下我們系統的機器學習模型預測及技術指標，為這檔股票寫一份簡短、專業且易懂的分析報告給散戶投資人看。

股票代號/名稱：{symbol}
日期：{prediction_data.get('latest_date')}

【機器學習模型預測】
- 明日預測方向：{prediction_data.get('prediction')}
- 系統信心度：{prediction_data.get('confidence', 0):.1%}
- 模型預估上漲機率：{prediction_data.get('up_probability', 0):.1%}
- 模型預估下跌機率：{prediction_data.get('down_probability', 0):.1%}

【目前的客觀技術指標】
{metrics_str}

請依照以下格式回覆（請使用繁體中文）：
### 📊 盤勢總結
（用 2-3 句話總結目前的技術面狀態與模型預測趨勢）

### 🔍 關鍵信號解讀
（挑選 2-3 個最值得注意的技術指標或指標背離現象，例如 RSI 是否超買/超賣，MACD 趨勢，MA 排列等進行解讀）

### 💡 投資建議與風險提示
（給予短線操作建議，並務必加上風險控管提示，如停損停利點位參考等）

請保持客觀，語氣專業但不要使用過於艱澀的術語。不要產生任何免責聲明，系統介面上已經有一份免責聲明了。
"""
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ 生成 AI 分析報告失敗: {e}")
            return f"⚠️ 無法產生 AI 分析報告。錯誤細節: {str(e)}"

# 提供一個全域實例獲取捷徑
_ai_analyzer = None

def get_ai_analyzer() -> AIAnalyzer:
    global _ai_analyzer
    if _ai_analyzer is None:
        _ai_analyzer = AIAnalyzer()
    return _ai_analyzer
