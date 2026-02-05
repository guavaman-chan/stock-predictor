# 股票隔日漲跌預測系統

基於機器學習的台股隔日漲跌預測模型，使用 XGBoost 演算法，整合 35 個技術面、籌碼面、市場環境特徵進行分析。

## 功能特色

- 📊 **35 個預測特徵**：技術指標、動能、波動率、籌碼面、市場環境
- 🤖 **XGBoost 模型**：處理非線性關係，自動特徵選擇
- 🌐 **Web 介面**：Streamlit 視覺化操作界面
- 📈 **即時預測**：輸入股票代號即可獲得預測結果

## 安裝

```bash
cd stock_predictor
pip install -r requirements.txt
```

## 使用方式

### 啟動 Web 介面

```bash
streamlit run app.py
```

### 命令列使用

```python
from src.model import StockPredictor

# 初始化預測器
predictor = StockPredictor()

# 訓練模型（使用 2 年歷史資料）
predictor.train('2330', days=730)

# 預測隔日漲跌
result = predictor.predict('2330')
print(f"預測: {result['prediction']}")
print(f"上漲機率: {result['up_probability']:.2%}")
```

## 系統架構

```
stock_predictor/
├── config/config.py      # 設定檔
├── src/
│   ├── data_fetcher.py   # 資料獲取
│   ├── feature_engineer.py # 特徵工程
│   └── model.py          # ML 模型
├── app.py                # Web 介面
└── requirements.txt
```

## 特徵列表

| 類別 | 特徵數 | 說明 |
|------|--------|------|
| 技術面 | 15 | 均線、RSI、MACD、KD、布林通道 |
| 動能 | 6 | ROC、動量、威廉指標 |
| 波動率 | 3 | ATR、歷史波動率 |
| 籌碼面 | 5 | 三大法人買賣超 |
| 基本面 | 3 | PE、PB、殖利率 |
| 市場 | 3 | 大盤、類股、星期效應 |

## 免責聲明

⚠️ 本系統僅供研究參考，不構成投資建議。股市有風險，投資需謹慎。
