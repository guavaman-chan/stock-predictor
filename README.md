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

## 可進一步優化方向

以下是下一階段值得優先投入的改進項目：

1. **時間序列驗證與回測流程**  
   目前可補強 walk-forward validation 與滾動視窗回測，避免資料洩漏，並更接近真實交易情境。

2. **模型校準與不平衡處理**  
   建議加入機率校準（Platt / Isotonic）與類別不平衡策略（class weight、focal loss 或重抽樣），提升「機率可解釋性」與穩定度。

3. **特徵重要性與漂移監控**  
   可加入 SHAP/Permutation importance 與資料分布監控，追蹤哪些特徵在不同市場階段失效，並建立預警機制。

4. **風險導向指標**  
   除了方向準確率，可加入策略層 KPI（最大回撤、Sharpe、勝率、期望值），避免模型僅在分類分數漂亮但實際報酬不佳。

5. **資料層韌性與快取**  
   建議在資料抓取流程增加快取、重試、資料品質檢查（缺值與異常值），讓訓練與預測更穩定可重現。

6. **MLOps 與部署治理**  
   可補上模型版本管理、實驗追蹤、定期重訓排程與線上監控，縮短從研究到上線的落差。

若要快速看到效益，建議先做「時間序列驗證 + 風險指標」這兩項，通常最能直接改善實際可用性。

## 免責聲明

⚠️ 本系統僅供研究參考，不構成投資建議。股市有風險，投資需謹慎。
