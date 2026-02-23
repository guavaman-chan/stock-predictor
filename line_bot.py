"""
LINE Bot 股票預測機器人
透過 LINE Messaging API 接收股票代號，回傳預測結果
"""

import os
import sys
import traceback

# 添加模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

app = Flask(__name__)

# ===== LINE API 設定 =====
# 優先從環境變數讀取（部署用），備用從 .env 讀取
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', '')
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', '')

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    print("⚠️ 請設定 LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 環境變數")

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# ===== 股票預測器（延遲載入）=====
_predictor = None

def get_predictor():
    """延遲載入 StockPredictor (避免啟動時就載入大量套件)"""
    global _predictor
    if _predictor is None:
        from src.model import StockPredictor
        _predictor = StockPredictor()
    return _predictor


# ===== 指令處理 =====

HELP_TEXT = """📈 股票預測機器人使用說明

🔹 輸入股票代號即可預測：
   例如：2330、2317、2454

🔹 支援指令：
   「幫助」或「help」→ 顯示本說明
   「熱門」→ 顯示熱門股票列表

💡 範例：直接輸入 2330
系統將自動訓練模型並回傳隔日漲跌預測！

⚠️ 本系統僅供研究參考，不構成投資建議。"""

POPULAR_STOCKS = """🔥 熱門股票列表

📌 半導體
   2330 台積電
   2454 聯發科
   3034 聯詠

📌 電子
   2317 鴻海
   2382 廣達
   3008 大立光

📌 金融
   2881 富邦金
   2882 國泰金
   2891 中信金

💡 直接輸入代號即可預測！"""


def format_prediction_result(prediction: dict, train_results: dict) -> str:
    """將預測結果格式化為 LINE 訊息"""
    symbol = prediction.get('symbol', '?')
    direction = prediction.get('prediction', '未知')
    up_prob = prediction.get('up_probability', 0)
    confidence = prediction.get('confidence', 0)
    latest_close = prediction.get('latest_close', 0)
    latest_date = prediction.get('latest_date', '?')

    # 方向 emoji
    if prediction.get('prediction_code') == 1:
        emoji = "📈"
        direction_text = "上漲"
    else:
        emoji = "📉"
        direction_text = "下跌"

    # 信心度等級
    if confidence >= 0.7:
        conf_bar = "🟢🟢🟢 高"
    elif confidence >= 0.55:
        conf_bar = "🟡🟡 中"
    else:
        conf_bar = "🔴 低"

    # 模型準確率
    accuracy = train_results.get('accuracy', 0)
    if 'best_accuracy' in train_results:
        accuracy = train_results['best_accuracy']

    msg = f"""{emoji} {symbol} 預測結果

📅 資料日期：{latest_date}
💰 最新收盤：{latest_close:.2f}

━━━━━━━━━━━━━━━
🎯 預測：明日{direction_text}
📊 上漲機率：{up_prob:.1%}
🔒 信心度：{confidence:.1%} {conf_bar}
🏆 模型準確率：{accuracy:.1%}
━━━━━━━━━━━━━━━

⚠️ 僅供研究參考，不構成投資建議"""

    return msg


def handle_stock_query(symbol: str) -> str:
    """處理股票查詢"""
    try:
        predictor = get_predictor()

        # 訓練並預測
        train_results = predictor.train(symbol, days=365)
        prediction = predictor.predict(symbol)

        if prediction is None:
            return f"❌ 無法取得 {symbol} 的預測結果，請確認股票代號是否正確。"

        return format_prediction_result(prediction, train_results)

    except Exception as e:
        error_msg = str(e)
        print(f"預測錯誤 ({symbol}): {traceback.format_exc()}")

        if "No data found" in error_msg or "No price data" in error_msg:
            return f"❌ 找不到 {symbol} 的股價資料\n\n請確認：\n• 代號是否正確（例如 2330）\n• 該股票是否仍在交易"
        elif "No timezone found" in error_msg:
            return f"❌ {symbol} 代號可能不正確，請重新輸入。"
        else:
            return f"❌ 預測過程發生錯誤：{error_msg}\n\n請稍後再試或更換股票代號。"


def process_message(text: str) -> str:
    """處理使用者訊息，回傳對應結果"""
    text = text.strip()

    # 指令判斷
    if text in ['幫助', 'help', '說明', '?', '？']:
        return HELP_TEXT
    elif text in ['熱門', '推薦', '清單']:
        return POPULAR_STOCKS

    # 股票代號判斷（純數字 4 位）
    if text.isdigit() and 4 <= len(text) <= 6:
        return handle_stock_query(text)

    # 帶有 .TW 後綴
    if text.upper().endswith('.TW') and text[:-3].isdigit():
        return handle_stock_query(text)

    # 無法辨識
    return f"🤔 無法辨識「{text}」\n\n請輸入台股代號（例如 2330）\n或輸入「幫助」查看使用說明"


# ===== Flask 路由 =====

@app.route("/")
def index():
    """健康檢查用"""
    return "📈 股票預測 LINE Bot 運行中！"


@app.route("/callback", methods=['POST'])
def callback():
    """LINE Webhook 回呼端點"""
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """處理收到的文字訊息"""
    user_message = event.message.text
    reply_text = process_message(user_message)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )


# ===== 啟動 =====

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 LINE Bot 啟動於 port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
