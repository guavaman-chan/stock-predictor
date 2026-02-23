"""
Streamlit Web 應用程式
股票隔日漲跌預測介面 - 含回饋修正功能
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import StockPredictor
from src.data_fetcher import StockDataFetcher
from src.feedback import FeedbackManager, IncrementalLearner

# 頁面設定
st.set_page_config(
    page_title="股票漲跌預測系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d4aa 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .up-prediction {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 2px solid #10b981;
    }
    
    .down-prediction {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(120deg, #7c3aed 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(124, 58, 237, 0.4);
    }
    
    .sidebar .stSelectbox {
        background: #1e293b;
        border-radius: 8px;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .feedback-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# 初始化
@st.cache_resource
def get_predictor():
    return StockPredictor()

@st.cache_resource
def get_fetcher():
    return StockDataFetcher()

@st.cache_resource
def get_feedback_manager():
    return FeedbackManager()


def render_prediction_page(symbol_input, history_days, train_mode="standard", n_iterations=3):
    """渲染預測頁面"""
    try:
        predictor = get_predictor()
        fetcher = get_fetcher()
        feedback_manager = get_feedback_manager()
        
        # 顯示進度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📊 正在獲取股票資料...")
        progress_bar.progress(20)
        
        # 獲取股價資料
        df = fetcher.get_stock_data(symbol_input, days=history_days)
        
        status_text.text("🔧 正在訓練預測模型...")
        progress_bar.progress(50)
        
        # 根據訓練模式選擇訓練方法
        if train_mode == "walk_forward":
            status_text.text(f"🔄 滾動式訓練中 (共 {n_iterations} 輪)...")
            train_results = predictor.walk_forward_train(
                symbol_input, 
                days=history_days, 
                n_iterations=n_iterations
            )
        else:
            train_results = predictor.train(symbol_input, days=history_days, save_model=True)
        
        status_text.text("🎯 正在生成預測...")
        progress_bar.progress(80)
        
        # 進行預測
        prediction = predictor.predict(symbol_input)
        
        # 儲存預測記錄（用於後續回饋）
        feedback_manager.save_prediction(symbol_input, prediction)
        
        progress_bar.progress(100)
        status_text.text("✅ 預測完成！")
        
        # 清除進度顯示
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # 顯示預測結果
        st.markdown("---")
        st.subheader(f"🎯 {symbol_input} 預測結果")
        
        # 預測卡片
        prediction_emoji = "📈" if prediction['prediction_code'] == 1 else "📉"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="預測方向",
                value=f"{prediction_emoji} {prediction['prediction']}",
            )
        
        with col2:
            st.metric(
                label="上漲機率",
                value=f"{prediction['up_probability']:.1%}",
            )
        
        with col3:
            st.metric(
                label="信心度",
                value=f"{prediction['confidence']:.1%}",
            )
        
        # 股價資訊
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 近期股價走勢")
            
            # K線圖
            fig = go.Figure()
            
            recent_df = df.tail(60)
            
            fig.add_trace(go.Candlestick(
                x=recent_df.index,
                open=recent_df['open'],
                high=recent_df['high'],
                low=recent_df['low'],
                close=recent_df['close'],
                name='股價',
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444'
            ))
            
            # 添加均線
            if len(recent_df) >= 5:
                ma5 = recent_df['close'].rolling(5).mean()
                fig.add_trace(go.Scatter(
                    x=recent_df.index, y=ma5,
                    mode='lines', name='MA5',
                    line=dict(color='#fbbf24', width=1)
                ))
            
            if len(recent_df) >= 20:
                ma20 = recent_df['close'].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=recent_df.index, y=ma20,
                    mode='lines', name='MA20',
                    line=dict(color='#3b82f6', width=1)
                ))
            
            fig.update_layout(
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 特徵重要性 Top 10")
            
            feature_importance = train_results['feature_importance'].head(10)
            
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
                yaxis=dict(autorange='reversed'),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 模型評估指標
        st.markdown("---")
        st.subheader("📋 模型評估指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # 相容滾動式訓練和一般訓練的回傳格式
        if 'best_accuracy' in train_results:
            # 滾動式訓練
            accuracy = train_results['best_accuracy']
            # 從最後一輪取得其他指標
            last_iter = train_results['iterations'][-1]
            precision = last_iter.get('test_precision', 0)
            recall = last_iter.get('test_recall', 0)
            f1 = last_iter.get('test_f1', 0)
        else:
            # 一般訓練
            accuracy = train_results.get('accuracy', 0)
            precision = train_results.get('precision', 0)
            recall = train_results.get('recall', 0)
            f1 = train_results.get('f1', 0)
        
        with col1:
            st.metric("準確率", f"{accuracy:.2%}")
        with col2:
            st.metric("精確率", f"{precision:.2%}")
        with col3:
            st.metric("召回率", f"{recall:.2%}")
        with col4:
            st.metric("F1 分數", f"{f1:.2%}")
        
        # 詳細資訊
        with st.expander("📄 查看詳細資訊"):
            st.json({
                "股票代號": prediction['symbol'],
                "最新收盤價": prediction['latest_close'],
                "資料日期": prediction['latest_date'],
                "訓練樣本數": train_results['train_samples'],
                "測試樣本數": train_results['test_samples'],
            })
    
    except Exception as e:
        st.error(f"❌ 預測過程發生錯誤: {str(e)}")
        st.exception(e)


def render_feedback_page(symbol_input):
    """渲染回饋頁面"""
    feedback_manager = get_feedback_manager()
    predictor = get_predictor()
    
    st.subheader("📝 結果回饋與模型修正")
    
    # 顯示回饋統計
    stats = feedback_manager.get_feedback_stats(symbol_input)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總預測次數", stats['total'])
    with col2:
        st.metric("已回饋次數", stats['with_feedback'])
    with col3:
        st.metric("預測正確次數", stats['correct'])
    with col4:
        st.metric("實際準確率", f"{stats['accuracy']:.1%}" if stats['with_feedback'] > 0 else "N/A")
    
    st.markdown("---")
    
    # 待回饋的預測
    pending = feedback_manager.get_pending_feedbacks(symbol_input)
    
    if pending:
        st.markdown("### 📋 待回饋的預測記錄")
        st.info(f"有 {len(pending)} 筆預測結果待您回饋實際漲跌情況")
        
        for i, pred in enumerate(pending[:5]):  # 最多顯示5筆
            with st.container():
                st.markdown(f"""
                <div class="feedback-card">
                    <strong>預測日期：</strong> {pred['prediction_date']}<br>
                    <strong>收盤價：</strong> {pred['latest_close']:.2f}<br>
                    <strong>預測方向：</strong> {'📈 上漲' if pred['predicted_direction'] == 1 else '📉 下跌'}<br>
                    <strong>預測機率：</strong> {pred['up_probability']:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("✅ 實際上漲", key=f"up_{i}_{pred['prediction_date']}"):
                        feedback_manager.submit_feedback(
                            symbol_input, 
                            pred['prediction_date'], 
                            actual_direction=1
                        )
                        st.success("已記錄：實際上漲")
                        st.rerun()
                
                with col2:
                    if st.button("❌ 實際下跌", key=f"down_{i}_{pred['prediction_date']}"):
                        feedback_manager.submit_feedback(
                            symbol_input, 
                            pred['prediction_date'], 
                            actual_direction=0
                        )
                        st.success("已記錄：實際下跌")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ 刪除此筆", key=f"del_{i}_{pred['prediction_date']}", type="secondary"):
                        feedback_manager.delete_prediction(symbol_input, pred['prediction_date'])
                        st.success("已刪除此筆預測記錄")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("目前沒有待回饋的預測記錄")
    
    # 模型重新訓練
    st.markdown("### 🔄 使用回饋資料重新訓練模型")
    
    feedback_data = feedback_manager.get_feedback_data(symbol_input)
    feedback_count = len(feedback_data) if feedback_data is not None else 0
    
    st.markdown(f"目前已累積 **{feedback_count}** 筆回饋資料")
    
    if feedback_count >= 10:
        st.markdown("""
        <div class="success-box">
        回饋資料已足夠！可以使用回饋資料來微調模型，提高預測準確度。
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 使用回饋資料重新訓練", type="primary"):
            with st.spinner("正在重新訓練模型..."):
                learner = IncrementalLearner(predictor)
                result = learner.retrain_with_feedback(symbol_input)
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.metric("新模型準確率", f"{result['accuracy']:.2%}")
                else:
                    st.warning(result['message'])
    else:
        st.markdown(f"""
        <div class="warning-box">
        需要至少 10 筆回饋資料才能重新訓練模型。目前還需要 {10 - feedback_count} 筆。
        </div>
        """, unsafe_allow_html=True)
    
    # 預測歷史記錄
    st.markdown("---")
    st.markdown("### 📊 預測歷史記錄")
    
    history = feedback_manager.get_prediction_history(symbol_input, limit=20)
    
    if history:
        # 清除全部按鈕
        col_title, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ 清除全部記錄", key="clear_all_predictions", type="secondary"):
                st.session_state['confirm_clear_all'] = True
        
        # 二次確認
        if st.session_state.get('confirm_clear_all', False):
            st.warning(f"⚠️ 確定要刪除 {symbol_input} 的所有 {len(history)} 筆預測記錄嗎？此操作無法復原！")
            col_yes, col_no, _ = st.columns([1, 1, 3])
            with col_yes:
                if st.button("✅ 確定刪除", key="confirm_yes"):
                    deleted = feedback_manager.delete_all_predictions(symbol_input)
                    st.session_state['confirm_clear_all'] = False
                    st.success(f"已刪除 {deleted} 筆記錄")
                    st.rerun()
            with col_no:
                if st.button("❌ 取消", key="confirm_no"):
                    st.session_state['confirm_clear_all'] = False
                    st.rerun()
        
        history_df = pd.DataFrame(history)
        
        # 確保必要欄位存在
        if 'is_correct' not in history_df.columns:
            history_df['is_correct'] = None
        
        # 選擇可用欄位
        available_cols = ['prediction_date', 'predicted_direction', 'up_probability', 'actual_direction', 'is_correct']
        existing_cols = [col for col in available_cols if col in history_df.columns]
        
        # 格式化顯示
        display_df = history_df[existing_cols].copy()
        
        # 重新命名欄位
        col_rename = {
            'prediction_date': '預測日期',
            'predicted_direction': '預測方向', 
            'up_probability': '上漲機率',
            'actual_direction': '實際結果',
            'is_correct': '預測正確'
        }
        display_df.columns = [col_rename.get(c, c) for c in display_df.columns]
        
        if '預測方向' in display_df.columns:
            display_df['預測方向'] = display_df['預測方向'].map({1: '📈 上漲', 0: '📉 下跌'})
        if '上漲機率' in display_df.columns:
            display_df['上漲機率'] = display_df['上漲機率'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        if '實際結果' in display_df.columns:
            display_df['實際結果'] = display_df['實際結果'].map({1: '📈 上漲', 0: '📉 下跌', None: '⏳ 待回饋'})
        if '預測正確' in display_df.columns:
            display_df['預測正確'] = display_df['預測正確'].map({True: '✅', False: '❌', None: '-'})
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 單筆刪除
        st.markdown("**刪除特定預測記錄：**")
        dates = [h.get('prediction_date', '未知') for h in history]
        selected_date = st.selectbox("選擇要刪除的預測日期", dates, key="delete_select")
        if st.button("🗑️ 刪除選定記錄", key="delete_selected"):
            if feedback_manager.delete_prediction(symbol_input, selected_date):
                st.success(f"已刪除 {selected_date} 的預測記錄")
                st.rerun()
            else:
                st.error("刪除失敗")
    else:
        st.info("目前沒有預測歷史記錄")


def render_risk_backtest_page(symbol_input: str, backtest_days: int = 365):
    """渲染風險回測頁面"""
    st.subheader("📊 風險回測分析")
    st.markdown("""
    此頁面提供更完整的策略回測，包含：
    - **報酬指標**：策略總報酬、年化報酬、對比買入持有
    - **風險指標**：夏普比率、Sortino 比率、最大回撤
    - **交易統計**：勝率、期望值、利潤因子
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        backtest_days_input = st.slider(
            "回測天數",
            min_value=60,
            max_value=730,
            value=365,
            step=30,
            help="選擇回測的歷史天數"
        )
    
    with col2:
        run_backtest = st.button("🚀 執行風險回測", use_container_width=True)
    
    if run_backtest:
        with st.spinner("正在執行回測分析..."):
            try:
                predictor = get_predictor()
                results = predictor.backtest_with_risk(symbol_input, days=backtest_days_input)
                
                # 顯示主要指標
                st.markdown("### 📈 報酬指標")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "策略總報酬",
                        f"{results['total_return']:.2%}",
                        delta=f"vs 買入持有 {results['buy_hold_return']:.2%}"
                    )
                with col2:
                    st.metric("年化報酬", f"{results['annual_return']:.2%}")
                with col3:
                    st.metric("年化波動", f"{results['volatility']:.2%}")
                with col4:
                    st.metric("交易天數", f"{results['trading_days']} 天")
                
                st.markdown("### ⚖️ 風險調整指標")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sharpe_color = "🟢" if results['sharpe_ratio'] > 1 else "🟡" if results['sharpe_ratio'] > 0 else "🔴"
                    st.metric(
                        f"{sharpe_color} 夏普比率",
                        f"{results['sharpe_ratio']:.2f}",
                        help="大於 1 為優秀，0-1 為可接受，小於 0 為不佳"
                    )
                with col2:
                    st.metric("Sortino 比率", f"{results['sortino_ratio']:.2f}")
                with col3:
                    dd_color = "🟢" if results['max_drawdown'] < 0.1 else "🟡" if results['max_drawdown'] < 0.2 else "🔴"
                    st.metric(
                        f"{dd_color} 最大回撤",
                        f"{results['max_drawdown']:.2%}",
                        help="越小越好，一般希望控制在 20% 以內"
                    )
                
                st.markdown("### 🎯 交易統計")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    win_color = "🟢" if results['win_rate'] > 0.55 else "🟡" if results['win_rate'] > 0.5 else "🔴"
                    st.metric(f"{win_color} 勝率", f"{results['win_rate']:.2%}")
                with col2:
                    st.metric("獲勝次數", f"{results['wins']} 次")
                with col3:
                    st.metric("虧損次數", f"{results['losses']} 次")
                with col4:
                    pf_color = "🟢" if results['profit_factor'] > 1.5 else "🟡" if results['profit_factor'] > 1 else "🔴"
                    st.metric(f"{pf_color} 利潤因子", f"{results['profit_factor']:.2f}")
                
                # 淨值曲線圖
                st.markdown("### 📉 淨值曲線")
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # 策略淨值
                fig.add_trace(go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve'].values,
                    mode='lines',
                    name='策略淨值',
                    line=dict(color='#00d4aa', width=2)
                ))
                
                # 買入持有淨值
                fig.add_trace(go.Scatter(
                    x=results['buy_hold_curve'].index,
                    y=results['buy_hold_curve'].values,
                    mode='lines',
                    name='買入持有',
                    line=dict(color='#7c3aed', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="策略 vs 買入持有",
                    xaxis_title="日期",
                    yaxis_title="淨值",
                    template="plotly_dark",
                    height=400,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 詳細報告
                with st.expander("📋 完整回測報告"):
                    st.code(results['report'], language=None)
                
            except Exception as e:
                st.error(f"❌ 回測過程發生錯誤: {e}")
                import traceback
                st.code(traceback.format_exc())


def main():
    # 標題
    st.markdown('<h1 class="main-header">📈 股票隔日漲跌預測系統</h1>', unsafe_allow_html=True)
    
    # 頁籤
    tab1, tab2, tab3 = st.tabs(["🎯 預測分析", "📝 結果回饋", "📊 風險回測"])
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 股票代號輸入
        symbol_input = st.text_input(
            "股票代號",
            value="2330",
            help="輸入台股代號，例如：2330 (台積電)、2317 (鴻海)"
        )
        
        # 常用股票快速選擇
        st.markdown("**快速選擇熱門股票：**")
        popular_stocks = {
            "2330 台積電": "2330",
            "2317 鴻海": "2317",
            "2454 聯發科": "2454",
            "2881 富邦金": "2881",
            "2882 國泰金": "2882",
            "3008 大立光": "3008",
        }
        
        cols = st.columns(2)
        for i, (name, code) in enumerate(popular_stocks.items()):
            with cols[i % 2]:
                if st.button(name, key=f"btn_{code}"):
                    symbol_input = code
        
        st.divider()
        
        # 歷史資料天數
        history_days = st.slider(
            "訓練資料天數",
            min_value=180,
            max_value=1095,
            value=730,
            step=90,
            help="用於訓練模型的歷史資料天數"
        )
        
        # 訓練模式選擇
        train_mode = st.radio(
            "訓練模式",
            options=["standard", "walk_forward"],
            format_func=lambda x: "🚀 一般訓練" if x == "standard" else "🔄 滾動式訓練（自動回測修正）",
            help="滾動式訓練會用過去資料訓練、近期資料測試，自動將錯誤樣本加入下輪訓練"
        )
        
        if train_mode == "walk_forward":
            n_iterations = st.slider(
                "訓練迭代次數",
                min_value=2,
                max_value=5,
                value=3,
                help="滾動式訓練的迭代次數，越多次可能越準但耗時更長"
            )
        else:
            n_iterations = 1
        
        st.divider()
        
        st.markdown("**☁️ 雲端儲存狀態**")
        try:
            from src.cloud_storage import get_cloud_storage
            cloud = get_cloud_storage()
            if cloud.enabled:
                st.success("✅ Supabase 已連線")
            else:
                st.warning("⚠️ 雲端儲存未啟用")
        except Exception as e:
            st.error(f"❌ 雲端儲存錯誤: {e}")
        
        st.divider()
        
        # 免責聲明
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ 免責聲明</strong><br>
        本系統僅供研究參考，不構成投資建議。
        股市有風險，投資需謹慎。
        </div>
        """, unsafe_allow_html=True)
    
    # 預測分析頁籤
    with tab1:
        # 執行按鈕
        if st.button("🚀 開始預測", type="primary", key="predict_btn"):
            st.session_state['show_prediction'] = True
            st.session_state['current_symbol'] = symbol_input
            
        if st.session_state.get('show_prediction', False) and st.session_state.get('current_symbol') == symbol_input:
            render_prediction_page(symbol_input, history_days, train_mode, n_iterations)
        else:
            # 首頁說明
            st.markdown("""
            <div class="info-box">
            <h3>👋 歡迎使用股票漲跌預測系統</h3>
            <p>本系統使用 <strong>XGBoost 機器學習模型</strong>，整合以下 35 個特徵進行預測：</p>
            <ul>
                <li>📊 <strong>技術面指標</strong>：均線、RSI、MACD、KD、布林通道等</li>
                <li>📈 <strong>動能指標</strong>：ROC、動量、威廉指標等</li>
                <li>💰 <strong>成交量指標</strong>：量比、OBV 等</li>
                <li>🏦 <strong>籌碼面</strong>：三大法人買賣超</li>
                <li>🌍 <strong>市場環境</strong>：大盤走勢、波動率等</li>
            </ul>
            <p>👆 點擊「開始預測」按鈕進行分析</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 使用說明
            st.subheader("💡 使用說明")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **步驟 1：輸入股票代號**
                - 在左側欄位輸入台股代號
                - 例如：2330（台積電）
                
                **步驟 2：調整參數**
                - 可調整訓練資料的天數
                - 建議使用 365-730 天
                """)
            
            with col2:
                st.markdown("""
                **步驟 3：開始預測**
                - 點擊「開始預測」按鈕
                - 系統將自動訓練模型並生成預測
                
                **步驟 4：回饋結果**
                - 隔天結果出爐後，到「結果回饋」頁籤
                - 輸入實際漲跌，幫助模型學習
                """)
    
    # 結果回饋頁籤
    with tab2:
        render_feedback_page(symbol_input)
    
    # 風險回測頁籤
    with tab3:
        render_risk_backtest_page(symbol_input)


if __name__ == "__main__":
    main()
