"""
風險指標與回測模組
計算策略層 KPI，評估模型實際交易效益
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from datetime import datetime


class RiskMetrics:
    """風險指標計算器"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, predictions: pd.Series) -> pd.Series:
        """
        根據預測計算策略報酬
        
        Args:
            prices: 收盤價序列
            predictions: 預測方向 (1=做多, 0=做空/不交易)
            
        Returns:
            每日報酬率序列
        """
        # 計算實際報酬率
        actual_returns = prices.pct_change().shift(-1)  # 隔日報酬
        
        # 根據預測方向調整報酬（預測上漲就做多，預測下跌就做空）
        strategy_returns = actual_returns * predictions.map({1: 1, 0: -1})
        
        return strategy_returns.dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                                risk_free_rate: float = 0.02,
                                annualize: bool = True) -> float:
        """
        計算夏普比率
        
        Args:
            returns: 報酬率序列
            risk_free_rate: 年化無風險利率
            annualize: 是否年化
            
        Returns:
            夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # 日均報酬與標準差
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 日化無風險利率
        daily_rf = risk_free_rate / 252
        
        # 夏普比率
        sharpe = (mean_return - daily_rf) / std_return
        
        if annualize:
            sharpe *= np.sqrt(252)  # 年化
        
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        計算最大回撤
        
        Args:
            prices: 價格或淨值序列
            
        Returns:
            (最大回撤比例, 高點日期, 低點日期)
        """
        if len(prices) == 0:
            return 0.0, None, None
        
        # 累積最高點
        cummax = prices.cummax()
        
        # 回撤
        drawdown = (prices - cummax) / cummax
        
        # 最大回撤
        max_dd = drawdown.min()
        
        # 找到最大回撤的位置
        max_dd_idx = drawdown.idxmin()
        peak_idx = prices[:max_dd_idx].idxmax() if max_dd_idx else None
        
        return abs(max_dd), peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_win_rate(predictions: pd.Series, 
                           actual_directions: pd.Series) -> Dict[str, float]:
        """
        計算勝率相關指標
        
        Args:
            predictions: 預測方向
            actual_directions: 實際方向
            
        Returns:
            勝率統計
        """
        if len(predictions) == 0:
            return {'win_rate': 0, 'total_trades': 0, 'wins': 0, 'losses': 0}
        
        correct = (predictions == actual_directions).sum()
        total = len(predictions)
        
        return {
            'win_rate': correct / total if total > 0 else 0,
            'total_trades': total,
            'wins': correct,
            'losses': total - correct
        }
    
    @staticmethod
    def calculate_expected_value(returns: pd.Series, 
                                  predictions: pd.Series,
                                  actual_directions: pd.Series) -> Dict[str, float]:
        """
        計算期望值
        
        Args:
            returns: 報酬率序列
            predictions: 預測方向
            actual_directions: 實際方向
            
        Returns:
            期望值統計
        """
        if len(returns) == 0:
            return {
                'expected_value': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # 區分贏輸
        correct_mask = predictions == actual_directions
        
        wins = returns[correct_mask]
        losses = returns[~correct_mask]
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        # 期望值 = 勝率 * 平均獲利 - 敗率 * 平均虧損
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # 利潤因子
        total_win = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        return {
            'expected_value': expected_value,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99
        }
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                                 risk_free_rate: float = 0.02,
                                 annualize: bool = True) -> float:
        """
        計算 Sortino 比率（只考慮下行風險）
        
        Args:
            returns: 報酬率序列
            risk_free_rate: 年化無風險利率
            annualize: 是否年化
            
        Returns:
            Sortino 比率
        """
        if len(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        # 只計算負報酬的標準差
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_std == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_std
        
        if annualize:
            sortino *= np.sqrt(252)
        
        return sortino


class Backtester:
    """回測引擎"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.metrics = RiskMetrics()
    
    def run_backtest(self, 
                     prices: pd.Series,
                     predictions: pd.Series,
                     transaction_cost: float = 0.001425,
                     tax_rate: float = 0.003) -> Dict:
        """
        執行回測
        
        Args:
            prices: 收盤價序列
            predictions: 預測方向序列
            transaction_cost: 手續費率
            tax_rate: 交易稅率
            
        Returns:
            回測結果
        """
        if len(prices) != len(predictions):
            raise ValueError("prices 和 predictions 長度必須相同")
        
        # 對齊索引
        common_idx = prices.index.intersection(predictions.index)
        prices = prices.loc[common_idx]
        predictions = predictions.loc[common_idx]
        
        # 計算每日報酬
        daily_returns = prices.pct_change().shift(-1).dropna()
        predictions = predictions.iloc[:-1]  # 對齊
        
        # 策略報酬（考慮交易成本）
        position_changes = predictions.diff().abs().fillna(0)
        costs = position_changes * (transaction_cost * 2 + tax_rate)
        
        strategy_returns = daily_returns * predictions.map({1: 1, 0: -1}) - costs
        
        # 計算淨值曲線
        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital
        
        # 計算買入持有報酬
        buy_hold_returns = (1 + daily_returns).cumprod() * self.initial_capital
        
        # 實際方向
        actual_directions = (daily_returns > 0).astype(int)
        
        # 計算各項指標
        max_dd, peak_date, trough_date = self.metrics.calculate_max_drawdown(equity_curve)
        win_stats = self.metrics.calculate_win_rate(predictions, actual_directions)
        ev_stats = self.metrics.calculate_expected_value(strategy_returns, predictions, actual_directions)
        
        return {
            'equity_curve': equity_curve,
            'buy_hold_curve': buy_hold_returns,
            'total_return': (equity_curve.iloc[-1] / self.initial_capital - 1) if len(equity_curve) > 0 else 0,
            'buy_hold_return': (buy_hold_returns.iloc[-1] / self.initial_capital - 1) if len(buy_hold_returns) > 0 else 0,
            'sharpe_ratio': self.metrics.calculate_sharpe_ratio(strategy_returns),
            'sortino_ratio': self.metrics.calculate_sortino_ratio(strategy_returns),
            'max_drawdown': max_dd,
            'max_dd_peak': peak_date,
            'max_dd_trough': trough_date,
            **win_stats,
            **ev_stats,
            'trading_days': len(strategy_returns),
            'annual_return': ((1 + strategy_returns.mean()) ** 252 - 1) if len(strategy_returns) > 0 else 0,
            'volatility': strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else 0,
        }
    
    def generate_report(self, backtest_result: Dict) -> str:
        """生成回測報告"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                      回 測 報 告                              ║
╠══════════════════════════════════════════════════════════════╣
║  交易天數: {backtest_result['trading_days']:>6} 天                                  ║
╠══════════════════════════════════════════════════════════════╣
║  【報酬指標】                                                  ║
║  策略總報酬: {backtest_result['total_return']:>8.2%}                                  ║
║  買入持有報酬: {backtest_result['buy_hold_return']:>8.2%}                                ║
║  年化報酬: {backtest_result['annual_return']:>8.2%}                                    ║
║  年化波動度: {backtest_result['volatility']:>8.2%}                                   ║
╠══════════════════════════════════════════════════════════════╣
║  【風險調整報酬】                                              ║
║  夏普比率: {backtest_result['sharpe_ratio']:>8.2f}                                    ║
║  Sortino 比率: {backtest_result['sortino_ratio']:>8.2f}                               ║
║  最大回撤: {backtest_result['max_drawdown']:>8.2%}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  【交易統計】                                                  ║
║  勝率: {backtest_result['win_rate']:>8.2%}                                        ║
║  獲勝次數: {backtest_result['wins']:>6} 次                                     ║
║  虧損次數: {backtest_result['losses']:>6} 次                                     ║
║  期望值: {backtest_result['expected_value']:>8.4f}                                     ║
║  利潤因子: {backtest_result['profit_factor']:>8.2f}                                    ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report


if __name__ == "__main__":
    # 測試
    np.random.seed(42)
    
    # 模擬價格
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    prices = pd.Series(100 * (1 + np.random.randn(252).cumsum() * 0.02), index=dates)
    
    # 模擬預測
    predictions = pd.Series(np.random.randint(0, 2, 252), index=dates)
    
    # 回測
    backtester = Backtester()
    results = backtester.run_backtest(prices, predictions)
    
    print(backtester.generate_report(results))
