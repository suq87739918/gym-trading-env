"""
Trading Visualization Analysis Module
✅ Enhanced Version: Individual trade P&L curves + Position curves + Detailed analysis
Provides interpretable output and trading behavior charts
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import warnings

from utils.config import get_config
from utils.logger import get_logger

class TradingVisualizer:
    """Trading Visualization Analyzer - Enhanced Version"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('TradingVisualizer', 'visualizer.log')
        
        # Set font and style for matplotlib
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style(self.config.get('PLOT_STYLE', 'darkgrid'))
        
        # Color configuration
        self.colors = {
            'bullish': '#00FF00',
            'bearish': '#FF0000',
            'neutral': '#808080',
            'buy': '#4CAF50',
            'sell': '#F44336',
            'close': '#2196F3',
            'price': '#1976D2',
            'volume': '#9C27B0',
            'indicator': '#FF9800',
            'smc': '#E91E63',
            'profit': '#4CAF50',
            'loss': '#F44336',
            'leverage': '#FF9800'
        }
    
    def create_enhanced_trading_dashboard(self, df: pd.DataFrame, trade_history: List[Dict], 
                                        portfolio_history: List[float], 
                                        reward_breakdown_history: List[Dict] = None,
                                        save_path: str = None) -> str:
        """
        ✅ Create Enhanced Trading Dashboard
        Including: Price candles, Trade P&L, Position changes, Leverage usage, Signal analysis
        """
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=7, cols=2,
                shared_xaxes=True,
                subplot_titles=[
                    'SOL/USDT Price Movement & Trading Points', 'Trade P&L Analysis',
                    'RSI Indicator & Overbought/Oversold', 'Position Type & Leverage Usage',
                    'SMC Signal Strength Analysis', 'Volume & Volatility',
                    'Portfolio Value Evolution', 'Reward Breakdown Details',
                    'Signal Confluence Analysis', 'Risk Control Indicators',
                    'Trading Frequency Statistics', 'Position Holding Time',
                    'Performance Metrics Overview', 'Market State Recognition'
                ],
                row_heights=[0.2, 0.15, 0.12, 0.12, 0.12, 0.1, 0.1],
                column_widths=[0.7, 0.3],
                vertical_spacing=0.02,
                specs=[
                    [{"colspan": 2}, None],  # Price chart occupies full row
                    [{}, {}],               # RSI and Position
                    [{}, {}],               # SMC and Volume
                    [{}, {}],               # Portfolio value and Reward
                    [{}, {}],               # Signal confluence and Risk control
                    [{}, {}],               # Trading frequency and Position time
                    [{"colspan": 2}, None]   # Performance metrics overview occupies full row
                ]
            )
            
            # Prepare time index
            timestamps = df.index[-len(portfolio_history):] if len(df) >= len(portfolio_history) else df.index
            df_subset = df.iloc[-len(timestamps):] if len(df) >= len(timestamps) else df
            
            # 1. Price movement and trading points (Row 1, full)
            self._add_enhanced_price_chart(fig, df_subset, trade_history, timestamps, row=1, col=1)
            
            # 2. RSI indicators (Row 2, left)
            self._add_rsi_analysis(fig, df_subset, timestamps, row=2, col=1)
            
            # 3. Trade P&L analysis (Row 2, right)
            self._add_trade_pnl_analysis(fig, trade_history, row=2, col=2)
            
            # 4. SMC signal analysis (Row 3, left)
            self._add_smc_analysis(fig, df_subset, timestamps, row=3, col=1)
            
            # 5. Position & leverage (Row 3, right)
            self._add_position_leverage_analysis(fig, trade_history, timestamps, row=3, col=2)
            
            # 6. Portfolio value (Row 4, left)
            self._add_portfolio_evolution(fig, portfolio_history, timestamps, row=4, col=1)
            
            # 7. Volume & volatility (Row 4, right)
            self._add_volume_volatility_analysis(fig, df_subset, timestamps, row=4, col=2)
            
            # 8. Signal confluence (Row 5, left)
            self._add_signal_confluence_analysis(fig, df_subset, timestamps, row=5, col=1)
            
            # 9. Reward breakdown (Row 5, right)
            if reward_breakdown_history:
                self._add_reward_breakdown_analysis(fig, reward_breakdown_history, timestamps, row=5, col=2)
            
            # 10. Trading frequency statistics (Row 6, left)
            self._add_trading_frequency_analysis(fig, trade_history, row=6, col=1)
            
            # 11. Risk control indicators (Row 6, right)
            self._add_risk_control_analysis(fig, trade_history, portfolio_history, row=6, col=2)
            
            # 12. Performance metrics overview (Row 7, full)
            self._add_performance_summary(fig, trade_history, portfolio_history, row=7, col=1)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'SOL/USDT Intelligent Trading System - Enhanced Analysis Dashboard',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                height=1800,
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            # Save chart
            if save_path is None:
                save_path = f"results/enhanced_trading_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            fig.write_html(save_path)
            self.logger.info(f"Enhanced trading dashboard saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced trading dashboard: {e}")
            return None
    
    def _add_enhanced_price_chart(self, fig, df: pd.DataFrame, trade_history: List[Dict], 
                                timestamps, row: int, col: int):
        """增强的价格图表"""
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=timestamps,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='SOL/USDT',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=row, col=col
        )
        
        # EMA线
        if 'ema_fast' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['ema_fast'],
                    mode='lines',
                    name='EMA-Fast',
                    line=dict(color='orange', width=1),
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        if 'ema_slow' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['ema_slow'],
                    mode='lines',
                    name='EMA-Slow',
                    line=dict(color='purple', width=1),
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['bb_upper'],
                    mode='lines',
                    name='Bollinger Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill=None,
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['bb_lower'],
                    mode='lines',
                    name='Bollinger Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # ✅ Enhanced trading markers
        for trade in trade_history:
            if trade.get('timestamp') and trade.get('timestamp') in timestamps:
                action = trade['action']
                price = trade['price']
                pnl = trade.get('pnl', 0)
                leverage = trade.get('leverage', 1.0)
                
                # Select style based on trade type
                if '开多' in action or 'Long' in action:
                    symbol = 'triangle-up'
                    color = self.colors['buy']
                    size = 15
                    action_display = 'Long Open'
                elif '开空' in action or 'Short' in action:
                    symbol = 'triangle-down'
                    color = self.colors['sell']
                    size = 15
                    action_display = 'Short Open'
                elif '平仓' in action or 'Close' in action:
                    symbol = 'circle'
                    color = self.colors['profit'] if pnl > 0 else self.colors['loss']
                    size = 12
                    action_display = 'Close Position'
                else:
                    continue
                
                # Build hover information
                hover_text = f"Action: {action_display}<br>Price: ${price:.4f}<br>Leverage: {leverage:.2f}x<br>"
                if pnl is not None:
                    hover_text += f"P&L: ${pnl:.2f}<br>Return: {trade.get('pnl_pct', 0)*100:.2f}%"
                
                fig.add_trace(
                    go.Scatter(
                        x=[trade['timestamp']],
                        y=[price],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=size,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        name=action_display,
                        hovertext=hover_text,
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    def _add_trade_pnl_analysis(self, fig, trade_history: List[Dict], row: int, col: int):
        """✅ Trade P&L Analysis"""
        if not trade_history:
            return
        
        # Extract P&L data from closing trades
        close_trades = [trade for trade in trade_history if '平仓' in trade.get('action', '') or 'Close' in trade.get('action', '')]
        
        if not close_trades:
            return
        
        trade_numbers = list(range(1, len(close_trades) + 1))
        pnls = [trade.get('pnl', 0) for trade in close_trades]
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        
        # P&L bar chart
        colors = [self.colors['profit'] if pnl > 0 else self.colors['loss'] for pnl in pnls]
        
        fig.add_trace(
            go.Bar(
                x=trade_numbers,
                y=pnls,
                name='Individual P&L',
                marker_color=colors,
                opacity=0.7,
                yaxis='y'
            ),
            row=row, col=col
        )
        
        # Cumulative P&L curve
        fig.add_trace(
            go.Scatter(
                x=trade_numbers,
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='gold', width=3),
                marker=dict(size=6),
                yaxis='y2'
            ),
            row=row, col=col
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=row, col=col)
    
    def _add_position_leverage_analysis(self, fig, trade_history: List[Dict], 
                                      timestamps, row: int, col: int):
        """✅ Position Type & Leverage Usage Analysis"""
        if not trade_history:
            return
        
        # Prepare position and leverage time series
        position_data = []
        leverage_data = []
        
        current_position = 0  # 0: No position, 1: Long, -1: Short
        current_leverage = 1.0
        
        trade_idx = 0
        for ts in timestamps:
            # Check if any trade occurred
            while trade_idx < len(trade_history) and trade_history[trade_idx].get('timestamp') == ts:
                trade = trade_history[trade_idx]
                action = trade.get('action', '')
                
                if '开多' in action or 'Long' in action:
                    current_position = 1
                    current_leverage = trade.get('leverage', 1.0)
                elif '开空' in action or 'Short' in action:
                    current_position = -1
                    current_leverage = trade.get('leverage', 1.0)
                elif '平仓' in action or 'Close' in action:
                    current_position = 0
                    current_leverage = 1.0
                
                trade_idx += 1
            
            position_data.append(current_position)
            leverage_data.append(current_leverage)
        
        # Position type chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=position_data,
                mode='lines',
                name='Position Type',
                line=dict(color='cyan', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,255,255,0.2)'
            ),
            row=row, col=col
        )
        
        # Leverage usage chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=leverage_data,
                mode='lines+markers',
                name='Leverage Multiple',
                line=dict(color=self.colors['leverage'], width=2),
                marker=dict(size=4),
                yaxis='y2'
            ),
            row=row, col=col
        )
    
    def _add_rsi_analysis(self, fig, df: pd.DataFrame, timestamps, row: int, col: int):
        """RSI Analysis"""
        if 'rsi' not in df.columns:
            return
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )
        
        # 超买超卖区域
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=col)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=col)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=row, col=col)
        
        # 背景色
        fig.add_shape(
            type="rect",
            x0=timestamps[0], x1=timestamps[-1],
            y0=70, y1=100,
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
        
        fig.add_shape(
            type="rect",
            x0=timestamps[0], x1=timestamps[-1],
            y0=0, y1=30,
            fillcolor="rgba(0,255,0,0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
    
    def _add_smc_analysis(self, fig, df: pd.DataFrame, timestamps, row: int, col: int):
        """SMC信号分析"""
        if 'smc_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['smc_signal'],
                    mode='lines',
                    name='SMC信号',
                    line=dict(color=self.colors['smc'], width=2),
                    fill='tozeroy'
                ),
                row=row, col=col
            )
        
        # BOS信号点
        if 'bos_bullish' in df.columns and 'bos_bearish' in df.columns:
            bull_mask = df['bos_bullish'] == 1
            bear_mask = df['bos_bearish'] == 1
            
            if bull_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[bull_mask],
                        y=[0.5] * bull_mask.sum(),
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='BOS牛市',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            if bear_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[bear_mask],
                        y=[-0.5] * bear_mask.sum(),
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='BOS熊市',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    def _add_portfolio_evolution(self, fig, portfolio_history: List[float], 
                               timestamps, row: int, col: int):
        """组合价值演变"""
        if not portfolio_history:
            return
        
        initial_value = portfolio_history[0]
        returns = [(v / initial_value - 1) * 100 for v in portfolio_history]
        
        # 组合价值
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=portfolio_history,
                mode='lines',
                name='组合价值',
                line=dict(color='gold', width=3)
            ),
            row=row, col=col
        )
        
        # 收益率
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=returns,
                mode='lines',
                name='累积收益率(%)',
                line=dict(color='cyan', width=2),
                yaxis='y2'
            ),
            row=row, col=col
        )
        
        # 基准线
        fig.add_hline(y=initial_value, line_dash="dash", line_color="white", 
                     opacity=0.5, row=row, col=col)
    
    def _add_volume_volatility_analysis(self, fig, df: pd.DataFrame, 
                                      timestamps, row: int, col: int):
        """成交量与波动率分析"""
        # 成交量
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=df['volume'],
                name='成交量',
                marker_color=self.colors['volume'],
                opacity=0.6
            ),
            row=row, col=col
        )
        
        # 波动率
        if 'atr_normalized' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['atr_normalized'],
                    mode='lines',
                    name='ATR波动率',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ),
                row=row, col=col
            )
    
    def _add_signal_confluence_analysis(self, fig, df: pd.DataFrame, 
                                      timestamps, row: int, col: int):
        """信号汇聚度分析"""
        if 'signal_confluence' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['signal_confluence'],
                    mode='lines',
                    name='信号汇聚度',
                    line=dict(color='orange', width=2),
                    fill='tozeroy'
                ),
                row=row, col=col
            )
            
            # 高汇聚度阈值线
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         opacity=0.7, row=row, col=col)
    
    def _add_reward_breakdown_analysis(self, fig, reward_breakdown_history: List[Dict], 
                                     timestamps, row: int, col: int):
        """Reward分解分析"""
        if not reward_breakdown_history:
            return
        
        # 提取主要reward组件
        reward_types = ['price_change_reward', 'trend_alignment', 'signal_confluence', 
                       'leverage_bonus', 'frequency_penalty', 'hold_duration_penalty']
        
        for reward_type in reward_types:
            values = [breakdown.get(reward_type, 0) for breakdown in reward_breakdown_history]
            
            if any(abs(v) > 0.001 for v in values):  # 只显示有意义的数据
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name=reward_type.replace('_', ' ').title(),
                        line=dict(width=1),
                        opacity=0.8
                    ),
                    row=row, col=col
                )
    
    def _add_trading_frequency_analysis(self, fig, trade_history: List[Dict], 
                                      row: int, col: int):
        """✅ Trading Frequency Statistics"""
        if not trade_history:
            return
        
        # Count trades by action type
        action_counts = {}
        for trade in trade_history:
            action = trade.get('action', 'Unknown')
            # Translate Chinese action names to English
            if '开多' in action:
                action = 'Long Open'
            elif '开空' in action:
                action = 'Short Open'
            elif '平仓' in action:
                action = 'Close Position'
            
            action_counts[action] = action_counts.get(action, 0) + 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        # Bar chart to show trading distribution
        fig.add_trace(
            go.Bar(
                x=actions,
                y=counts,
                name="Trading Distribution",
                marker_color=['#2E8B57', '#CD5C5C', '#4682B4', '#FF8C00'][:len(actions)]
            ),
            row=row, col=col
        )
        
        # Update y-axis title
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_risk_control_analysis(self, fig, trade_history: List[Dict], 
                                 portfolio_history: List[float], row: int, col: int):
        """✅ Risk Control Indicators"""
        if not portfolio_history:
            return
        
        # Calculate drawdown
        peak = portfolio_history[0]
        drawdowns = []
        
        for value in portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        # Show drawdown curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(drawdowns))),
                y=[-dd * 100 for dd in drawdowns],  # Negative values for display
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)'
            ),
            row=row, col=col
        )
        
        # Maximum drawdown line
        max_drawdown = max(drawdowns)
        fig.add_hline(y=-max_drawdown * 100, line_dash="dash", line_color="red", 
                     row=row, col=col)
    
    def _add_performance_summary(self, fig, trade_history: List[Dict], 
                               portfolio_history: List[float], row: int, col: int):
        """✅ Performance Metrics Overview"""
        if not trade_history or not portfolio_history:
            return
        
        # Calculate key indicators
        total_trades = len([t for t in trade_history if '平仓' in t.get('action', '') or 'Close' in t.get('action', '')])
        winning_trades = len([t for t in trade_history if ('平仓' in t.get('action', '') or 'Close' in t.get('action', '')) and t.get('pnl', 0) > 0])
        
        win_rate = winning_trades / max(total_trades, 1)
        total_return = (portfolio_history[-1] - portfolio_history[0]) / portfolio_history[0]
        
        # Average leverage
        avg_leverage = np.mean([t.get('leverage', 1.0) for t in trade_history if t.get('leverage')])
        
        # Create horizontal bar chart for key metrics
        metrics = ['Total Trades', 'Win Rate (%)', 'Total Return (%)', 'Avg Leverage (x)']
        values = [
            total_trades,
            win_rate * 100,
            total_return * 100,
            avg_leverage
        ]
        
        # Normalize values for better visualization
        normalized_values = []
        display_values = []
        for i, (metric, value) in enumerate(zip(metrics, values)):
            if 'Trades' in metric:
                normalized_values.append(value / 100)  # Scale down trade count
                display_values.append(f"{int(value)}")
            elif 'Rate' in metric or 'Return' in metric:
                normalized_values.append(value)
                display_values.append(f"{value:.1f}%")
            else:
                normalized_values.append(value)
                display_values.append(f"{value:.2f}x")
        
        # Color code the bars
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        
        fig.add_trace(
            go.Bar(
                x=normalized_values,
                y=metrics,
                orientation='h',
                name='Performance Metrics',
                marker_color=colors,
                text=display_values,
                textposition='auto',
            ),
            row=row, col=col
        )
        
        # Update x-axis title
        fig.update_xaxes(title_text="Normalized Values", row=row, col=col)

    def create_backtest_analysis_chart(self, df: pd.DataFrame, trade_history: List[Dict], 
                                     reward_history: List[float] = None,
                                     portfolio_history: List[float] = None,
                                     save_path: str = None) -> str:
        """
        ✅ Create Backtest Analysis Chart
        Compatible with main.py calling interface
        """
        try:
            # Use existing enhanced dashboard functionality
            result_path = self.create_enhanced_trading_dashboard(
                df=df,
                trade_history=trade_history,
                portfolio_history=portfolio_history,
                reward_breakdown_history=None,
                save_path=save_path
            )
            
            if result_path:
                self.logger.info(f"Backtest analysis chart generated: {result_path}")
                return result_path
            else:
                self.logger.error("Failed to generate backtest analysis chart")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create backtest analysis chart: {e}")
            return None

    def create_performance_analysis_chart(self, summary: Dict, save_path: str = None) -> str:
        """
        ✅ Create Performance Analysis Chart
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Trading Performance Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Profit distribution pie chart
            win_rate = summary.get('win_rate', 0) * 100
            loss_rate = 100 - win_rate
            
            axes[0, 0].pie([win_rate, loss_rate], 
                          labels=[f'Profitable Trades ({win_rate:.1f}%)', f'Loss Trades ({loss_rate:.1f}%)'],
                          colors=[self.colors['profit'], self.colors['loss']],
                          autopct='%1.1f%%')
            axes[0, 0].set_title('Win Rate Distribution')
            
            # 2. Key metrics bar chart
            metrics = ['Total Return', 'Max Drawdown', 'Sharpe Ratio', 'Profit Factor']
            values = [
                summary.get('total_return', 0) * 100,
                summary.get('max_drawdown', 0) * 100,
                summary.get('sharpe_ratio', 0),
                summary.get('profit_factor', 0)
            ]
            
            colors = [self.colors['profit'] if v > 0 else self.colors['loss'] for v in values]
            axes[0, 1].bar(metrics, values, color=colors)
            axes[0, 1].set_title('Key Performance Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Portfolio value curve
            portfolio_history = summary.get('portfolio_history', [])
            if portfolio_history:
                axes[1, 0].plot(portfolio_history, color=self.colors['price'], linewidth=2)
                axes[1, 0].set_title('Portfolio Value Evolution')
                axes[1, 0].set_xlabel('Trading Steps')
                axes[1, 0].set_ylabel('Portfolio Value ($)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Trading statistics table
            stats_text = f"""
            Total Trades: {summary.get('total_trades', 0)}
            Win Rate: {win_rate:.2f}%
            Total Return: {summary.get('total_return', 0)*100:.2f}%
            Max Drawdown: {summary.get('max_drawdown', 0)*100:.2f}%
            Average Leverage: {summary.get('avg_leverage', 1.0):.2f}x
            Total Fees: ${summary.get('total_fees', 0):.2f}
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Detailed Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"results/performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance analysis chart saved: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create performance analysis chart: {e}")
            return None

    def create_signal_analysis_report(self, df: pd.DataFrame, trade_history: List[Dict], 
                                    save_path: str = None) -> str:
        """
        ✅ Create Signal Analysis Report
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Signal Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. SMC signal strength distribution
            if 'smc_signal' in df.columns:
                axes[0, 0].hist(df['smc_signal'].dropna(), bins=50, alpha=0.7, color=self.colors['smc'])
                axes[0, 0].set_title('SMC Signal Strength Distribution')
                axes[0, 0].set_xlabel('Signal Strength')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. RSI distribution
            if 'rsi' in df.columns:
                axes[0, 1].hist(df['rsi'].dropna(), bins=50, alpha=0.7, color=self.colors['indicator'])
                axes[0, 1].axvline(x=70, color='red', linestyle='--', label='Overbought')
                axes[0, 1].axvline(x=30, color='green', linestyle='--', label='Oversold')
                axes[0, 1].set_title('RSI Indicator Distribution')
                axes[0, 1].set_xlabel('RSI Value')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Trading behavior analysis
            if trade_history:
                actions = [trade.get('action', 'Unknown') for trade in trade_history]
                # Translate Chinese action names
                translated_actions = []
                for action in actions:
                    if '开多' in action:
                        translated_actions.append('Long Open')
                    elif '开空' in action:
                        translated_actions.append('Short Open')
                    elif '平仓' in action:
                        translated_actions.append('Close Position')
                    else:
                        translated_actions.append('Unknown')
                
                action_counts = {}
                for action in translated_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                if action_counts:
                    axes[1, 0].bar(action_counts.keys(), action_counts.values(), 
                                  color=[self.colors['buy'], self.colors['sell'], self.colors['close']][:len(action_counts)])
                    axes[1, 0].set_title('Trading Behavior Distribution')
                    axes[1, 0].set_ylabel('Count')
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Leverage usage analysis
            if trade_history:
                leverages = [trade.get('leverage', 1.0) for trade in trade_history if trade.get('leverage')]
                if leverages:
                    axes[1, 1].hist(leverages, bins=20, alpha=0.7, color=self.colors['leverage'])
                    axes[1, 1].set_title('Leverage Usage Distribution')
                    axes[1, 1].set_xlabel('Leverage Multiple')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"results/signal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Signal analysis report saved: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Failed to create signal analysis report: {e}")
            return None

def main():
    """Main function for testing visualization features"""
    print("🎨 Enhanced Trading Visualization Analyzer Test")
    
    # Test code can be added here
    visualizer = TradingVisualizer()
    print("✅ Enhanced visualizer initialization completed")
    print("📊 New features:")
    print("  - Individual trade P&L curve")
    print("  - Position & leverage analysis")
    print("  - Trading frequency statistics")
    print("  - Risk control indicators")
    print("  - Comprehensive performance summary")

if __name__ == "__main__":
    main() 