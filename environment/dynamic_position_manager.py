"""
动态仓位管理器 - 增强版
实现Kelly公式动态仓位和波动率目标仓位策略
支持基于胜率、盈亏比和市场波动的智能仓位调整
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import warnings

from utils.config import get_config
from utils.logger import get_logger

class PositionSizingMethod(Enum):
    """仓位计算方法枚举"""
    FIXED_RATIO = "fixed_ratio"              # 固定比例
    KELLY_FORMULA = "kelly_formula"          # Kelly公式
    VOLATILITY_TARGET = "volatility_target"  # 波动率目标
    KELLY_VOLATILITY = "kelly_volatility"    # Kelly + 波动率混合
    ADAPTIVE_KELLY = "adaptive_kelly"        # 自适应Kelly
    RISK_PARITY = "risk_parity"             # 风险平价

@dataclass
class PositionSizingConfig:
    """动态仓位配置"""
    # 基础配置
    method: PositionSizingMethod = PositionSizingMethod.KELLY_VOLATILITY
    max_position_ratio: float = 0.5  # 最大仓位比例
    min_position_ratio: float = 0.01  # 最小仓位比例
    
    # Kelly公式配置
    kelly_lookback_period: int = 100  # Kelly计算回望期
    kelly_multiplier: float = 0.5     # Kelly乘数（保守系数）
    kelly_update_frequency: int = 20   # Kelly参数更新频率
    min_trades_for_kelly: int = 30     # Kelly计算最少交易数
    
    # 波动率目标配置
    risk_per_trade: float = 0.02      # 每笔交易风险比例（2%）
    volatility_lookback: int = 20     # 波动率计算回望期
    atr_multiplier: float = 2.0       # ATR倍数（止损距离）
    min_stop_distance: float = 0.01   # 最小止损距离（1%）
    max_stop_distance: float = 0.1    # 最大止损距离（10%）
    
    # 混合策略权重
    kelly_weight: float = 0.6         # Kelly策略权重
    volatility_weight: float = 0.4    # 波动率策略权重
    
    # 自适应调整
    enable_adaptive: bool = True      # 启用自适应调整
    performance_threshold: float = 0.1  # 表现阈值
    adjustment_factor: float = 0.2    # 调整因子
    
    # 风险控制
    max_consecutive_losses: int = 5   # 最大连续亏损
    drawdown_reduction_threshold: float = 0.1  # 回撤减仓阈值
    recovery_multiplier: float = 1.2  # 恢复期乘数

@dataclass
class TradingStatistics:
    """交易统计数据"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.total_trades == 0:
            return 0.5  # 默认胜率
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """盈亏比"""
        if self.losing_trades == 0 or self.total_loss == 0:
            return 2.0  # 默认盈亏比
        avg_profit = self.total_profit / max(self.winning_trades, 1)
        avg_loss = abs(self.total_loss) / self.losing_trades
        return avg_profit / avg_loss
    
    @property
    def expectancy(self) -> float:
        """期望值"""
        if self.total_trades == 0:
            return 0.0
        return (self.total_profit + self.total_loss) / self.total_trades

class DynamicPositionManager:
    """动态仓位管理器 - 增强版"""
    
    def __init__(self, config: PositionSizingConfig = None):
        self.config = config or PositionSizingConfig()
        self.logger = get_logger('DynamicPositionManager', 'position_manager.log')
        
        # 交易统计
        self.stats = TradingStatistics()
        self.trade_history = deque(maxlen=self.config.kelly_lookback_period)
        
        # Kelly公式计算缓存
        self.kelly_fraction = 0.1  # 初始Kelly比例
        self.kelly_update_counter = 0
        
        # 波动率计算缓存
        self.volatility_history = deque(maxlen=self.config.volatility_lookback)
        self.current_volatility = 0.02  # 初始波动率
        
        # 自适应调整状态
        self.performance_history = deque(maxlen=50)
        self.adjustment_factor = 1.0
        
        # 账户状态
        self.account_balance = 10000.0
        self.peak_balance = 10000.0
        self.current_drawdown = 0.0
        
        self.logger.info(f"🎯 动态仓位管理器初始化完成，方法: {self.config.method.value}")
    
    def calculate_position_size(self, 
                              account_balance: float,
                              current_price: float,
                              stop_loss_price: float,
                              signal_strength: float = 1.0,
                              market_data: pd.Series = None) -> Tuple[float, Dict]:
        """
        计算动态仓位大小
        
        Args:
            account_balance: 账户余额
            current_price: 当前价格
            stop_loss_price: 止损价格
            signal_strength: 信号强度 (0-1)
            market_data: 市场数据
            
        Returns:
            (position_ratio, calculation_info): 仓位比例和计算信息
        """
        try:
            # 更新账户状态
            self.account_balance = account_balance
            self.peak_balance = max(self.peak_balance, account_balance)
            self.current_drawdown = (self.peak_balance - account_balance) / self.peak_balance
            
            # 🧮 添加详细调试日志 - 输入参数
            self.logger.info(f"🧮 仓位计算开始:")
            self.logger.info(f"   ├─ 当前信号强度: {signal_strength:.4f}")
            self.logger.info(f"   ├─ 账户余额: ${account_balance:.2f}")
            self.logger.info(f"   ├─ 当前价格: ${current_price:.4f}")
            self.logger.info(f"   ├─ 止损价格: ${stop_loss_price:.4f}")
            self.logger.info(f"   ├─ 当前回撤: {self.current_drawdown:.2%}")
            self.logger.info(f"   └─ 计算方法: {self.config.method.value}")
            
            # 根据配置的方法计算仓位
            if self.config.method == PositionSizingMethod.FIXED_RATIO:
                position_ratio, info = self._calculate_fixed_ratio(signal_strength)
            elif self.config.method == PositionSizingMethod.KELLY_FORMULA:
                position_ratio, info = self._calculate_kelly_position()
            elif self.config.method == PositionSizingMethod.VOLATILITY_TARGET:
                position_ratio, info = self._calculate_volatility_target_position(
                    current_price, stop_loss_price, market_data)
            elif self.config.method == PositionSizingMethod.KELLY_VOLATILITY:
                position_ratio, info = self._calculate_kelly_volatility_position(
                    current_price, stop_loss_price, market_data)
            elif self.config.method == PositionSizingMethod.ADAPTIVE_KELLY:
                position_ratio, info = self._calculate_adaptive_kelly_position(
                    current_price, stop_loss_price, signal_strength, market_data)
            else:
                position_ratio, info = self._calculate_fixed_ratio(signal_strength)
            
            # 🧮 记录原始计算结果
            raw_position_ratio = position_ratio
            self.logger.info(f"📊 原始仓位计算结果: {raw_position_ratio:.4f}")
            
            # 应用风险控制
            position_ratio = self._apply_risk_controls(position_ratio)
            self.logger.info(f"⚖️ 风险控制后仓位: {position_ratio:.4f}")
            
            # 应用自适应调整
            if self.config.enable_adaptive:
                position_ratio = self._apply_adaptive_adjustment(position_ratio)
                self.logger.info(f"🎯 自适应调整后仓位: {position_ratio:.4f}")
            
            # 应用信号强度调整
            position_ratio *= signal_strength
            self.logger.info(f"📡 信号强度调整后仓位: {position_ratio:.4f}")
            
            # 最终边界检查
            position_ratio = np.clip(position_ratio, 
                                   self.config.min_position_ratio, 
                                   self.config.max_position_ratio)
            
            # 🧮 计算实际投资金额和验证
            position_value = account_balance * position_ratio
            required_margin = position_value  # 简化：假设1倍杠杆
            
            # ✅ 关键验证：确保position_size > 0 且资金充足
            if position_ratio <= 0:
                self.logger.warning(f"⚠️ 仓位计算为0或负值: {position_ratio:.4f}, 设置为最小仓位")
                position_ratio = self.config.min_position_ratio
                position_value = account_balance * position_ratio
            
            if position_value > account_balance * 0.95:  # 留5%缓冲
                self.logger.warning(f"⚠️ 仓位金额过大: ${position_value:.2f} > ${account_balance*0.95:.2f}, 调整为95%余额")
                position_ratio = 0.95
                position_value = account_balance * position_ratio
            
            # 🧮 最终结果日志
            info['final_position_ratio'] = position_ratio
            info['signal_strength'] = signal_strength
            info['adjustment_factor'] = self.adjustment_factor
            info['current_drawdown'] = self.current_drawdown
            info['position_value'] = position_value
            info['required_margin'] = required_margin
            info['validation_passed'] = position_ratio > 0 and position_value < account_balance
            
            self.logger.info(f"🧮 仓位计算完成:")
            self.logger.info(f"   ├─ 最终建议仓位: {position_ratio:.4f}")
            self.logger.info(f"   ├─ 投资金额: ${position_value:.2f}")
            self.logger.info(f"   ├─ 资金充足性: {'✅ 通过' if info['validation_passed'] else '❌ 失败'}")
            self.logger.info(f"   └─ 计算方法: {self.config.method.value}")
            
            return position_ratio, info
            
        except Exception as e:
            self.logger.error(f"❌ 仓位计算失败: {e}")
            # 返回保守的默认仓位
            return self.config.min_position_ratio, {'error': str(e), 'validation_passed': False}
    
    def _calculate_fixed_ratio(self, signal_strength: float) -> Tuple[float, Dict]:
        """计算固定比例仓位"""
        base_ratio = 0.1  # 基础10%仓位
        position_ratio = base_ratio * signal_strength
        
        info = {
            'method': 'fixed_ratio',
            'base_ratio': base_ratio,
            'signal_adjusted_ratio': position_ratio
        }
        
        return position_ratio, info
    
    def _calculate_kelly_position(self) -> Tuple[float, Dict]:
        """计算Kelly公式仓位"""
        # 更新Kelly参数
        if self.kelly_update_counter % self.config.kelly_update_frequency == 0:
            self._update_kelly_parameters()
        
        self.kelly_update_counter += 1
        
        # 应用Kelly乘数（保守系数）
        position_ratio = self.kelly_fraction * self.config.kelly_multiplier
        
        info = {
            'method': 'kelly_formula',
            'raw_kelly_fraction': self.kelly_fraction,
            'kelly_multiplier': self.config.kelly_multiplier,
            'win_rate': self.stats.win_rate,
            'profit_factor': self.stats.profit_factor,
            'total_trades': self.stats.total_trades
        }
        
        return position_ratio, info
    
    def _calculate_volatility_target_position(self, 
                                            current_price: float,
                                            stop_loss_price: float,
                                            market_data: pd.Series = None) -> Tuple[float, Dict]:
        """计算波动率目标仓位"""
        # 计算止损距离
        if stop_loss_price > 0:
            stop_distance = abs(current_price - stop_loss_price) / current_price
        else:
            # 使用ATR估算止损距离
            stop_distance = self._estimate_stop_distance_from_atr(market_data)
        
        # 确保止损距离在合理范围内
        stop_distance = np.clip(stop_distance, 
                               self.config.min_stop_distance, 
                               self.config.max_stop_distance)
        
        # 根据波动率目标公式计算仓位
        # 仓位规模 = 账户风险阈值 / 止损距离
        position_ratio = self.config.risk_per_trade / stop_distance
        
        info = {
            'method': 'volatility_target',
            'stop_distance': stop_distance,
            'risk_per_trade': self.config.risk_per_trade,
            'current_volatility': self.current_volatility
        }
        
        return position_ratio, info
    
    def _calculate_kelly_volatility_position(self, 
                                           current_price: float,
                                           stop_loss_price: float,
                                           market_data: pd.Series = None) -> Tuple[float, Dict]:
        """计算Kelly + 波动率混合仓位"""
        # 分别计算Kelly和波动率目标仓位
        kelly_ratio, kelly_info = self._calculate_kelly_position()
        vol_ratio, vol_info = self._calculate_volatility_target_position(
            current_price, stop_loss_price, market_data)
        
        # 加权平均
        position_ratio = (kelly_ratio * self.config.kelly_weight + 
                         vol_ratio * self.config.volatility_weight)
        
        info = {
            'method': 'kelly_volatility_hybrid',
            'kelly_component': kelly_ratio,
            'volatility_component': vol_ratio,
            'kelly_weight': self.config.kelly_weight,
            'volatility_weight': self.config.volatility_weight,
            'kelly_info': kelly_info,
            'volatility_info': vol_info
        }
        
        return position_ratio, info
    
    def _calculate_adaptive_kelly_position(self, 
                                         current_price: float,
                                         stop_loss_price: float,
                                         signal_strength: float,
                                         market_data: pd.Series = None) -> Tuple[float, Dict]:
        """计算自适应Kelly仓位"""
        # 基础Kelly计算
        base_kelly_ratio, kelly_info = self._calculate_kelly_position()
        
        # 市场状态调整
        market_regime = self._detect_market_regime(market_data)
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # 信号质量调整
        signal_multiplier = 0.5 + signal_strength * 0.5  # 0.5-1.0范围
        
        # 波动率调整
        vol_adjustment = self._calculate_volatility_adjustment(market_data)
        
        # 综合调整
        position_ratio = (base_kelly_ratio * 
                         regime_multiplier * 
                         signal_multiplier * 
                         vol_adjustment)
        
        info = {
            'method': 'adaptive_kelly',
            'base_kelly_ratio': base_kelly_ratio,
            'market_regime': market_regime,
            'regime_multiplier': regime_multiplier,
            'signal_multiplier': signal_multiplier,
            'volatility_adjustment': vol_adjustment,
            'kelly_info': kelly_info
        }
        
        return position_ratio, info
    
    def _update_kelly_parameters(self):
        """更新Kelly公式参数"""
        if len(self.trade_history) < self.config.min_trades_for_kelly:
            self.kelly_fraction = 0.1  # 默认10%
            return
        
        try:
            # 从交易历史计算胜率和盈亏比
            wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            losses = len(self.trade_history) - wins
            
            if wins == 0 or losses == 0:
                self.kelly_fraction = 0.1
                return
            
            p = wins / len(self.trade_history)  # 胜率
            
            # 计算平均盈亏比
            profits = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
            losses_list = [-trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
            
            if not profits or not losses_list:
                self.kelly_fraction = 0.1
                return
            
            avg_profit = np.mean(profits)
            avg_loss = np.mean(losses_list)
            
            if avg_loss == 0:
                self.kelly_fraction = 0.1
                return
            
            R = avg_profit / avg_loss  # 盈亏比
            
            # Kelly公式: f* = p - (1-p)/R
            kelly_fraction = p - (1 - p) / R
            
            # 限制Kelly分数在合理范围内
            self.kelly_fraction = np.clip(kelly_fraction, 0.01, 0.5)
            
            self.logger.debug(f"📊 Kelly参数更新: p={p:.3f}, R={R:.3f}, f*={kelly_fraction:.3f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Kelly参数更新失败: {e}")
            self.kelly_fraction = 0.1
    
    def _estimate_stop_distance_from_atr(self, market_data: pd.Series = None) -> float:
        """基于ATR估算止损距离"""
        if market_data is None or 'atr_normalized' not in market_data:
            return 0.02  # 默认2%
        
        atr_normalized = market_data.get('atr_normalized', 0.02)
        stop_distance = atr_normalized * self.config.atr_multiplier
        
        return np.clip(stop_distance, 
                      self.config.min_stop_distance, 
                      self.config.max_stop_distance)
    
    def _apply_risk_controls(self, position_ratio: float) -> float:
        """应用风险控制"""
        # 连续亏损控制
        if self.stats.consecutive_losses >= self.config.max_consecutive_losses:
            position_ratio *= 0.5  # 减半仓位
            self.logger.warning(f"⚠️ 连续亏损{self.stats.consecutive_losses}次，减半仓位")
        
        # 回撤控制
        if self.current_drawdown > self.config.drawdown_reduction_threshold:
            drawdown_factor = 1.0 - self.current_drawdown * 2  # 回撤越大，减仓越多
            position_ratio *= max(drawdown_factor, 0.3)  # 最多减至30%
            self.logger.warning(f"⚠️ 当前回撤{self.current_drawdown:.1%}，调整仓位因子至{drawdown_factor:.3f}")
        
        return position_ratio
    
    def _apply_adaptive_adjustment(self, position_ratio: float) -> float:
        """应用自适应调整"""
        if len(self.performance_history) < 10:
            return position_ratio
        
        # 计算最近表现
        recent_performance = np.mean(list(self.performance_history)[-10:])
        
        # 根据表现调整仓位
        if recent_performance > self.config.performance_threshold:
            # 表现良好，适当增加仓位
            self.adjustment_factor = min(self.adjustment_factor * (1 + self.config.adjustment_factor), 1.5)
        elif recent_performance < -self.config.performance_threshold:
            # 表现不佳，减少仓位
            self.adjustment_factor = max(self.adjustment_factor * (1 - self.config.adjustment_factor), 0.5)
        
        return position_ratio * self.adjustment_factor
    
    def _detect_market_regime(self, market_data: pd.Series = None) -> str:
        """检测市场状态"""
        if market_data is None:
            return 'neutral'
        
        # 基于ATR和趋势检测市场状态
        atr = market_data.get('atr_normalized', 0.02)
        trend_strength = market_data.get('trend_strength', 0)
        
        if atr > 0.04:
            return 'high_volatility'
        elif atr < 0.015:
            if abs(trend_strength) > 0.5:
                return 'trending'
            else:
                return 'low_volatility'
        else:
            return 'neutral'
    
    def _get_regime_multiplier(self, market_regime: str) -> float:
        """获取市场状态乘数"""
        multipliers = {
            'trending': 1.3,        # 趋势市场增加仓位
            'high_volatility': 0.7, # 高波动市场减少仓位
            'low_volatility': 1.1,  # 低波动市场适当增加
            'neutral': 1.0          # 中性市场保持不变
        }
        return multipliers.get(market_regime, 1.0)
    
    def _calculate_volatility_adjustment(self, market_data: pd.Series = None) -> float:
        """计算波动率调整因子"""
        if market_data is None:
            return 1.0
        
        current_vol = market_data.get('atr_normalized', 0.02)
        
        # 更新波动率历史
        self.volatility_history.append(current_vol)
        
        if len(self.volatility_history) < 5:
            return 1.0
        
        # 计算相对波动率
        avg_vol = np.mean(self.volatility_history)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # 波动率调整：高波动减仓，低波动增仓
        if vol_ratio > 1.5:
            return 0.7  # 高波动减仓30%
        elif vol_ratio < 0.7:
            return 1.2  # 低波动增仓20%
        else:
            return 1.0
    
    def update_trade_result(self, pnl: float, trade_info: Dict = None):
        """更新交易结果"""
        try:
            # 更新基础统计
            self.stats.total_trades += 1
            
            if pnl > 0:
                self.stats.winning_trades += 1
                self.stats.total_profit += pnl
                self.stats.consecutive_wins += 1
                self.stats.consecutive_losses = 0
            else:
                self.stats.losing_trades += 1
                self.stats.total_loss += pnl
                self.stats.consecutive_losses += 1
                self.stats.consecutive_wins = 0
                self.stats.max_consecutive_losses = max(
                    self.stats.max_consecutive_losses, 
                    self.stats.consecutive_losses
                )
            
            # 更新交易历史
            trade_record = {
                'pnl': pnl,
                'timestamp': trade_info.get('timestamp') if trade_info else None,
                'position_ratio': trade_info.get('position_ratio') if trade_info else None
            }
            self.trade_history.append(trade_record)
            
            # 更新表现历史
            if self.account_balance > 0:
                performance = pnl / self.account_balance
                self.performance_history.append(performance)
            
            self.logger.debug(f"📈 交易结果更新: PnL={pnl:.2f}, 胜率={self.stats.win_rate:.1%}")
            
        except Exception as e:
            self.logger.error(f"❌ 更新交易结果失败: {e}")
    
    def get_position_sizing_summary(self) -> Dict:
        """获取仓位管理摘要"""
        return {
            'method': self.config.method.value,
            'current_kelly_fraction': self.kelly_fraction,
            'current_adjustment_factor': self.adjustment_factor,
            'current_drawdown': self.current_drawdown,
            'trading_statistics': {
                'total_trades': self.stats.total_trades,
                'win_rate': self.stats.win_rate,
                'profit_factor': self.stats.profit_factor,
                'expectancy': self.stats.expectancy,
                'consecutive_losses': self.stats.consecutive_losses,
                'max_consecutive_losses': self.stats.max_consecutive_losses
            },
            'config': {
                'max_position_ratio': self.config.max_position_ratio,
                'kelly_multiplier': self.config.kelly_multiplier,
                'risk_per_trade': self.config.risk_per_trade,
                'enable_adaptive': self.config.enable_adaptive
            }
        }
    
    def reset_statistics(self):
        """重置统计数据"""
        self.stats = TradingStatistics()
        self.trade_history.clear()
        self.performance_history.clear()
        self.kelly_fraction = 0.1
        self.adjustment_factor = 1.0
        self.logger.info("📊 仓位管理统计数据已重置")

def main():
    """测试动态仓位管理器"""
    print("🎯 测试动态仓位管理器")
    
    # 创建不同配置的仓位管理器
    configs = [
        PositionSizingConfig(method=PositionSizingMethod.KELLY_FORMULA),
        PositionSizingConfig(method=PositionSizingMethod.VOLATILITY_TARGET),
        PositionSizingConfig(method=PositionSizingMethod.KELLY_VOLATILITY),
        PositionSizingConfig(method=PositionSizingMethod.ADAPTIVE_KELLY)
    ]
    
    for config in configs:
        print(f"\n📊 测试方法: {config.method.value}")
        
        manager = DynamicPositionManager(config)
        
        # 模拟一些交易结果
        for i in range(50):
            # 模拟盈亏
            pnl = np.random.normal(10, 30)  # 平均盈利10，标准差30
            manager.update_trade_result(pnl)
            
            # 计算仓位
            position_ratio, info = manager.calculate_position_size(
                account_balance=10000 + i * 50,
                current_price=100.0,
                stop_loss_price=98.0,
                signal_strength=0.8
            )
            
            if i % 10 == 0:
                print(f"  交易{i+1}: 仓位比例={position_ratio:.3f}")
        
        # 显示摘要
        summary = manager.get_position_sizing_summary()
        print(f"  最终胜率: {summary['trading_statistics']['win_rate']:.1%}")
        print(f"  最终Kelly分数: {summary['current_kelly_fraction']:.3f}")

if __name__ == "__main__":
    main() 