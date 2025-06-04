"""
增强风险控制模块 - 实现自适应止盈止损策略
包含ATR自适应止损、移动止盈、分批止盈、技术位止损等功能
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class StopLossType(Enum):
    """止损类型枚举"""
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_ADAPTIVE = "atr_adaptive"
    VOLATILITY_PERCENTAGE = "volatility_percentage"
    TECHNICAL_LEVEL = "technical_level"
    HYBRID = "hybrid"

class TakeProfitType(Enum):
    """止盈类型枚举"""
    FIXED_PERCENTAGE = "fixed_percentage"
    DYNAMIC_ATR = "dynamic_atr"
    TRAILING_STOP = "trailing_stop"
    PARTIAL_PROFIT = "partial_profit"
    TECHNICAL_TARGET = "technical_target"

@dataclass
class RiskControlConfig:
    """风险控制配置"""
    # ATR止损配置
    atr_period: int = 14
    atr_multiplier_stop: float = 2.0
    atr_multiplier_take_profit: float = 3.0
    
    # 波动率止损配置
    volatility_lookback: int = 20
    volatility_multiplier: float = 2.0
    
    # 移动止盈配置
    trailing_activation_profit: float = 0.05  # 5%盈利启动移动止盈
    trailing_stop_distance: float = 0.03      # 3%的追踪距离
    trailing_step_size: float = 0.01          # 1%的步进大小
    
    # 分批止盈配置
    partial_profit_levels: List[float] = None
    partial_profit_sizes: List[float] = None
    
    # 技术位止损配置
    technical_buffer_atr: float = 1.0  # 技术位缓冲距离（ATR倍数）
    
    # 混合策略权重
    hybrid_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.partial_profit_levels is None:
            self.partial_profit_levels = [0.03, 0.06, 0.10]  # 3%, 6%, 10%
        if self.partial_profit_sizes is None:
            self.partial_profit_sizes = [0.3, 0.3, 0.4]     # 30%, 30%, 40%
        if self.hybrid_weights is None:
            self.hybrid_weights = {
                'atr': 0.4,
                'volatility': 0.3,
                'technical': 0.3
            }

@dataclass
class PositionRiskState:
    """持仓风险状态"""
    entry_price: float
    entry_time: int
    position_type: int  # 1=多头, -1=空头
    position_size: float
    
    # 止损止盈价位
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    
    # 移动止盈状态
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    highest_profit: float = 0.0
    
    # 分批止盈状态
    partial_profits_taken: List[int] = None
    remaining_position_ratio: float = 1.0
    
    # 技术位信息
    key_support_level: float = 0.0
    key_resistance_level: float = 0.0
    
    def __post_init__(self):
        if self.partial_profits_taken is None:
            self.partial_profits_taken = []

class EnhancedRiskController:
    """增强风险控制器"""
    
    def __init__(self, config: RiskControlConfig = None):
        self.config = config or RiskControlConfig()
        self.logger = logging.getLogger(__name__)
        
        # 风险统计
        self.total_stop_losses = 0
        self.total_take_profits = 0
        self.total_trailing_stops = 0
        self.total_partial_profits = 0
        
        # 性能统计
        self.atr_stop_performance = []
        self.technical_stop_performance = []
        self.trailing_profit_performance = []
    
    def calculate_atr_adaptive_stop_loss(self, df: pd.DataFrame, current_idx: int, 
                                       entry_price: float, position_type: int) -> float:
        """
        ✅ ATR自适应止损计算
        
        Args:
            df: 包含OHLC数据的DataFrame
            current_idx: 当前数据索引
            entry_price: 入场价格
            position_type: 仓位类型 (1=多头, -1=空头)
            
        Returns:
            止损价格
        """
        try:
            # 确保有足够的历史数据
            if current_idx < self.config.atr_period:
                # 数据不足时使用简单百分比止损
                fallback_pct = 0.02  # 2%
                if position_type == 1:
                    return entry_price * (1 - fallback_pct)
                else:
                    return entry_price * (1 + fallback_pct)
            
            # 提取价格数据
            end_idx = current_idx + 1
            start_idx = max(0, end_idx - self.config.atr_period - 10)  # 多取一些数据确保计算准确
            
            high_prices = df['high'].iloc[start_idx:end_idx].values
            low_prices = df['low'].iloc[start_idx:end_idx].values
            close_prices = df['close'].iloc[start_idx:end_idx].values
            
            # 计算ATR
            atr_values = talib.ATR(high_prices, low_prices, close_prices, 
                                 timeperiod=self.config.atr_period)
            current_atr = atr_values[-1]
            
            # 处理NaN值
            if np.isnan(current_atr) or current_atr <= 0:
                # 使用简单范围估算ATR
                recent_ranges = high_prices[-10:] - low_prices[-10:]
                current_atr = np.mean(recent_ranges)
            
            # 计算止损距离
            stop_distance = current_atr * self.config.atr_multiplier_stop
            
            # 确保止损距离合理（不超过10%）
            max_stop_distance = entry_price * 0.10
            stop_distance = min(stop_distance, max_stop_distance)
            
            # 计算止损价格
            if position_type == 1:  # 多头
                stop_loss_price = entry_price - stop_distance
            else:  # 空头
                stop_loss_price = entry_price + stop_distance
            
            # 记录ATR止损信息
            atr_percentage = (stop_distance / entry_price) * 100
            self.logger.debug(f"🔧 ATR自适应止损: ATR={current_atr:.6f}, "
                            f"止损距离={stop_distance:.6f} ({atr_percentage:.2f}%), "
                            f"止损价={stop_loss_price:.6f}")
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"❌ ATR自适应止损计算失败: {e}")
            # 回退到固定百分比止损
            fallback_pct = 0.025  # 2.5%
            if position_type == 1:
                return entry_price * (1 - fallback_pct)
            else:
                return entry_price * (1 + fallback_pct)
    
    def calculate_volatility_adaptive_stop_loss(self, df: pd.DataFrame, current_idx: int,
                                              entry_price: float, position_type: int) -> float:
        """
        ✅ 波动率自适应止损计算
        基于近期价格波动率的标准差
        """
        try:
            if current_idx < self.config.volatility_lookback:
                fallback_pct = 0.02
                if position_type == 1:
                    return entry_price * (1 - fallback_pct)
                else:
                    return entry_price * (1 + fallback_pct)
            
            # 计算近期收益率
            end_idx = current_idx + 1
            start_idx = max(0, end_idx - self.config.volatility_lookback)
            
            close_prices = df['close'].iloc[start_idx:end_idx].values
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # 计算波动率（标准差）
            volatility = np.std(returns)
            
            # 止损距离 = 波动率 * 倍数
            stop_distance_pct = volatility * self.config.volatility_multiplier
            
            # 限制在合理范围内
            stop_distance_pct = np.clip(stop_distance_pct, 0.005, 0.08)  # 0.5%-8%
            
            # 计算止损价格
            if position_type == 1:  # 多头
                stop_loss_price = entry_price * (1 - stop_distance_pct)
            else:  # 空头
                stop_loss_price = entry_price * (1 + stop_distance_pct)
            
            self.logger.debug(f"📊 波动率自适应止损: 波动率={volatility:.4f}, "
                            f"止损比例={stop_distance_pct*100:.2f}%, "
                            f"止损价={stop_loss_price:.6f}")
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"❌ 波动率自适应止损计算失败: {e}")
            fallback_pct = 0.025
            if position_type == 1:
                return entry_price * (1 - fallback_pct)
            else:
                return entry_price * (1 + fallback_pct)
    
    def calculate_technical_level_stop_loss(self, df: pd.DataFrame, current_idx: int,
                                          entry_price: float, position_type: int) -> Tuple[float, Dict]:
        """
        ✅ 关键技术位止损计算
        基于支撑阻力位、摆动点等技术分析
        """
        try:
            technical_info = {}
            
            # 1. 寻找最近的摆动点
            swing_high, swing_low = self._find_recent_swing_points(df, current_idx)
            technical_info['swing_high'] = swing_high
            technical_info['swing_low'] = swing_low
            
            # 2. 寻找布林带支撑阻力
            support_level, resistance_level = self._find_bollinger_levels(df, current_idx)
            technical_info['bb_support'] = support_level
            technical_info['bb_resistance'] = resistance_level
            
            # 3. 计算ATR作为缓冲距离
            atr_buffer = self._calculate_atr_buffer(df, current_idx)
            technical_info['atr_buffer'] = atr_buffer
            
            # 4. 确定最终技术位止损
            if position_type == 1:  # 多头止损
                # 选择最近的支撑位
                potential_stops = []
                
                if swing_low > 0 and swing_low < entry_price:
                    potential_stops.append(swing_low - atr_buffer)
                
                if support_level > 0 and support_level < entry_price:
                    potential_stops.append(support_level - atr_buffer)
                
                # 选择最高的（最接近入场价的）支撑位
                if potential_stops:
                    technical_stop = max(potential_stops)
                    # 确保不超过8%止损
                    max_stop = entry_price * 0.92
                    technical_stop = max(technical_stop, max_stop)
                else:
                    # 没有找到合适的技术位，使用固定比例
                    technical_stop = entry_price * 0.97
                
                technical_info['selected_level'] = 'support'
                
            else:  # 空头止损
                # 选择最近的阻力位
                potential_stops = []
                
                if swing_high > entry_price:
                    potential_stops.append(swing_high + atr_buffer)
                
                if resistance_level > entry_price:
                    potential_stops.append(resistance_level + atr_buffer)
                
                # 选择最低的（最接近入场价的）阻力位
                if potential_stops:
                    technical_stop = min(potential_stops)
                    # 确保不超过8%止损
                    max_stop = entry_price * 1.08
                    technical_stop = min(technical_stop, max_stop)
                else:
                    # 没有找到合适的技术位，使用固定比例
                    technical_stop = entry_price * 1.03
                
                technical_info['selected_level'] = 'resistance'
            
            technical_info['final_stop'] = technical_stop
            
            self.logger.debug(f"🎯 技术位止损: {technical_info}")
            
            return technical_stop, technical_info
            
        except Exception as e:
            self.logger.error(f"❌ 技术位止损计算失败: {e}")
            fallback_pct = 0.03
            if position_type == 1:
                return entry_price * (1 - fallback_pct), {}
            else:
                return entry_price * (1 + fallback_pct), {}
    
    def calculate_hybrid_stop_loss(self, df: pd.DataFrame, current_idx: int,
                                 entry_price: float, position_type: int) -> Tuple[float, Dict]:
        """
        ✅ 混合策略止损计算
        综合ATR、波动率、技术位等多种方法
        """
        try:
            stop_calculations = {}
            
            # 1. ATR自适应止损
            atr_stop = self.calculate_atr_adaptive_stop_loss(df, current_idx, entry_price, position_type)
            stop_calculations['atr_stop'] = atr_stop
            
            # 2. 波动率自适应止损
            vol_stop = self.calculate_volatility_adaptive_stop_loss(df, current_idx, entry_price, position_type)
            stop_calculations['volatility_stop'] = vol_stop
            
            # 3. 技术位止损
            tech_stop, tech_info = self.calculate_technical_level_stop_loss(df, current_idx, entry_price, position_type)
            stop_calculations['technical_stop'] = tech_stop
            stop_calculations['technical_info'] = tech_info
            
            # 4. 加权平均计算最终止损
            weights = self.config.hybrid_weights
            
            if position_type == 1:  # 多头 - 选择较高的止损价（较宽松）
                final_stop = (
                    atr_stop * weights['atr'] +
                    vol_stop * weights['volatility'] +
                    tech_stop * weights['technical']
                )
            else:  # 空头 - 选择较低的止损价（较宽松）
                final_stop = (
                    atr_stop * weights['atr'] +
                    vol_stop * weights['volatility'] +
                    tech_stop * weights['technical']
                )
            
            stop_calculations['final_stop'] = final_stop
            stop_calculations['weights_used'] = weights
            
            self.logger.info(f"🔄 混合策略止损: ATR={atr_stop:.6f}, Vol={vol_stop:.6f}, "
                           f"Tech={tech_stop:.6f}, Final={final_stop:.6f}")
            
            return final_stop, stop_calculations
            
        except Exception as e:
            self.logger.error(f"❌ 混合策略止损计算失败: {e}")
            # 回退到ATR止损
            return self.calculate_atr_adaptive_stop_loss(df, current_idx, entry_price, position_type), {}
    
    def setup_trailing_take_profit(self, position_state: PositionRiskState, 
                                 current_price: float) -> bool:
        """
        ✅ 设置移动止盈机制
        
        Returns:
            是否成功激活移动止盈
        """
        try:
            # 计算当前盈利比例
            if position_state.position_type == 1:  # 多头
                profit_pct = (current_price - position_state.entry_price) / position_state.entry_price
            else:  # 空头
                profit_pct = (position_state.entry_price - current_price) / position_state.entry_price
            
            # 检查是否达到启动条件
            if profit_pct >= self.config.trailing_activation_profit and not position_state.trailing_stop_active:
                position_state.trailing_stop_active = True
                position_state.highest_profit = profit_pct
                
                # 设置初始移动止盈价格
                if position_state.position_type == 1:  # 多头
                    position_state.trailing_stop_price = current_price * (1 - self.config.trailing_stop_distance)
                else:  # 空头
                    position_state.trailing_stop_price = current_price * (1 + self.config.trailing_stop_distance)
                
                self.logger.info(f"🚀 移动止盈激活: 当前盈利={profit_pct*100:.2f}%, "
                               f"移动止损价={position_state.trailing_stop_price:.6f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 移动止盈设置失败: {e}")
            return False
    
    def update_trailing_stop(self, position_state: PositionRiskState, 
                           current_price: float) -> bool:
        """
        ✅ 更新移动止损价格
        
        Returns:
            是否触发移动止损出场
        """
        try:
            if not position_state.trailing_stop_active:
                return False
            
            # 计算当前盈利
            if position_state.position_type == 1:  # 多头
                current_profit = (current_price - position_state.entry_price) / position_state.entry_price
            else:  # 空头
                current_profit = (position_state.entry_price - current_price) / position_state.entry_price
            
            # 更新最高盈利和移动止损价
            if current_profit > position_state.highest_profit:
                position_state.highest_profit = current_profit
                
                # 更新移动止损价格
                if position_state.position_type == 1:  # 多头
                    new_trailing_stop = current_price * (1 - self.config.trailing_stop_distance)
                    position_state.trailing_stop_price = max(position_state.trailing_stop_price, new_trailing_stop)
                else:  # 空头
                    new_trailing_stop = current_price * (1 + self.config.trailing_stop_distance)
                    position_state.trailing_stop_price = min(position_state.trailing_stop_price, new_trailing_stop)
                
                self.logger.debug(f"📈 移动止损更新: 新高盈利={current_profit*100:.2f}%, "
                                f"新止损价={position_state.trailing_stop_price:.6f}")
            
            # 检查是否触发移动止损
            trailing_triggered = False
            if position_state.position_type == 1:  # 多头
                if current_price <= position_state.trailing_stop_price:
                    trailing_triggered = True
            else:  # 空头
                if current_price >= position_state.trailing_stop_price:
                    trailing_triggered = True
            
            if trailing_triggered:
                self.total_trailing_stops += 1
                self.logger.info(f"💫 移动止损触发: 当前价={current_price:.6f}, "
                               f"止损价={position_state.trailing_stop_price:.6f}, "
                               f"最高盈利={position_state.highest_profit*100:.2f}%")
                
                # 记录移动止盈性能
                final_profit = current_profit
                self.trailing_profit_performance.append({
                    'entry_price': position_state.entry_price,
                    'exit_price': current_price,
                    'max_profit': position_state.highest_profit,
                    'final_profit': final_profit,
                    'profit_preserved': final_profit / position_state.highest_profit if position_state.highest_profit > 0 else 0
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 移动止损更新失败: {e}")
            return False
    
    def execute_partial_take_profit(self, position_state: PositionRiskState, 
                                  current_price: float) -> Tuple[bool, float, int]:
        """
        ✅ 执行分批止盈
        
        Returns:
            (是否执行了分批止盈, 平仓比例, 分批级别)
        """
        try:
            # 计算当前盈利
            if position_state.position_type == 1:  # 多头
                profit_pct = (current_price - position_state.entry_price) / position_state.entry_price
            else:  # 空头
                profit_pct = (position_state.entry_price - current_price) / position_state.entry_price
            
            # 检查是否达到分批止盈条件
            for i, target_profit in enumerate(self.config.partial_profit_levels):
                if (profit_pct >= target_profit and 
                    i not in position_state.partial_profits_taken and
                    position_state.remaining_position_ratio > 0):
                    
                    # 执行分批止盈
                    close_ratio = self.config.partial_profit_sizes[i]
                    position_state.partial_profits_taken.append(i)
                    position_state.remaining_position_ratio -= close_ratio
                    
                    self.total_partial_profits += 1
                    
                    self.logger.info(f"💰 分批止盈执行: 级别{i+1}, 目标盈利={target_profit*100:.1f}%, "
                                   f"当前盈利={profit_pct*100:.2f}%, 平仓比例={close_ratio:.1%}, "
                                   f"剩余仓位={position_state.remaining_position_ratio:.1%}")
                    
                    return True, close_ratio, i
            
            return False, 0.0, -1
            
        except Exception as e:
            self.logger.error(f"❌ 分批止盈执行失败: {e}")
            return False, 0.0, -1
    
    def calculate_dynamic_take_profit(self, df: pd.DataFrame, current_idx: int,
                                    entry_price: float, position_type: int) -> float:
        """
        ✅ 动态止盈计算
        基于ATR和技术位的综合止盈策略
        """
        try:
            # 计算ATR基础止盈
            if current_idx >= self.config.atr_period:
                end_idx = current_idx + 1
                start_idx = max(0, end_idx - self.config.atr_period - 5)
                
                high_prices = df['high'].iloc[start_idx:end_idx].values
                low_prices = df['low'].iloc[start_idx:end_idx].values
                close_prices = df['close'].iloc[start_idx:end_idx].values
                
                atr_values = talib.ATR(high_prices, low_prices, close_prices, 
                                     timeperiod=self.config.atr_period)
                current_atr = atr_values[-1]
                
                if np.isnan(current_atr) or current_atr <= 0:
                    current_atr = np.mean(high_prices[-5:] - low_prices[-5:])
            else:
                current_atr = entry_price * 0.02  # 回退值
            
            # ATR止盈距离
            tp_distance = current_atr * self.config.atr_multiplier_take_profit
            
            # 寻找技术目标位
            tech_target = self._find_technical_target(df, current_idx, entry_price, position_type)
            
            # 综合计算止盈价格
            if position_type == 1:  # 多头
                atr_target = entry_price + tp_distance
                final_target = max(atr_target, tech_target) if tech_target > entry_price else atr_target
            else:  # 空头
                atr_target = entry_price - tp_distance
                final_target = min(atr_target, tech_target) if 0 < tech_target < entry_price else atr_target
            
            # 确保最小盈利目标
            min_profit_pct = 0.015  # 最小1.5%盈利
            if position_type == 1:
                min_target = entry_price * (1 + min_profit_pct)
                final_target = max(final_target, min_target)
            else:
                min_target = entry_price * (1 - min_profit_pct)
                final_target = min(final_target, min_target)
            
            profit_pct = abs(final_target - entry_price) / entry_price * 100
            self.logger.debug(f"🎯 动态止盈: ATR目标={atr_target:.6f}, "
                            f"技术目标={tech_target:.6f}, "
                            f"最终目标={final_target:.6f} ({profit_pct:.2f}%)")
            
            return final_target
            
        except Exception as e:
            self.logger.error(f"❌ 动态止盈计算失败: {e}")
            # 回退到固定比例止盈
            fallback_pct = 0.025  # 2.5%
            if position_type == 1:
                return entry_price * (1 + fallback_pct)
            else:
                return entry_price * (1 - fallback_pct)
    
    def _find_recent_swing_points(self, df: pd.DataFrame, current_idx: int, 
                                window: int = 10) -> Tuple[float, float]:
        """寻找最近的摆动高低点"""
        try:
            start_idx = max(0, current_idx - 50)  # 查看最近50根K线
            end_idx = current_idx + 1
            
            highs = df['high'].iloc[start_idx:end_idx].values
            lows = df['low'].iloc[start_idx:end_idx].values
            
            # 寻找摆动高点
            swing_high = 0.0
            for i in range(window, len(highs) - window):
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    swing_high = highs[i]
                    break
            
            # 寻找摆动低点
            swing_low = 0.0
            for i in range(window, len(lows) - window):
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    swing_low = lows[i]
                    break
            
            return swing_high, swing_low
            
        except Exception as e:
            self.logger.error(f"❌ 摆动点寻找失败: {e}")
            return 0.0, 0.0
    
    def _find_bollinger_levels(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float]:
        """寻找布林带支撑阻力位"""
        try:
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper = df['bb_upper'].iloc[current_idx]
                bb_lower = df['bb_lower'].iloc[current_idx]
                return bb_lower, bb_upper
            else:
                return 0.0, 0.0
        except:
            return 0.0, 0.0
    
    def _calculate_atr_buffer(self, df: pd.DataFrame, current_idx: int) -> float:
        """计算ATR缓冲距离"""
        try:
            if current_idx >= self.config.atr_period:
                end_idx = current_idx + 1
                start_idx = max(0, end_idx - self.config.atr_period)
                
                high_prices = df['high'].iloc[start_idx:end_idx].values
                low_prices = df['low'].iloc[start_idx:end_idx].values
                close_prices = df['close'].iloc[start_idx:end_idx].values
                
                atr_values = talib.ATR(high_prices, low_prices, close_prices, 
                                     timeperiod=self.config.atr_period)
                current_atr = atr_values[-1]
                
                if np.isnan(current_atr) or current_atr <= 0:
                    current_atr = np.mean(high_prices - low_prices)
                
                return current_atr * self.config.technical_buffer_atr
            else:
                return df['close'].iloc[current_idx] * 0.005  # 0.5%作为回退
        except:
            return df['close'].iloc[current_idx] * 0.005
    
    def _find_technical_target(self, df: pd.DataFrame, current_idx: int,
                             entry_price: float, position_type: int) -> float:
        """寻找技术目标位"""
        try:
            # 基于布林带和近期高低点
            if position_type == 1:  # 多头目标
                if 'bb_upper' in df.columns:
                    bb_upper = df['bb_upper'].iloc[current_idx]
                    if bb_upper > entry_price:
                        return bb_upper * 0.99  # 略低于布林带上轨
                
                # 寻找近期高点
                start_idx = max(0, current_idx - 20)
                recent_highs = df['high'].iloc[start_idx:current_idx+1]
                resistance = recent_highs.max()
                if resistance > entry_price:
                    return resistance * 0.995
                
                return entry_price * 1.03  # 默认3%目标
                
            else:  # 空头目标
                if 'bb_lower' in df.columns:
                    bb_lower = df['bb_lower'].iloc[current_idx]
                    if 0 < bb_lower < entry_price:
                        return bb_lower * 1.01  # 略高于布林带下轨
                
                # 寻找近期低点
                start_idx = max(0, current_idx - 20)
                recent_lows = df['low'].iloc[start_idx:current_idx+1]
                support = recent_lows.min()
                if 0 < support < entry_price:
                    return support * 1.005
                
                return entry_price * 0.97  # 默认3%目标
                
        except Exception as e:
            self.logger.error(f"❌ 技术目标寻找失败: {e}")
            if position_type == 1:
                return entry_price * 1.025
            else:
                return entry_price * 0.975
    
    def get_risk_control_summary(self) -> Dict:
        """获取风险控制统计摘要"""
        return {
            'total_stop_losses': self.total_stop_losses,
            'total_take_profits': self.total_take_profits,
            'total_trailing_stops': self.total_trailing_stops,
            'total_partial_profits': self.total_partial_profits,
            'avg_atr_stop_efficiency': np.mean(self.atr_stop_performance) if self.atr_stop_performance else 0.0,
            'avg_trailing_preservation': self._calculate_avg_trailing_preservation(),
            'partial_profit_efficiency': self._calculate_partial_profit_efficiency(),
            'config': {
                'atr_period': self.config.atr_period,
                'atr_multiplier_stop': self.config.atr_multiplier_stop,
                'atr_multiplier_take_profit': self.config.atr_multiplier_take_profit,
                'trailing_activation_profit': self.config.trailing_activation_profit,
                'trailing_stop_distance': self.config.trailing_stop_distance,
                'partial_profit_levels': self.config.partial_profit_levels,
                'partial_profit_sizes': self.config.partial_profit_sizes
            }
        }
    
    def _calculate_avg_trailing_preservation(self) -> float:
        """计算移动止盈的平均利润保存率"""
        if not self.trailing_profit_performance:
            return 0.0
        
        # 处理不同格式的性能数据
        preservations = []
        for p in self.trailing_profit_performance:
            if isinstance(p, dict) and 'profit_preserved' in p:
                preservations.append(p['profit_preserved'])
            elif isinstance(p, (int, float)):
                preservations.append(p)  # 直接使用数值
        
        return np.mean(preservations) if preservations else 0.0
    
    def _calculate_partial_profit_efficiency(self) -> float:
        """计算分批止盈效率"""
        if self.total_partial_profits == 0:
            return 0.0
        
        # 简单效率计算：执行的分批止盈次数占比
        return min(1.0, self.total_partial_profits / 100)  # 假设理想情况下100次交易

def main():
    """测试增强风险控制功能"""
    import pandas as pd
    
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    data = {
        'high': np.cumsum(np.random.randn(n) * 0.01) + 100,
        'low': np.cumsum(np.random.randn(n) * 0.01) + 99,
        'close': np.cumsum(np.random.randn(n) * 0.01) + 99.5,
    }
    df = pd.DataFrame(data)
    
    # 创建风险控制器
    config = RiskControlConfig(
        atr_period=14,
        atr_multiplier_stop=2.0,
        trailing_activation_profit=0.03,
        partial_profit_levels=[0.02, 0.05, 0.08]
    )
    risk_controller = EnhancedRiskController(config)
    
    # 测试ATR自适应止损
    current_idx = 50
    entry_price = 100.0
    position_type = 1
    
    atr_stop = risk_controller.calculate_atr_adaptive_stop_loss(df, current_idx, entry_price, position_type)
    print(f"ATR自适应止损: {atr_stop:.4f}")
    
    vol_stop = risk_controller.calculate_volatility_adaptive_stop_loss(df, current_idx, entry_price, position_type)
    print(f"波动率自适应止损: {vol_stop:.4f}")
    
    tech_stop, tech_info = risk_controller.calculate_technical_level_stop_loss(df, current_idx, entry_price, position_type)
    print(f"技术位止损: {tech_stop:.4f}")
    print(f"技术信息: {tech_info}")
    
    hybrid_stop, hybrid_info = risk_controller.calculate_hybrid_stop_loss(df, current_idx, entry_price, position_type)
    print(f"混合策略止损: {hybrid_stop:.4f}")
    
    # 测试动态止盈
    take_profit = risk_controller.calculate_dynamic_take_profit(df, current_idx, entry_price, position_type)
    print(f"动态止盈: {take_profit:.4f}")
    
    # 测试仓位风险状态
    position_state = PositionRiskState(
        entry_price=entry_price,
        entry_time=current_idx,
        position_type=position_type,
        position_size=1000.0
    )
    
    # 模拟价格变化测试移动止盈
    test_prices = [101, 102, 103, 105, 104, 102]
    for price in test_prices:
        risk_controller.setup_trailing_take_profit(position_state, price)
        trailing_triggered = risk_controller.update_trailing_stop(position_state, price)
        partial_executed, close_ratio, level = risk_controller.execute_partial_take_profit(position_state, price)
        
        print(f"价格={price:.2f}, 移动止盈={position_state.trailing_stop_active}, "
              f"止损触发={trailing_triggered}, 分批止盈={partial_executed}")
    
    # 获取统计摘要
    summary = risk_controller.get_risk_control_summary()
    print(f"\n风险控制摘要: {summary}")

if __name__ == "__main__":
    main() 