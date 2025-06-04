"""
SMC (Smart Money Concepts) 信号计算模块
包含PO3阶段、BOS、Order Block、Liquidity Sweep等市场结构分析信号
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from utils.config import get_config
from utils.logger import get_logger

class SMCSignals:
    """SMC信号计算器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('SMCSignals', 'smc_signals.log')
        
        # SMC参数
        self.po3_lookback = self.config.get('PO3_LOOKBACK', 20)
        self.bos_threshold = self.config.get('BOS_THRESHOLD', 0.002)
        self.order_block_min_size = self.config.get('ORDER_BLOCK_MIN_SIZE', 10)
        self.liquidity_threshold = self.config.get('LIQUIDITY_THRESHOLD', 1.5)
    
    def calculate_all_smc_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有SMC信号 - 增强版
        集成技术指标过滤和信号融合策略
        """
        try:
            df_with_smc = df.copy()
            
            # 1. 基础SMC信号计算
            df_with_smc = self.identify_swing_points(df_with_smc)
            df_with_smc = self.identify_po3_phases(df_with_smc)
            df_with_smc = self.identify_bos(df_with_smc)
            df_with_smc = self.identify_order_blocks(df_with_smc)
            df_with_smc = self.identify_liquidity_sweeps(df_with_smc)
            df_with_smc = self.identify_fair_value_gaps(df_with_smc)
            df_with_smc = self.identify_market_structure(df_with_smc)
            df_with_smc = self.calculate_smc_signal_strength(df_with_smc)
            
            # 2. ✅ 增强：组合信号计算（集成技术指标）
            df_with_smc = self._add_enhanced_combined_signals(df_with_smc)
            
            # 3. ✅ 增强：应用信号过滤器
            try:
                from data.signal_filter import EnhancedSignalFilter
                signal_filter = EnhancedSignalFilter()
                df_with_smc = signal_filter.apply_enhanced_signal_filter(df_with_smc, 'enhanced_smc_signal')
                
                # 记录过滤器摘要
                filter_summary = signal_filter.get_filter_summary(df_with_smc)
                self.logger.info(f"🔍 信号过滤器摘要: {filter_summary}")
                
            except ImportError as e:
                self.logger.warning(f"信号过滤器导入失败，跳过过滤步骤: {e}")
            except Exception as e:
                self.logger.error(f"应用信号过滤器失败: {e}")
            
            self.logger.info(f"✅ 增强SMC信号计算完成，总特征数: {len(df_with_smc.columns)}")
            return df_with_smc
            
        except Exception as e:
            self.logger.exception(f"❌ 计算SMC信号时发生错误: {e}")
            return df
    
    def identify_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """识别摆动高点和低点"""
        try:
            # 识别摆动高点
            df['swing_high'] = 0
            df['swing_low'] = 0
            
            for i in range(window, len(df) - window):
                # 检查是否为摆动高点
                if (df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max() and
                    df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i+1]):
                    df['swing_high'].iloc[i] = 1
                
                # 检查是否为摆动低点
                if (df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min() and
                    df['low'].iloc[i] < df['low'].iloc[i-1] and
                    df['low'].iloc[i] < df['low'].iloc[i+1]):
                    df['swing_low'].iloc[i] = 1
            
            # 计算最近的摆动点位置
            df['last_swing_high'] = df['high'].where(df['swing_high'] == 1).fillna(method='ffill')
            df['last_swing_low'] = df['low'].where(df['swing_low'] == 1).fillna(method='ffill')
            
            self.logger.debug("摆动点识别完成")
            
        except Exception as e:
            self.logger.error(f"识别摆动点失败: {e}")
        
        return df
    
    def identify_po3_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别PO3 (Power of Three) 阶段"""
        try:
            # 初始化PO3阶段
            df['po3_accumulation'] = 0
            df['po3_manipulation'] = 0
            df['po3_distribution'] = 0
            df['po3_phase'] = 0  # 0: 无明确阶段, 1: 积累, 2: 操控, 3: 分配
            
            # 使用滚动窗口分析PO3阶段
            for i in range(self.po3_lookback, len(df)):
                window_data = df.iloc[i-self.po3_lookback:i+1]
                
                # 计算价格区间和波动性
                price_range = window_data['high'].max() - window_data['low'].min()
                price_std = window_data['close'].std()
                volume_ratio = window_data['volume'].mean() / df['volume'].iloc[:i+1].mean()
                
                # 积累阶段: 低波动, 高成交量, 横盘整理
                if (price_std < df['close'].iloc[:i+1].std() * 0.5 and
                    volume_ratio > 1.2 and
                    price_range < df['close'].iloc[i] * 0.03):
                    df['po3_accumulation'].iloc[i] = 1
                    df['po3_phase'].iloc[i] = 1
                
                # 操控阶段: 急剧价格移动，假突破
                recent_move = abs(df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
                if (recent_move > 0.02 and
                    volume_ratio > 1.5):
                    # 检查是否为假突破
                    if (df['close'].iloc[i-1] > df['last_swing_high'].iloc[i-1] and
                        df['close'].iloc[i] < df['last_swing_high'].iloc[i-1]):
                        df['po3_manipulation'].iloc[i] = 1
                        df['po3_phase'].iloc[i] = 2
                    elif (df['close'].iloc[i-1] < df['last_swing_low'].iloc[i-1] and
                          df['close'].iloc[i] > df['last_swing_low'].iloc[i-1]):
                        df['po3_manipulation'].iloc[i] = 1
                        df['po3_phase'].iloc[i] = 2
                
                # 分配阶段: 强势突破后的持续移动
                if (recent_move > 0.03 and
                    volume_ratio > 1.8 and
                    df['po3_manipulation'].iloc[i-5:i].sum() > 0):
                    df['po3_distribution'].iloc[i] = 1
                    df['po3_phase'].iloc[i] = 3
            
            # 计算PO3阶段强度
            df['po3_strength'] = (df['po3_accumulation'] * 0.3 + 
                                df['po3_manipulation'] * 0.5 + 
                                df['po3_distribution'] * 0.8)
            
            self.logger.debug("PO3阶段识别完成")
            
        except Exception as e:
            self.logger.error(f"识别PO3阶段失败: {e}")
        
        return df
    
    def identify_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别BOS (Break of Structure) 信号"""
        try:
            df['bos_bullish'] = 0
            df['bos_bearish'] = 0
            df['bos_strength'] = 0.0
            
            for i in range(10, len(df)):
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]
                current_close = df['close'].iloc[i]
                prev_close = df['close'].iloc[i-1]
                
                # 查找最近的重要阻力和支撑位
                recent_high = df['high'].iloc[i-10:i].max()
                recent_low = df['low'].iloc[i-10:i].min()
                
                # 牛市结构突破
                if (current_close > recent_high and
                    (current_close - recent_high) / recent_high > self.bos_threshold):
                    # 确认突破强度
                    volume_confirmation = df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.2
                    price_momentum = (current_close - prev_close) / prev_close
                    
                    if volume_confirmation and price_momentum > 0.005:
                        df['bos_bullish'].iloc[i] = 1
                        df['bos_strength'].iloc[i] = min(price_momentum * 10, 1.0)
                
                # 熊市结构突破
                if (current_close < recent_low and
                    (recent_low - current_close) / recent_low > self.bos_threshold):
                    # 确认突破强度
                    volume_confirmation = df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.2
                    price_momentum = abs((current_close - prev_close) / prev_close)
                    
                    if volume_confirmation and price_momentum > 0.005:
                        df['bos_bearish'].iloc[i] = 1
                        df['bos_strength'].iloc[i] = min(price_momentum * 10, 1.0)
            
            # 综合BOS信号
            df['bos_signal'] = df['bos_bullish'] - df['bos_bearish']
            
            self.logger.debug("BOS信号识别完成")
            
        except Exception as e:
            self.logger.error(f"识别BOS信号失败: {e}")
        
        return df
    
    def identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别Order Block (订单区块)"""
        try:
            df['bullish_order_block'] = 0
            df['bearish_order_block'] = 0
            df['order_block_strength'] = 0.0
            
            for i in range(self.order_block_min_size, len(df) - 5):
                # 寻找牛市订单区块
                # 条件：连续下跌后出现强势上涨
                lookback_start = max(0, i - self.order_block_min_size)
                recent_candles = df.iloc[lookback_start:i+1]
                
                # 检查是否有连续下跌
                declining_closes = (recent_candles['close'].diff() < 0).sum()
                if declining_closes >= self.order_block_min_size * 0.7:
                    # 检查后续是否有强势反弹
                    future_candles = df.iloc[i:min(i+6, len(df))]
                    if len(future_candles) > 3:
                        max_future_high = future_candles['high'].max()
                        current_low = df['low'].iloc[i]
                        
                        reversal_strength = (max_future_high - current_low) / current_low
                        if reversal_strength > 0.01:  # 至少1%的反弹
                            df['bullish_order_block'].iloc[i] = 1
                            df['order_block_strength'].iloc[i] = min(reversal_strength * 50, 1.0)
                
                # 检查熊市订单区块
                # 条件：连续上涨后出现强势下跌
                rising_closes = (recent_candles['close'].diff() > 0).sum()
                if rising_closes >= self.order_block_min_size * 0.7:
                    # 检查后续是否有强势下跌
                    future_candles = df.iloc[i:min(i+6, len(df))]
                    if len(future_candles) > 3:
                        min_future_low = future_candles['low'].min()
                        current_high = df['high'].iloc[i]
                        
                        decline_strength = (current_high - min_future_low) / current_high
                        if decline_strength > 0.01:  # 至少1%的下跌
                            df['bearish_order_block'].iloc[i] = 1
                            df['order_block_strength'].iloc[i] = min(decline_strength * 50, 1.0)
            
            # 综合订单区块信号
            df['order_block_signal'] = df['bullish_order_block'] - df['bearish_order_block']
            
            self.logger.debug("Order Block识别完成")
            
        except Exception as e:
            self.logger.error(f"识别Order Block失败: {e}")
        
        return df
    
    def identify_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别Liquidity Sweep (流动性扫荡)"""
        try:
            df['liquidity_sweep_high'] = 0
            df['liquidity_sweep_low'] = 0
            df['liquidity_sweep_strength'] = 0.0
            
            for i in range(20, len(df)):
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]
                current_volume = df['volume'].iloc[i]
                
                # 寻找最近的显著高点和低点
                lookback_highs = df['high'].iloc[i-20:i]
                lookback_lows = df['low'].iloc[i-20:i]
                lookback_volumes = df['volume'].iloc[i-20:i]
                
                # 计算平均成交量
                avg_volume = lookback_volumes.mean()
                
                # 识别高点流动性扫荡
                recent_high = lookback_highs.max()
                if (current_high > recent_high and
                    current_volume > avg_volume * self.liquidity_threshold):
                    
                    # 检查是否快速回落（假突破特征）
                    if i < len(df) - 3:
                        future_lows = df['low'].iloc[i+1:i+4]
                        if len(future_lows) > 0 and future_lows.min() < recent_high:
                            sweep_strength = (current_volume / avg_volume) / self.liquidity_threshold
                            df['liquidity_sweep_high'].iloc[i] = 1
                            df['liquidity_sweep_strength'].iloc[i] = min(sweep_strength, 2.0)
                
                # 识别低点流动性扫荡
                recent_low = lookback_lows.min()
                if (current_low < recent_low and
                    current_volume > avg_volume * self.liquidity_threshold):
                    
                    # 检查是否快速反弹（假跌破特征）
                    if i < len(df) - 3:
                        future_highs = df['high'].iloc[i+1:i+4]
                        if len(future_highs) > 0 and future_highs.max() > recent_low:
                            sweep_strength = (current_volume / avg_volume) / self.liquidity_threshold
                            df['liquidity_sweep_low'].iloc[i] = 1
                            df['liquidity_sweep_strength'].iloc[i] = min(sweep_strength, 2.0)
            
            # 综合流动性扫荡信号
            df['liquidity_sweep_signal'] = df['liquidity_sweep_high'] - df['liquidity_sweep_low']
            
            self.logger.debug("Liquidity Sweep识别完成")
            
        except Exception as e:
            self.logger.error(f"识别Liquidity Sweep失败: {e}")
        
        return df
    
    def identify_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别Fair Value Gap (公允价值缺口)"""
        try:
            df['fvg_bullish'] = 0
            df['fvg_bearish'] = 0
            df['fvg_size'] = 0.0
            
            for i in range(2, len(df)):
                # 获取三根连续K线
                if i >= 2:
                    candle1 = df.iloc[i-2]  # 第一根K线
                    candle2 = df.iloc[i-1]  # 中间K线
                    candle3 = df.iloc[i]    # 第三根K线
                    
                    # 牛市FVG: 第一根K线的高点 < 第三根K线的低点
                    if candle1['high'] < candle3['low']:
                        gap_size = (candle3['low'] - candle1['high']) / candle1['high']
                        if gap_size > 0.001:  # 至少0.1%的缺口
                            df['fvg_bullish'].iloc[i] = 1
                            df['fvg_size'].iloc[i] = gap_size
                    
                    # 熊市FVG: 第一根K线的低点 > 第三根K线的高点
                    elif candle1['low'] > candle3['high']:
                        gap_size = (candle1['low'] - candle3['high']) / candle1['low']
                        if gap_size > 0.001:  # 至少0.1%的缺口
                            df['fvg_bearish'].iloc[i] = 1
                            df['fvg_size'].iloc[i] = gap_size
            
            # 综合FVG信号
            df['fvg_signal'] = df['fvg_bullish'] - df['fvg_bearish']
            
            self.logger.debug("Fair Value Gap识别完成")
            
        except Exception as e:
            self.logger.error(f"识别Fair Value Gap失败: {e}")
        
        return df
    
    def identify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别整体市场结构"""
        try:
            df['market_structure'] = 0  # -1: 熊市, 0: 震荡, 1: 牛市
            df['structure_strength'] = 0.0
            
            # 使用摆动点分析市场结构
            for i in range(50, len(df)):
                # 获取最近的摆动高点和低点
                recent_swing_highs = df['high'].iloc[i-50:i][df['swing_high'].iloc[i-50:i] == 1]
                recent_swing_lows = df['low'].iloc[i-50:i][df['swing_low'].iloc[i-50:i] == 1]
                
                if len(recent_swing_highs) >= 2 and len(recent_swing_lows) >= 2:
                    # 分析高点和低点的趋势
                    highs_trend = np.polyfit(range(len(recent_swing_highs)), recent_swing_highs.values, 1)[0]
                    lows_trend = np.polyfit(range(len(recent_swing_lows)), recent_swing_lows.values, 1)[0]
                    
                    # 牛市结构: 高点和低点都在上升
                    if highs_trend > 0 and lows_trend > 0:
                        df['market_structure'].iloc[i] = 1
                        df['structure_strength'].iloc[i] = min((highs_trend + lows_trend) / df['close'].iloc[i] * 1000, 1.0)
                    
                    # 熊市结构: 高点和低点都在下降
                    elif highs_trend < 0 and lows_trend < 0:
                        df['market_structure'].iloc[i] = -1
                        df['structure_strength'].iloc[i] = min(abs(highs_trend + lows_trend) / df['close'].iloc[i] * 1000, 1.0)
                    
                    # 震荡结构: 高点下降但低点上升，或相反
                    else:
                        df['market_structure'].iloc[i] = 0
                        df['structure_strength'].iloc[i] = 0.1
            
            self.logger.debug("市场结构识别完成")
            
        except Exception as e:
            self.logger.error(f"识别市场结构失败: {e}")
        
        return df
    
    def calculate_smc_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合SMC信号强度"""
        try:
            # 权重设置
            weights = {
                'po3_strength': 0.15,
                'bos_strength': 0.25,
                'order_block_strength': 0.20,
                'liquidity_sweep_strength': 0.20,
                'fvg_size': 0.10,
                'structure_strength': 0.10
            }
            
            # 计算加权信号强度
            df['smc_bullish_strength'] = 0.0
            df['smc_bearish_strength'] = 0.0
            
            for i in range(len(df)):
                bullish_signals = 0.0
                bearish_signals = 0.0
                
                # PO3信号
                if df['po3_phase'].iloc[i] == 3:  # 分配阶段
                    bullish_signals += df['po3_strength'].iloc[i] * weights['po3_strength']
                elif df['po3_phase'].iloc[i] == 2:  # 操控阶段
                    bearish_signals += df['po3_strength'].iloc[i] * weights['po3_strength']
                
                # BOS信号
                if df['bos_bullish'].iloc[i]:
                    bullish_signals += df['bos_strength'].iloc[i] * weights['bos_strength']
                elif df['bos_bearish'].iloc[i]:
                    bearish_signals += df['bos_strength'].iloc[i] * weights['bos_strength']
                
                # Order Block信号
                if df['bullish_order_block'].iloc[i]:
                    bullish_signals += df['order_block_strength'].iloc[i] * weights['order_block_strength']
                elif df['bearish_order_block'].iloc[i]:
                    bearish_signals += df['order_block_strength'].iloc[i] * weights['order_block_strength']
                
                # Liquidity Sweep信号
                if df['liquidity_sweep_high'].iloc[i]:
                    bearish_signals += df['liquidity_sweep_strength'].iloc[i] * weights['liquidity_sweep_strength']
                elif df['liquidity_sweep_low'].iloc[i]:
                    bullish_signals += df['liquidity_sweep_strength'].iloc[i] * weights['liquidity_sweep_strength']
                
                # FVG信号
                if df['fvg_bullish'].iloc[i]:
                    bullish_signals += df['fvg_size'].iloc[i] * weights['fvg_size'] * 100
                elif df['fvg_bearish'].iloc[i]:
                    bearish_signals += df['fvg_size'].iloc[i] * weights['fvg_size'] * 100
                
                # 市场结构信号
                if df['market_structure'].iloc[i] == 1:
                    bullish_signals += df['structure_strength'].iloc[i] * weights['structure_strength']
                elif df['market_structure'].iloc[i] == -1:
                    bearish_signals += df['structure_strength'].iloc[i] * weights['structure_strength']
                
                df['smc_bullish_strength'].iloc[i] = min(bullish_signals, 1.0)
                df['smc_bearish_strength'].iloc[i] = min(bearish_signals, 1.0)
            
            # 综合SMC信号
            df['smc_signal'] = df['smc_bullish_strength'] - df['smc_bearish_strength']
            df['smc_signal_abs'] = abs(df['smc_signal'])
            
            # ✅ 增加组合信号识别 - 提升交易机会
            df = self._add_combined_signals(df)
            
            self.logger.debug("SMC信号强度计算完成")
            
        except Exception as e:
            self.logger.error(f"计算SMC信号强度失败: {e}")
        
        return df
    
    def _add_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 增加组合信号识别 - 提升交易频率同时维持胜率
        实现用户要求的组合信号：
        1. SMC + K线形态组合
        2. RSI结合SMC信号
        3. PO3 + EMA多空排列
        4. 信号打分机制: signal_score = x*PO3 + y*RSI + z*structure
        """
        try:
            # 初始化组合信号
            df['combined_bullish_score'] = 0.0
            df['combined_bearish_score'] = 0.0
            df['signal_quality_score'] = 0.0
            
            for i in range(max(20, len(df) - len(df) + 20), len(df)):
                current_data = df.iloc[i]
                bullish_score = 0.0
                bearish_score = 0.0
                
                # ===== 组合方式1: SMC + K线形态 =====
                # CHoCH + 连续阳线/阴线确认
                if current_data.get('po3_phase', 0) == 3:  # CHoCH (分配阶段)
                    # 检查最近3根K线形态
                    recent_closes = df['close'].iloc[max(0, i-2):i+1]
                    recent_opens = df['open'].iloc[max(0, i-2):i+1]
                    
                    # 连续阳线确认
                    consecutive_bullish = all(close > open for close, open in zip(recent_closes, recent_opens))
                    if consecutive_bullish:
                        bullish_score += 0.4  # SMC+K线形态组合分数
                        self.logger.debug(f"📈 CHoCH+连续阳线确认，位置{i}")
                    
                    # 连续阴线确认
                    consecutive_bearish = all(close < open for close, open in zip(recent_closes, recent_opens))
                    if consecutive_bearish:
                        bearish_score += 0.4
                        self.logger.debug(f"📉 CHoCH+连续阴线确认，位置{i}")
                
                # BOS + 2连阳/阴确认
                if current_data.get('bos_bullish', 0):
                    recent_bullish_candles = sum(1 for j in range(max(0, i-1), i+1) 
                                               if df['close'].iloc[j] > df['open'].iloc[j])
                    if recent_bullish_candles >= 2:
                        bullish_score += 0.5  # 强势确认
                        self.logger.debug(f"💪 BOS+2连阳确认，位置{i}")
                
                if current_data.get('bos_bearish', 0):
                    recent_bearish_candles = sum(1 for j in range(max(0, i-1), i+1) 
                                               if df['close'].iloc[j] < df['open'].iloc[j])
                    if recent_bearish_candles >= 2:
                        bearish_score += 0.5
                        self.logger.debug(f"💪 BOS+2连阴确认，位置{i}")
                
                # ===== 组合方式2: RSI结合SMC =====
                rsi = current_data.get('rsi', 50)
                smc_signal = current_data.get('smc_signal', 0)
                
                # RSI超卖 + SMC看涨信号
                if rsi < 30 and (current_data.get('bos_bullish', 0) or current_data.get('bullish_order_block', 0)):
                    bullish_score += 0.6  # 高置信度组合
                    self.logger.debug(f"🔄 RSI超卖+SMC看涨，RSI={rsi:.1f}，位置{i}")
                
                # RSI超买 + SMC看跌信号
                if rsi > 70 and (current_data.get('bos_bearish', 0) or current_data.get('bearish_order_block', 0)):
                    bearish_score += 0.6
                    self.logger.debug(f"🔄 RSI超买+SMC看跌，RSI={rsi:.1f}，位置{i}")
                
                # RSI中性区间 + 强SMC信号
                if 40 <= rsi <= 60 and abs(smc_signal) > 0.3:
                    if smc_signal > 0:
                        bullish_score += 0.3
                    else:
                        bearish_score += 0.3
                    self.logger.debug(f"⚖️ RSI中性+强SMC信号，RSI={rsi:.1f}，SMC={smc_signal:.2f}，位置{i}")
                
                # ===== 组合方式3: PO3 + EMA多空排列 =====
                if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
                    ema_fast = current_data.get('ema_fast', 0)
                    ema_slow = current_data.get('ema_slow', 0)
                    current_price = current_data.get('close', 0)
                    po3_phase = current_data.get('po3_phase', 0)
                    
                    # PO3分配阶段 + EMA多头排列
                    if po3_phase == 3 and current_price > ema_fast > ema_slow:
                        bullish_score += 0.5  # 趋势+结构双重确认
                        self.logger.debug(f"📊 PO3分配+EMA多头排列，位置{i}")
                    
                    # PO3操控阶段 + EMA空头排列
                    elif po3_phase == 2 and current_price < ema_fast < ema_slow:
                        bearish_score += 0.5
                        self.logger.debug(f"📊 PO3操控+EMA空头排列，位置{i}")
                    
                    # 新增：PO3积累阶段识别机会
                    elif po3_phase == 1:  # 积累阶段
                        # 在积累阶段寻找方向突破机会
                        if current_price > ema_fast and ema_fast > ema_slow:
                            bullish_score += 0.3  # 积累阶段的多头突破
                        elif current_price < ema_fast and ema_fast < ema_slow:
                            bearish_score += 0.3  # 积累阶段的空头突破
                
                # ===== 组合方式4: 多重技术指标汇聚 =====
                # MACD + SMC + 布林带位置
                if 'macd' in df.columns and 'bb_position' in df.columns:
                    macd = current_data.get('macd', 0)
                    bb_position = current_data.get('bb_position', 0.5)
                    
                    # MACD向上 + 布林带下轨附近 + SMC看涨
                    if macd > 0 and bb_position < 0.2 and smc_signal > 0.2:
                        bullish_score += 0.4
                        self.logger.debug(f"🎯 MACD+布林带下轨+SMC看涨汇聚，位置{i}")
                    
                    # MACD向下 + 布林带上轨附近 + SMC看跌
                    elif macd < 0 and bb_position > 0.8 and smc_signal < -0.2:
                        bearish_score += 0.4
                        self.logger.debug(f"🎯 MACD+布林带上轨+SMC看跌汇聚，位置{i}")
                
                # ===== 新增组合方式5: 成交量确认 =====
                if 'volume' in df.columns:
                    current_volume = current_data.get('volume', 0)
                    avg_volume = df['volume'].iloc[max(0, i-19):i+1].mean()  # 20周期平均成交量
                    
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    # 高成交量 + 任何看涨信号
                    if volume_ratio > 1.5 and bullish_score > 0:
                        bullish_score += 0.2  # 成交量确认奖励
                        self.logger.debug(f"📢 高成交量确认看涨信号，成交量比={volume_ratio:.2f}，位置{i}")
                    
                    # 高成交量 + 任何看跌信号
                    elif volume_ratio > 1.5 and bearish_score > 0:
                        bearish_score += 0.2  # 成交量确认奖励
                        self.logger.debug(f"📢 高成交量确认看跌信号，成交量比={volume_ratio:.2f}，位置{i}")
                
                # ===== ✅ 信号打分机制: signal_score = x*PO3 + y*RSI + z*structure =====
                # 按照用户要求实现打分机制
                po3_score = 0
                rsi_score = 0  
                structure_score = 0
                
                # PO3贡献分数
                po3_phase = current_data.get('po3_phase', 0)
                if po3_phase == 3:  # 分配阶段
                    po3_score = 0.4
                elif po3_phase == 2:  # 操控阶段
                    po3_score = 0.3
                elif po3_phase == 1:  # 积累阶段
                    po3_score = 0.2
                
                # RSI贡献分数
                if rsi < 30:
                    rsi_score = 0.3  # 超卖
                elif rsi > 70:
                    rsi_score = -0.3  # 超买
                elif 45 <= rsi <= 55:
                    rsi_score = 0.1  # 中性偏好
                
                # 结构贡献分数
                if current_data.get('bos_bullish', 0):
                    structure_score = 0.4
                elif current_data.get('bos_bearish', 0):
                    structure_score = -0.4
                elif current_data.get('bullish_order_block', 0):
                    structure_score = 0.3
                elif current_data.get('bearish_order_block', 0):
                    structure_score = -0.3
                
                # 综合信号得分计算
                x, y, z = 0.4, 0.3, 0.3  # 权重系数
                signal_score = x * po3_score + y * rsi_score + z * structure_score
                
                # 方向性调整
                if signal_score > 0:
                    bullish_score += signal_score
                else:
                    bearish_score += abs(signal_score)
                
                # ===== 信号质量评分 =====
                # 基于信号数量和强度计算整体质量
                signal_count = 0
                signal_strength_sum = 0
                
                # 统计有效信号
                if current_data.get('bos_bullish', 0) or current_data.get('bos_bearish', 0):
                    signal_count += 1
                    signal_strength_sum += current_data.get('bos_strength', 0)
                
                if current_data.get('bullish_order_block', 0) or current_data.get('bearish_order_block', 0):
                    signal_count += 1
                    signal_strength_sum += current_data.get('order_block_strength', 0)
                
                if abs(smc_signal) > 0.1:
                    signal_count += 1
                    signal_strength_sum += abs(smc_signal)
                
                if rsi < 30 or rsi > 70:
                    signal_count += 1
                    signal_strength_sum += abs(rsi - 50) / 50
                
                # 综合质量评分
                if signal_count > 0:
                    quality_score = (signal_strength_sum / signal_count) * min(signal_count / 3, 1.0)
                    df.loc[df.index[i], 'signal_quality_score'] = quality_score
                
                # 记录组合信号得分
                df.loc[df.index[i], 'combined_bullish_score'] = min(bullish_score, 1.0)
                df.loc[df.index[i], 'combined_bearish_score'] = min(bearish_score, 1.0)
            
            # ✅ 计算最终组合信号
            df['combined_signal'] = df['combined_bullish_score'] - df['combined_bearish_score']
            
            # ✅ 增强原始SMC信号（如果组合信号更强）
            df['enhanced_smc_signal'] = df['smc_signal'].copy()
            
            # 当组合信号更强且质量评分较高时，增强原始信号
            strong_combined_mask = (abs(df['combined_signal']) > abs(df['smc_signal'])) & (df['signal_quality_score'] > 0.3)
            df.loc[strong_combined_mask, 'enhanced_smc_signal'] = df.loc[strong_combined_mask, 'combined_signal']
            
            # ✅ 信号汇聚度升级版
            df['signal_confluence_enhanced'] = 0.0
            for i in range(len(df)):
                signals = [
                    df['smc_signal'].iloc[i],
                    df['combined_signal'].iloc[i]
                ]
                
                # 添加技术指标信号
                if 'ema_cross_signal' in df.columns:
                    signals.append(df['ema_cross_signal'].iloc[i] * 0.3)
                
                if 'rsi' in df.columns:
                    rsi_signal = 0
                    rsi_val = df['rsi'].iloc[i]
                    if rsi_val < 30:
                        rsi_signal = 0.5
                    elif rsi_val > 70:
                        rsi_signal = -0.5
                    signals.append(rsi_signal)
                
                # ✅ 新增：MACD信号
                if 'macd' in df.columns:
                    macd_val = df['macd'].iloc[i]
                    macd_signal = 0.3 if macd_val > 0 else -0.3
                    signals.append(macd_signal)
                
                # ✅ 新增：布林带位置信号
                if 'bb_position' in df.columns:
                    bb_pos = df['bb_position'].iloc[i]
                    if bb_pos < 0.2:
                        bb_signal = 0.4  # 超卖区域
                    elif bb_pos > 0.8:
                        bb_signal = -0.4  # 超买区域
                    else:
                        bb_signal = 0
                    signals.append(bb_signal)
                
                # 计算信号一致性
                if signals:
                    # 同向信号比例
                    positive_signals = sum(1 for s in signals if s > 0.1)
                    negative_signals = sum(1 for s in signals if s < -0.1)
                    total_signals = len([s for s in signals if abs(s) > 0.1])
                    
                    if total_signals > 0:
                        confluence = max(positive_signals, negative_signals) / total_signals
                        df.loc[df.index[i], 'signal_confluence_enhanced'] = confluence
            
            # ✅ 统计信息
            enhanced_signals = df[abs(df['enhanced_smc_signal']) > abs(df['smc_signal'])].shape[0]
            avg_quality = df['signal_quality_score'].mean()
            avg_confluence = df['signal_confluence_enhanced'].mean()
            
            self.logger.info(f"✅ 组合信号识别完成！")
            self.logger.info(f"📊 增强信号数量: {enhanced_signals}, 平均质量: {avg_quality:.3f}, 平均汇聚度: {avg_confluence:.3f}")
            self.logger.info(f"🆕 新增特征: enhanced_smc_signal, combined_signal, signal_quality_score, signal_confluence_enhanced")
            
        except Exception as e:
            self.logger.error(f"组合信号计算失败: {e}")
        
        return df
    
    def _add_enhanced_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 增强版组合信号计算
        集成新增的技术指标：ADX、Stochastic、MFI、K线形态等
        实现更精确的多指标共振确认
        """
        try:
            # 初始化增强版组合信号列
            df['enhanced_bullish_score'] = 0.0
            df['enhanced_bearish_score'] = 0.0
            df['enhanced_signal_quality'] = 0.0
            df['technical_indicator_confirmation'] = 0.0
            
            for i in range(len(df)):
                try:
                    current_data = df.iloc[i]
                    
                    # ===== 收集基础SMC信号 =====
                    smc_signal = current_data.get('smc_signal', 0)
                    bos_bullish = current_data.get('bos_bullish', 0)
                    bos_bearish = current_data.get('bos_bearish', 0)
                    po3_phase = current_data.get('po3_phase', 0)
                    order_block_strength = current_data.get('order_block_strength', 0)
                    
                    enhanced_bullish_score = 0.0
                    enhanced_bearish_score = 0.0
                    
                    # ===== 基础SMC评分 =====
                    if smc_signal > 0:
                        enhanced_bullish_score += abs(smc_signal) * 0.4
                    elif smc_signal < 0:
                        enhanced_bearish_score += abs(smc_signal) * 0.4
                    
                    if bos_bullish:
                        enhanced_bullish_score += 0.3
                    if bos_bearish:
                        enhanced_bearish_score += 0.3
                    
                    # PO3阶段评分
                    if po3_phase == 3:  # 分配阶段
                        enhanced_bullish_score += 0.2
                    elif po3_phase == 1:  # 积累阶段可能反弹
                        enhanced_bearish_score += 0.15
                    
                    # ===== ✅ 新增：技术指标确认评分 =====
                    tech_confirmation_score = 0.0
                    
                    # 1. 趋势确认 (EMA + ADX + MACD)
                    ema_cross = current_data.get('ema_cross_signal', 0)
                    adx = current_data.get('adx', 25)
                    adx_trend_strength = current_data.get('trend_strength', 0)
                    macd = current_data.get('macd', 0)
                    
                    # ADX趋势强度权重
                    adx_weight = 1.0
                    if adx > 40:  # 强趋势
                        adx_weight = 1.5
                    elif adx < 20:  # 无趋势
                        adx_weight = 0.5
                    
                    # 趋势指标对齐评分
                    if ema_cross > 0 and macd > 0:
                        tech_confirmation_score += 0.3 * adx_weight
                        enhanced_bullish_score += 0.25 * adx_weight
                    elif ema_cross < 0 and macd < 0:
                        tech_confirmation_score += 0.3 * adx_weight
                        enhanced_bearish_score += 0.25 * adx_weight
                    
                    # 2. 动量确认 (RSI + Stochastic + MFI)
                    rsi = current_data.get('rsi', 50)
                    stoch_k = current_data.get('stoch_k', 50)
                    stoch_d = current_data.get('stoch_d', 50)
                    mfi = current_data.get('mfi', 50)
                    
                    # 动量超买超卖评分
                    momentum_signals = []
                    
                    # RSI信号
                    if rsi < 30:
                        momentum_signals.append('RSI超卖')
                        enhanced_bullish_score += 0.2
                    elif rsi > 70:
                        momentum_signals.append('RSI超买')
                        enhanced_bearish_score += 0.2
                    
                    # Stochastic信号
                    if stoch_k < 20 and stoch_k > stoch_d:
                        momentum_signals.append('Stoch超卖金叉')
                        enhanced_bullish_score += 0.25
                    elif stoch_k > 80 and stoch_k < stoch_d:
                        momentum_signals.append('Stoch超买死叉')
                        enhanced_bearish_score += 0.25
                    
                    # MFI资金流确认
                    if mfi < 20:
                        momentum_signals.append('MFI资金流超卖')
                        enhanced_bullish_score += 0.15
                    elif mfi > 80:
                        momentum_signals.append('MFI资金流超买')
                        enhanced_bearish_score += 0.15
                    
                    # 动量共振奖励
                    if len(momentum_signals) >= 2:
                        tech_confirmation_score += 0.3
                        self.logger.debug(f"动量共振确认 @{i}: {momentum_signals}")
                    
                    # 3. 波动率确认 (布林带 + ATR)
                    bb_position = current_data.get('bb_position', 0.5)
                    bb_squeeze = current_data.get('bb_squeeze', 0)
                    bb_expansion = current_data.get('bb_expansion', 0)
                    atr_normalized = current_data.get('atr_normalized', 0.02)
                    
                    # 布林带位置确认
                    if bb_position < 0.1:  # 极度超卖
                        enhanced_bullish_score += 0.3
                        tech_confirmation_score += 0.2
                    elif bb_position > 0.9:  # 极度超买
                        enhanced_bearish_score += 0.3
                        tech_confirmation_score += 0.2
                    elif bb_position < 0.2:  # 超卖
                        enhanced_bullish_score += 0.2
                    elif bb_position > 0.8:  # 超买
                        enhanced_bearish_score += 0.2
                    
                    # 布林带压缩后扩张 - 突破信号
                    if bb_expansion and atr_normalized > 0.03:
                        tech_confirmation_score += 0.25
                        # 根据突破方向确定信号
                        if current_data.get('close', 0) > current_data.get('bb_middle', 0):
                            enhanced_bullish_score += 0.2
                        else:
                            enhanced_bearish_score += 0.2
                    
                    # 4. 成交量确认
                    volume_ratio = current_data.get('volume_ratio', 1.0)
                    volume_sma_ratio = current_data.get('volume_sma_ratio', 1.0)
                    
                    volume_confirmation = 0.0
                    if volume_ratio > 2.0:  # 成交量激增
                        volume_confirmation = 0.3
                    elif volume_ratio > 1.5:  # 成交量放大
                        volume_confirmation = 0.2
                    elif volume_sma_ratio > 1.3:  # 短期成交量活跃
                        volume_confirmation = 0.1
                    
                    tech_confirmation_score += volume_confirmation
                    enhanced_bullish_score += volume_confirmation
                    enhanced_bearish_score += volume_confirmation
                    
                    # 5. ✅ 新增：K线形态确认
                    candle_pattern = current_data.get('candle_pattern', 0)
                    hammer_pattern = current_data.get('hammer_pattern', 0)
                    shooting_star_pattern = current_data.get('shooting_star_pattern', 0)
                    doji_pattern = current_data.get('doji_pattern', 0)
                    
                    if hammer_pattern and bb_position < 0.3:
                        enhanced_bullish_score += 0.25
                        tech_confirmation_score += 0.15
                        self.logger.debug(f"锤子线形态确认看涨 @{i}")
                    
                    if shooting_star_pattern and bb_position > 0.7:
                        enhanced_bearish_score += 0.25
                        tech_confirmation_score += 0.15
                        self.logger.debug(f"流星线形态确认看跌 @{i}")
                    
                    if doji_pattern:
                        # 十字星在关键位置的意义
                        if bb_position < 0.2 or bb_position > 0.8:
                            tech_confirmation_score += 0.1
                    
                    # ===== 综合信号质量评分 =====
                    # 基于信号数量和技术确认强度
                    base_quality = abs(smc_signal) * 0.4
                    tech_quality = min(tech_confirmation_score, 0.6) 
                    volume_quality = min(volume_confirmation * 2, 0.3)
                    
                    # 信号一致性奖励
                    signal_consistency = 0.0
                    if enhanced_bullish_score > enhanced_bearish_score and enhanced_bullish_score > 0.5:
                        signal_consistency = 0.2
                    elif enhanced_bearish_score > enhanced_bullish_score and enhanced_bearish_score > 0.5:
                        signal_consistency = 0.2
                    
                    enhanced_signal_quality = base_quality + tech_quality + volume_quality + signal_consistency
                    enhanced_signal_quality = min(enhanced_signal_quality, 1.0)
                    
                    # ===== 记录结果 =====
                    df.loc[df.index[i], 'enhanced_bullish_score'] = min(enhanced_bullish_score, 1.0)
                    df.loc[df.index[i], 'enhanced_bearish_score'] = min(enhanced_bearish_score, 1.0)
                    df.loc[df.index[i], 'enhanced_signal_quality'] = enhanced_signal_quality
                    df.loc[df.index[i], 'technical_indicator_confirmation'] = min(tech_confirmation_score, 1.0)
                    
                except Exception as e:
                    self.logger.error(f"计算增强组合信号失败 @{i}: {e}")
                    continue
            
            # ===== 生成最终增强信号 =====
            df['enhanced_combined_signal'] = df['enhanced_bullish_score'] - df['enhanced_bearish_score']
            
            # ✅ 增强原始SMC信号
            df['enhanced_smc_signal'] = df['smc_signal'].copy()
            
            # 当技术指标确认度高且组合信号更强时，使用组合信号
            enhancement_mask = (
                (df['technical_indicator_confirmation'] > 0.5) & 
                (df['enhanced_signal_quality'] > 0.6) &
                (abs(df['enhanced_combined_signal']) > abs(df['smc_signal']))
            )
            
            df.loc[enhancement_mask, 'enhanced_smc_signal'] = df.loc[enhancement_mask, 'enhanced_combined_signal']
            
            # ===== 统计信息 =====
            enhanced_count = enhancement_mask.sum()
            avg_tech_confirmation = df['technical_indicator_confirmation'].mean()
            avg_enhanced_quality = df['enhanced_signal_quality'].mean()
            
            self.logger.info(f"✅ 增强组合信号计算完成")
            self.logger.info(f"📊 信号增强数量: {enhanced_count}, 平均技术确认度: {avg_tech_confirmation:.3f}")
            self.logger.info(f"📊 平均信号质量: {avg_enhanced_quality:.3f}")
            self.logger.info(f"🆕 新增特征: enhanced_smc_signal, enhanced_signal_quality, technical_indicator_confirmation")
            
        except Exception as e:
            self.logger.error(f"增强组合信号计算失败: {e}")
        
        return df
    
    def get_smc_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取SMC信号摘要"""
        summary = {
            'po3_phases': {
                'accumulation_count': df['po3_accumulation'].sum(),
                'manipulation_count': df['po3_manipulation'].sum(),
                'distribution_count': df['po3_distribution'].sum()
            },
            'bos_signals': {
                'bullish_count': df['bos_bullish'].sum(),
                'bearish_count': df['bos_bearish'].sum(),
                'avg_strength': df['bos_strength'][df['bos_strength'] > 0].mean() if df['bos_strength'].sum() > 0 else 0
            },
            'order_blocks': {
                'bullish_count': df['bullish_order_block'].sum(),
                'bearish_count': df['bearish_order_block'].sum(),
                'avg_strength': df['order_block_strength'][df['order_block_strength'] > 0].mean() if df['order_block_strength'].sum() > 0 else 0
            },
            'liquidity_sweeps': {
                'high_sweeps': df['liquidity_sweep_high'].sum(),
                'low_sweeps': df['liquidity_sweep_low'].sum(),
                'avg_strength': df['liquidity_sweep_strength'][df['liquidity_sweep_strength'] > 0].mean() if df['liquidity_sweep_strength'].sum() > 0 else 0
            },
            'fvg_signals': {
                'bullish_count': df['fvg_bullish'].sum(),
                'bearish_count': df['fvg_bearish'].sum(),
                'avg_size': df['fvg_size'][df['fvg_size'] > 0].mean() if df['fvg_size'].sum() > 0 else 0
            },
            'market_structure': {
                'bullish_periods': (df['market_structure'] == 1).sum(),
                'bearish_periods': (df['market_structure'] == -1).sum(),
                'ranging_periods': (df['market_structure'] == 0).sum()
            }
        }
        
        return summary

def main():
    """主函数，用于测试SMC信号计算"""
    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators
    
    # 加载数据
    collector = DataCollector()
    df = collector.load_data()
    
    if df.empty:
        print("请先运行数据收集器获取数据")
        return
    
    # 计算技术指标（SMC信号依赖于基础指标）
    indicator_calculator = TechnicalIndicators()
    df_with_indicators = indicator_calculator.calculate_all_indicators(df)
    
    # 计算SMC信号
    smc_calculator = SMCSignals()
    df_with_smc = smc_calculator.calculate_all_smc_signals(df_with_indicators)
    
    # 显示结果
    print(f"添加SMC信号前列数: {len(df_with_indicators.columns)}")
    print(f"添加SMC信号后列数: {len(df_with_smc.columns)}")
    print(f"新增SMC信号数量: {len(df_with_smc.columns) - len(df_with_indicators.columns)}")
    
    # 显示SMC信号摘要
    summary = smc_calculator.get_smc_summary(df_with_smc)
    print(f"\nSMC信号摘要:")
    for category, stats in summary.items():
        print(f"{category}: {stats}")

if __name__ == "__main__":
    main() 