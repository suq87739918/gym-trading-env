"""
技术指标计算模块 - 优化版
包含精选的核心技术指标，删除冗余和功能重复的指标
"""
import numpy as np
import pandas as pd
import ta
from typing import Tuple, Dict, Any, List
from utils.config import get_config
from utils.logger import get_logger

# 尝试导入talib，如果失败则使用ta库的替代实现
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ 警告: talib 未安装，将使用 ta 库作为替代")

class TechnicalIndicators:
    """技术指标计算器 - 优化版"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('TechnicalIndicators', 'indicators.log')
        
        # ✅ 增强的核心指标类别 - 新增ADX等
        self.core_indicators = {
            'trend': ['ema_fast', 'ema_slow', 'price_vs_ema_fast', 'ema_cross_signal', 'adx', 'trend_strength'],
            'momentum': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d'],
            'volatility': ['atr_normalized', 'bb_position', 'bb_width', 'bb_squeeze'],
            'volume': ['volume_ratio', 'obv_normalized', 'price_vs_vwap', 'volume_sma_ratio', 'mfi'],
            'price_action': ['price_change_5', 'price_change_10', 'close_position_in_range', 'candle_pattern'],
            'signal_filter': ['signal_confluence', 'trend_alignment', 'momentum_alignment', 'volatility_regime']
        }
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 计算增强的技术指标系统 - 为信号过滤优化
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了增强技术指标的DataFrame
        """
        try:
            df_with_indicators = df.copy()
            
            # 1. 趋势类指标（EMA系统 + ADX）
            df_with_indicators = self._add_enhanced_trend_indicators(df_with_indicators)
            
            # 2. 动量类指标（RSI + MACD + Stochastic）
            df_with_indicators = self._add_enhanced_momentum_indicators(df_with_indicators)
            
            # 3. 波动率指标（ATR + 布林带 + 压缩检测）
            df_with_indicators = self._add_enhanced_volatility_indicators(df_with_indicators)
            
            # 4. 成交量指标（增强版）
            df_with_indicators = self._add_enhanced_volume_indicators(df_with_indicators)
            
            # 5. 价格行为指标（增强版）
            df_with_indicators = self._add_enhanced_price_action_indicators(df_with_indicators)
            
            # 6. ✅ 新增：信号过滤指标
            df_with_indicators = self._add_signal_filter_indicators(df_with_indicators)
            
            # 添加归一化版本
            df_with_indicators = self._add_normalized_features(df_with_indicators)
            
            self.logger.info(f"✅ 增强技术指标计算完成，共 {len(df_with_indicators.columns) - len(df.columns)} 个新特征")
            return df_with_indicators
            
        except Exception as e:
            self.logger.exception(f"❌ 计算增强技术指标时发生错误: {e}")
            return df
    
    def _add_enhanced_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 添加增强趋势类指标 - 新增ADX"""
        try:
            fast_period = self.config.get('EMA_FAST', 20)
            slow_period = self.config.get('EMA_SLOW', 50)
            adx_period = self.config.get('ADX_PERIOD', 14)
            
            # 计算EMA
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=fast_period).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=slow_period).ema_indicator()
            
            # 价格相对EMA位置（归一化）
            df['price_vs_ema_fast'] = (df['close'] - df['ema_fast']) / df['ema_fast']
            df['price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['ema_slow']
            
            # EMA交叉信号（简化版）
            df['ema_cross_signal'] = 0
            df.loc[df['ema_fast'] > df['ema_slow'], 'ema_cross_signal'] = 1  # 金叉
            df.loc[df['ema_fast'] < df['ema_slow'], 'ema_cross_signal'] = -1  # 死叉
            
            # ✅ 新增：ADX - 平均趋向指数
            try:
                adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=adx_period)
                df['adx'] = adx_indicator.adx()
                df['di_plus'] = adx_indicator.adx_pos()
                df['di_minus'] = adx_indicator.adx_neg()
                
                # ADX归一化 (0-100 -> 0-1)
                df['adx_normalized'] = df['adx'] / 100.0
                
                # 趋势强度分类
                df['trend_strength'] = 0  # 0: 无趋势, 1: 弱趋势, 2: 强趋势
                df.loc[df['adx'] >= 20, 'trend_strength'] = 1  # 弱趋势
                df.loc[df['adx'] >= 40, 'trend_strength'] = 2  # 强趋势
                
                self.logger.debug("✅ ADX指标计算完成")
                
            except Exception as e:
                self.logger.warning(f"ADX计算失败，使用默认值: {e}")
                df['adx'] = 25.0
                df['di_plus'] = 0.0
                df['di_minus'] = 0.0
                df['adx_normalized'] = 0.25
                df['trend_strength'] = 1
            
            self.logger.debug("✅ 增强趋势指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算增强趋势指标失败: {e}")
            
        return df
    
    def _add_enhanced_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 添加增强动量类指标 - 新增Stochastic"""
        try:
            # RSI（主要动量指标）
            rsi_period = self.config.get('RSI_PERIOD', 14)
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
            
            # MACD（趋势动量指标）
            macd_indicator = ta.trend.MACD(df['close'])
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_histogram'] = macd_indicator.macd_diff()
            
            # MACD归一化
            df['macd_normalized'] = df['macd'] / df['close']
            
            # ✅ 新增：Stochastic指标
            try:
                stoch_period = self.config.get('STOCH_PERIOD', 14)
                stoch_indicator = ta.momentum.StochasticOscillator(
                    df['high'], df['low'], df['close'], 
                    window=stoch_period, smooth_window=3
                )
                df['stoch_k'] = stoch_indicator.stoch()
                df['stoch_d'] = stoch_indicator.stoch_signal()
                
                # Stochastic交叉信号
                df['stoch_cross_signal'] = 0
                df.loc[(df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 20), 'stoch_cross_signal'] = 1  # 超卖区金叉
                df.loc[(df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > 80), 'stoch_cross_signal'] = -1  # 超买区死叉
                
                self.logger.debug("✅ Stochastic指标计算完成")
                
            except Exception as e:
                self.logger.warning(f"Stochastic计算失败，使用默认值: {e}")
                df['stoch_k'] = 50.0
                df['stoch_d'] = 50.0
                df['stoch_cross_signal'] = 0
            
            self.logger.debug("✅ 增强动量指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算增强动量指标失败: {e}")
            
        return df
    
    def _add_enhanced_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 添加增强波动率指标 - 新增布林带压缩检测"""
        try:
            # ATR（波动率指标）
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['atr_normalized'] = df['atr'] / df['close']
            
            # 布林带（波动率和超买超卖）
            bb_period = self.config.get('BB_PERIOD', 20)
            bb_std = self.config.get('BB_STD', 2.0)
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
            
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # 布林带关键特征
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_position'] = df['bb_position'].fillna(0.5)
            
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ✅ 新增：布林带压缩检测（突破前兆）
            try:
                # 计算布林带宽度的滚动平均
                bb_width_ma = df['bb_width'].rolling(window=20).mean()
                
                # 压缩检测：当前宽度显著小于历史平均
                df['bb_squeeze'] = 0
                squeeze_threshold = 0.8  # 压缩阈值
                df.loc[df['bb_width'] < bb_width_ma * squeeze_threshold, 'bb_squeeze'] = 1
                
                # 压缩后的扩张信号（突破信号）
                df['bb_expansion'] = 0
                expansion_threshold = 1.2  # 扩张阈值
                df.loc[df['bb_width'] > bb_width_ma * expansion_threshold, 'bb_expansion'] = 1
                
                self.logger.debug("✅ 布林带压缩检测完成")
                
            except Exception as e:
                self.logger.warning(f"布林带压缩检测失败: {e}")
                df['bb_squeeze'] = 0
                df['bb_expansion'] = 0
            
            self.logger.debug("✅ 增强波动率指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算增强波动率指标失败: {e}")
            
        return df
    
    def _add_enhanced_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 添加增强成交量指标 - 新增MFI等"""
        try:
            volume_ma_period = self.config.get('VOLUME_MA_PERIOD', 20)
            
            # 相对成交量（最重要的成交量指标）
            df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # ✅ 新增：成交量SMA比率（多周期确认）
            volume_sma_5 = df['volume'].rolling(window=5).mean()
            volume_sma_20 = df['volume'].rolling(window=20).mean()
            df['volume_sma_ratio'] = volume_sma_5 / volume_sma_20
            df['volume_sma_ratio'] = df['volume_sma_ratio'].fillna(1.0)
            
            # OBV（累积成交量指标）
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            # OBV归一化处理
            obv_max = df['obv'].abs().rolling(window=100).max()
            df['obv_normalized'] = df['obv'] / (obv_max + 1e-8)
            df['obv_normalized'] = df['obv_normalized'].fillna(0.0)
            
            # VWAP相对位置
            df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            df['price_vs_vwap'] = df['price_vs_vwap'].fillna(0.0)
            
            # ✅ 新增：MFI（资金流量指数）
            try:
                mfi_period = self.config.get('MFI_PERIOD', 14)
                df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=mfi_period).money_flow_index()
                df['mfi_normalized'] = df['mfi'] / 100.0
                
                self.logger.debug("✅ MFI指标计算完成")
                
            except Exception as e:
                self.logger.warning(f"MFI计算失败，使用默认值: {e}")
                df['mfi'] = 50.0
                df['mfi_normalized'] = 0.5
            
            self.logger.debug("✅ 增强成交量指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算增强成交量指标失败: {e}")
            
        return df
    
    def _add_enhanced_price_action_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ 添加增强价格行为指标 - 新增K线形态识别"""
        try:
            # 关键价格变化
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # 价格在日内区间的位置
            df['close_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['close_position_in_range'] = df['close_position_in_range'].fillna(0.5)
            
            # ✅ 新增：简单K线形态识别
            try:
                # 计算实体和影线
                df['body_size'] = abs(df['close'] - df['open']) / df['open']
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
                
                # 十字星形态（小实体大影线）
                df['doji_pattern'] = 0
                doji_mask = (df['body_size'] < 0.002) & ((df['upper_shadow'] > 0.003) | (df['lower_shadow'] > 0.003))
                df.loc[doji_mask, 'doji_pattern'] = 1
                
                # 锤子线/倒锤子线形态
                df['hammer_pattern'] = 0
                hammer_mask = (df['lower_shadow'] > df['body_size'] * 2) & (df['upper_shadow'] < df['body_size'] * 0.5)
                df.loc[hammer_mask, 'hammer_pattern'] = 1
                
                # 流星线形态
                df['shooting_star_pattern'] = 0
                star_mask = (df['upper_shadow'] > df['body_size'] * 2) & (df['lower_shadow'] < df['body_size'] * 0.5)
                df.loc[star_mask, 'shooting_star_pattern'] = 1
                
                # 综合K线形态信号
                df['candle_pattern'] = 0
                df.loc[df['doji_pattern'] == 1, 'candle_pattern'] = 0.3  # 十字星：中性
                df.loc[df['hammer_pattern'] == 1, 'candle_pattern'] = 0.7  # 锤子线：看涨
                df.loc[df['shooting_star_pattern'] == 1, 'candle_pattern'] = -0.7  # 流星线：看跌
                
                self.logger.debug("✅ K线形态识别完成")
                
            except Exception as e:
                self.logger.warning(f"K线形态识别失败: {e}")
                df['candle_pattern'] = 0
                df['doji_pattern'] = 0
                df['hammer_pattern'] = 0
                df['shooting_star_pattern'] = 0
            
            self.logger.debug("✅ 增强价格行为指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算增强价格行为指标失败: {e}")
            
        return df
    
    def _add_signal_filter_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加信号过滤相关的指标"""
        try:
            # 获取基础数据
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # 1. 增强的RSI系列
            if TALIB_AVAILABLE:
                df['rsi_14'] = talib.RSI(close, timeperiod=14)
                df['rsi_21'] = talib.RSI(close, timeperiod=21)
            else:
                df['rsi_14'] = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi()
                df['rsi_21'] = ta.momentum.RSIIndicator(pd.Series(close), window=21).rsi()
            
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            
            # 2. 增强的布林带系列
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle  
                df['bb_lower'] = bb_lower
            else:
                bb_indicator = ta.volatility.BollingerBands(pd.Series(close), window=20, window_dev=2)
                df['bb_upper'] = bb_indicator.bollinger_hband()
                df['bb_middle'] = bb_indicator.bollinger_mavg()
                df['bb_lower'] = bb_indicator.bollinger_lband()
            
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 布林带挤压和扩张
            bb_width_ma = pd.Series(df['bb_width']).rolling(20).mean()
            df['bb_squeeze'] = (df['bb_width'] < bb_width_ma * 0.8).astype(int)
            df['bb_expansion'] = (df['bb_width'] > bb_width_ma * 1.2).astype(int)
            
            # 3. ADX增强系列
            if TALIB_AVAILABLE:
                df['adx'] = talib.ADX(high, low, close, timeperiod=14)
                df['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
                df['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            else:
                adx_indicator = ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=14)
                df['adx'] = adx_indicator.adx()
                df['di_plus'] = adx_indicator.adx_pos()
                df['di_minus'] = adx_indicator.adx_neg()
            
            # ADX趋势强度分类
            df['adx_strong_trend'] = (df['adx'] > 40).astype(int)
            df['adx_weak_trend'] = (df['adx'] < 20).astype(int)
            
            # 4. 成交量增强指标
            if TALIB_AVAILABLE:
                df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
                df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            else:
                df['volume_sma_10'] = pd.Series(volume).rolling(10).mean()
                df['volume_sma_20'] = pd.Series(volume).rolling(20).mean()
            
            df['volume_ratio'] = volume / df['volume_sma_20']
            df['volume_sma_ratio'] = df['volume_sma_10'] / df['volume_sma_20']
            
            # 成交量异常检测
            volume_std = pd.Series(volume).rolling(20).std()
            volume_mean = pd.Series(volume).rolling(20).mean()
            df['volume_zscore'] = (volume - volume_mean) / volume_std
            
            # 5. 信号汇聚度计算
            df['signal_confluence'] = self._calculate_signal_confluence(df)
            
            # 6. 市场状态指标
            df['market_volatility'] = df['atr'] / close
            df['market_momentum'] = df['rsi_14'] / 50 - 1  # 标准化动量
            
            # 7. 价格位置指标
            df['price_percentile'] = pd.Series(close).rolling(100).rank(pct=True)
            
            self.logger.debug("✅ 信号过滤指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算信号过滤指标失败: {e}")
        
        return df
    
    def _calculate_signal_confluence(self, df: pd.DataFrame) -> pd.Series:
        """计算信号汇聚度"""
        try:
            confluence_scores = []
            
            for i in range(len(df)):
                score = 0.0
                total_signals = 0
                
                # RSI信号
                if not pd.isna(df['rsi_14'].iloc[i]):
                    if df['rsi_14'].iloc[i] > 70:
                        score += 1  # 超买
                        total_signals += 1
                    elif df['rsi_14'].iloc[i] < 30:
                        score += 1  # 超卖
                        total_signals += 1
                    else:
                        total_signals += 1
                
                # MACD信号
                if not pd.isna(df['macd_histogram'].iloc[i]):
                    if df['macd_histogram'].iloc[i] > 0:
                        score += 0.5
                    total_signals += 1
                
                # 布林带信号
                if not pd.isna(df['bb_position'].iloc[i]):
                    if df['bb_position'].iloc[i] > 0.8 or df['bb_position'].iloc[i] < 0.2:
                        score += 1
                    total_signals += 1
                
                # ADX信号
                if not pd.isna(df['adx'].iloc[i]):
                    if df['adx'].iloc[i] > 25:
                        score += 0.5
                    total_signals += 1
                
                # 成交量信号
                if not pd.isna(df['volume_ratio'].iloc[i]):
                    if df['volume_ratio'].iloc[i] > 1.5:
                        score += 0.5
                    total_signals += 1
                
                # 计算汇聚度
                if total_signals > 0:
                    confluence_scores.append(score / total_signals)
                else:
                    confluence_scores.append(0.0)
            
            return pd.Series(confluence_scores, index=df.index)
            
        except Exception as e:
            self.logger.error(f"计算信号汇聚度失败: {e}")
            return pd.Series(0.0, index=df.index)

    def calculate_core_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算精选的核心技术指标
        删除冗余指标，只保留最有效的指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了核心技术指标的DataFrame
        """
        try:
            df_with_indicators = df.copy()
            
            # 1. 趋势类指标（EMA系统）
            df_with_indicators = self._add_trend_indicators(df_with_indicators)
            
            # 2. 动量类指标（RSI + MACD）
            df_with_indicators = self._add_momentum_indicators(df_with_indicators)
            
            # 3. 波动率指标（ATR + 布林带）
            df_with_indicators = self._add_volatility_indicators(df_with_indicators)
            
            # 4. 成交量指标（精选）
            df_with_indicators = self._add_volume_indicators(df_with_indicators)
            
            # 5. 价格行为指标
            df_with_indicators = self._add_price_action_indicators(df_with_indicators)
            
            # 添加归一化版本
            df_with_indicators = self._add_normalized_features(df_with_indicators)
            
            self.logger.info(f"✅ 核心技术指标计算完成，共 {len(df_with_indicators.columns) - len(df.columns)} 个新特征")
            return df_with_indicators
            
        except Exception as e:
            self.logger.exception(f"❌ 计算核心技术指标时发生错误: {e}")
            return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势类指标 - 只保留EMA系统"""
        try:
            fast_period = self.config.get('EMA_FAST', 20)
            slow_period = self.config.get('EMA_SLOW', 50)
            
            # 计算EMA
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=fast_period).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=slow_period).ema_indicator()
            
            # 价格相对EMA位置（归一化）
            df['price_vs_ema_fast'] = (df['close'] - df['ema_fast']) / df['ema_fast']
            df['price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['ema_slow']
            
            # EMA交叉信号（简化版）
            df['ema_cross_signal'] = 0
            df.loc[df['ema_fast'] > df['ema_slow'], 'ema_cross_signal'] = 1  # 金叉
            df.loc[df['ema_fast'] < df['ema_slow'], 'ema_cross_signal'] = -1  # 死叉
            
            self.logger.debug("✅ 趋势指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算趋势指标失败: {e}")
            
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量类指标 - 只保留RSI和MACD"""
        try:
            # RSI（主要动量指标）
            rsi_period = self.config.get('RSI_PERIOD', 14)
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
            
            # MACD（趋势动量指标）
            macd_indicator = ta.trend.MACD(df['close'])
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_histogram'] = macd_indicator.macd_diff()
            
            # MACD归一化
            df['macd_normalized'] = df['macd'] / df['close']
            
            self.logger.debug("✅ 动量指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算动量指标失败: {e}")
            
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率指标 - 只保留ATR和布林带核心特征"""
        try:
            # ATR（波动率指标）
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['atr_normalized'] = df['atr'] / df['close']
            
            # 布林带（波动率和超买超卖）
            bb_period = self.config.get('BB_PERIOD', 20)
            bb_std = self.config.get('BB_STD', 2.0)
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=bb_std)
            
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # 布林带关键特征
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_position'] = df['bb_position'].fillna(0.5)
            
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            self.logger.debug("✅ 波动率指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算波动率指标失败: {e}")
            
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标 - 精选核心指标"""
        try:
            volume_ma_period = self.config.get('VOLUME_MA_PERIOD', 20)
            
            # 相对成交量（最重要的成交量指标）
            df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # OBV（累积成交量指标）
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            # OBV归一化处理
            obv_max = df['obv'].abs().rolling(window=100).max()
            df['obv_normalized'] = df['obv'] / (obv_max + 1e-8)
            df['obv_normalized'] = df['obv_normalized'].fillna(0.0)
            
            # VWAP相对位置
            df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            df['price_vs_vwap'] = df['price_vs_vwap'].fillna(0.0)
            
            self.logger.debug("✅ 成交量指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算成交量指标失败: {e}")
            
        return df
    
    def _add_price_action_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格行为指标 - 简化版"""
        try:
            # 关键价格变化
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # 价格在日内区间的位置
            df['close_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['close_position_in_range'] = df['close_position_in_range'].fillna(0.5)
            
            self.logger.debug("✅ 价格行为指标计算完成")
            
        except Exception as e:
            self.logger.error(f"❌ 计算价格行为指标失败: {e}")
            
        return df
    
    def _add_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加归一化特征"""
        try:
            # RSI归一化（0-100转换为0-1）
            df['rsi_normalized'] = df['rsi'] / 100.0
            
            # 价格变化裁剪（避免极值）
            df['price_change_5'] = df['price_change_5'].clip(-0.1, 0.1)  # 限制在±10%
            df['price_change_10'] = df['price_change_10'].clip(-0.2, 0.2)  # 限制在±20%
            
            # ATR归一化
            df['atr_normalized'] = df['atr_normalized'].clip(0, 0.1)  # 限制ATR
            
            # 成交量比率裁剪
            df['volume_ratio'] = df['volume_ratio'].clip(0.1, 5.0)  # 限制成交量比率
            
            self.logger.debug("✅ 特征归一化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 特征归一化失败: {e}")
            
        return df
    
    def get_core_feature_list(self) -> List[str]:
        """获取核心特征列表"""
        all_features = []
        for category, features in self.core_indicators.items():
            all_features.extend(features)
        
        # 添加归一化特征
        normalized_features = [
            'rsi_normalized', 'macd_normalized', 'atr_normalized',
            'bb_position', 'bb_width', 'volume_ratio', 'obv_normalized', 
            'price_vs_vwap', 'price_change_5', 'price_change_10', 
            'close_position_in_range', 'price_vs_ema_fast', 'price_vs_ema_slow'
        ]
        
        all_features.extend(normalized_features)
        return list(set(all_features))  # 去重
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
        """
        对特征进行归一化处理
        
        Args:
            df: 数据DataFrame
            feature_columns: 需要归一化的特征列，如果为None则自动选择
            
        Returns:
            归一化后的DataFrame
        """
        if feature_columns is None:
            # 自动选择数值型特征列
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除价格列
            exclude_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        df_normalized = df.copy()
        
        for col in feature_columns:
            if col in df_normalized.columns:
                try:
                    # 使用滚动窗口标准化
                    rolling_mean = df_normalized[col].rolling(window=100, min_periods=1).mean()
                    rolling_std = df_normalized[col].rolling(window=100, min_periods=1).std()
                    
                    df_normalized[f'{col}_normalized'] = ((df_normalized[col] - rolling_mean) / 
                                                        (rolling_std + 1e-8))
                    
                    # 限制极值
                    df_normalized[f'{col}_normalized'] = df_normalized[f'{col}_normalized'].clip(-3, 3)
                    
                except Exception as e:
                    self.logger.warning(f"归一化特征 {col} 失败: {e}")
        
        return df_normalized
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        兼容性方法：调用核心指标计算
        保持向后兼容
        """
        return self.calculate_core_indicators(df)
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取特征摘要信息"""
        core_features = self.get_core_feature_list()
        available_features = [f for f in core_features if f in df.columns]
        
        summary = {
            'total_core_features': len(core_features),
            'available_features': len(available_features),
            'missing_features': len(core_features) - len(available_features),
            'feature_categories': {
                category: [f for f in features if f in df.columns]
                for category, features in self.core_indicators.items()
            },
            'total_rows': len(df),
            'missing_values': {
                feature: df[feature].isnull().sum()
                for feature in available_features
            },
            'feature_stats': {
                feature: {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }
                for feature in available_features if df[feature].dtype in ['float64', 'int64']
            }
        }
        
        return summary
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证核心特征的质量"""
        core_features = self.get_core_feature_list()
        validation_results = {
            'valid_features': [],
            'invalid_features': [],
            'missing_features': [],
            'quality_issues': []
        }
        
        for feature in core_features:
            if feature not in df.columns:
                validation_results['missing_features'].append(feature)
                continue
            
            feature_data = df[feature]
            
            # 检查是否有有效数据
            if feature_data.isnull().all():
                validation_results['invalid_features'].append({
                    'feature': feature,
                    'issue': 'all_null'
                })
                continue
            
            # 检查异常值
            if feature_data.dtype in ['float64', 'int64']:
                if np.isinf(feature_data).any():
                    validation_results['quality_issues'].append({
                        'feature': feature,
                        'issue': 'infinite_values',
                        'count': np.isinf(feature_data).sum()
                    })
                
                # 检查极值
                q1, q3 = feature_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = feature_data[(feature_data < q1 - 3*iqr) | (feature_data > q3 + 3*iqr)]
                if len(outliers) > len(feature_data) * 0.05:  # 超过5%的异常值
                    validation_results['quality_issues'].append({
                        'feature': feature,
                        'issue': 'too_many_outliers',
                        'outlier_ratio': len(outliers) / len(feature_data)
                    })
            
            validation_results['valid_features'].append(feature)
        
        return validation_results

    def get_enhanced_feature_list(self) -> List[str]:
        """获取增强特征列表（用于特征工程集成）"""
        enhanced_features = [
            # 基础OHLCV
            'open', 'high', 'low', 'close', 'volume',
            
            # 趋势指标
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ema_fast', 'ema_slow',
            'ema_cross_signal', 'trend_strength',
            
            # 动量指标  
            'rsi', 'rsi_14', 'rsi_21', 'rsi_overbought', 'rsi_oversold',
            'macd', 'macd_signal', 'macd_hist', 'macd_cross_signal',
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
            
            # 波动率指标
            'atr', 'atr_normalized', 'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position', 'bb_squeeze', 'bb_expansion',
            'adx', 'di_plus', 'di_minus', 'adx_strong_trend', 'adx_weak_trend',
            
            # 成交量指标
            'volume_sma', 'volume_ema', 'volume_ratio', 'volume_sma_10',
            'volume_sma_20', 'volume_sma_ratio', 'volume_zscore', 'obv',
            
            # 价格行为指标
            'price_change', 'volume_change', 'high_low_ratio', 'close_position',
            'typical_price', 'price_percentile',
            
            # 综合指标
            'signal_confluence', 'market_volatility', 'market_momentum',
            'trend_consistency'
        ]
        
        return enhanced_features
    
    def validate_features_for_ml(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证特征是否适合机器学习（增强版）"""
        try:
            validation_report = {
                'total_features': 0,
                'valid_features': 0,
                'problematic_features': [],
                'feature_quality_scores': {},
                'recommendations': []
            }
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            validation_report['total_features'] = len(numeric_columns)
            
            for col in numeric_columns:
                feature_issues = []
                quality_score = 1.0
                
                # 检查缺失值
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.5:
                    feature_issues.append(f"高缺失率: {missing_pct:.2%}")
                    quality_score -= 0.5
                elif missing_pct > 0.1:
                    feature_issues.append(f"中等缺失率: {missing_pct:.2%}")
                    quality_score -= 0.2
                
                # 检查常数特征
                if df[col].nunique() <= 1:
                    feature_issues.append("常数特征")
                    quality_score = 0
                
                # 检查无穷值
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    feature_issues.append(f"包含{inf_count}个无穷值")
                    quality_score -= 0.3
                
                # 检查方差
                if df[col].var() < 1e-10:
                    feature_issues.append("极低方差")
                    quality_score -= 0.4
                
                # 检查分布偏度
                try:
                    from scipy.stats import skew
                    feature_skew = abs(skew(df[col].dropna()))
                    if feature_skew > 10:
                        feature_issues.append(f"极端偏度: {feature_skew:.2f}")
                        quality_score -= 0.2
                except:
                    pass
                
                # 记录结果
                validation_report['feature_quality_scores'][col] = quality_score
                
                if len(feature_issues) > 0:
                    validation_report['problematic_features'].append({
                        'feature': col,
                        'issues': feature_issues,
                        'quality_score': quality_score
                    })
                
                if quality_score > 0.7:
                    validation_report['valid_features'] += 1
            
            # 生成建议
            problematic_count = len(validation_report['problematic_features'])
            if problematic_count > validation_report['total_features'] * 0.3:
                validation_report['recommendations'].append("问题特征过多，建议重新设计特征工程")
            
            if validation_report['valid_features'] < 10:
                validation_report['recommendations'].append("有效特征过少，可能影响模型性能")
            
            high_quality_features = [f for f, score in validation_report['feature_quality_scores'].items() 
                                   if score > 0.8]
            validation_report['recommendations'].append(f"推荐使用{len(high_quality_features)}个高质量特征")
            
            return validation_report
            
        except Exception as e:
            self.logger.exception(f"❌ 特征验证失败: {e}")
            return {'error': str(e)}

def main():
    """主函数，用于测试技术指标计算"""
    from data_collector import DataCollector
    
    # 加载数据
    collector = DataCollector()
    df = collector.load_data()
    
    if df.empty:
        print("请先运行数据收集器获取数据")
        return
    
    # 计算技术指标
    indicator_calculator = TechnicalIndicators()
    df_with_indicators = indicator_calculator.calculate_core_indicators(df)
    
    # 显示结果
    print(f"原始数据列数: {len(df.columns)}")
    print(f"添加指标后列数: {len(df_with_indicators.columns)}")
    print(f"新增指标数量: {len(df_with_indicators.columns) - len(df.columns)}")
    
    # 显示部分指标
    indicator_columns = [col for col in df_with_indicators.columns if col not in df.columns]
    print(f"\n新增的技术指标: {indicator_columns[:10]}...")
    
    # 获取特征摘要
    summary = indicator_calculator.get_feature_summary(df_with_indicators)
    print(f"\n特征摘要:")
    print(f"总特征数: {summary['total_core_features']}")
    print(f"可用特征数: {summary['available_features']}")
    print(f"缺失特征数: {summary['missing_features']}")
    print(f"特征类别: {summary['feature_categories']}")
    print(f"数据行数: {summary['total_rows']}")
    print(f"缺失值: {summary['missing_values']}")
    print(f"特征统计: {summary['feature_stats']}")

if __name__ == "__main__":
    main() 