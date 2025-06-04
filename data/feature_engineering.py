"""
特征工程模块 - 增强版数据输入和特征构造系统
实现多时间尺度特征、特征重要性分析、特征选择、降维处理等功能
解决31个特征的优化问题，提升模型效果
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import talib
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from utils.config import get_config
from utils.logger import get_logger

class EnhancedFeatureEngineer:
    """增强版特征工程器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('FeatureEngineer', 'feature_engineer.log')
        
        # 特征配置
        self.feature_config = {
            'enable_multi_timeframe': True,
            'timeframes': ['1h', '4h', '1d'],  # 多时间框架
            'enable_volume_features': True,
            'enable_cyclical_features': True,
            'enable_volatility_regime': True,
            'max_features': 50,  # 最大特征数限制
            'correlation_threshold': 0.9,  # 相关性阈值
            'importance_threshold': 0.01,  # 重要性阈值
        }
        
        # 初始化scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # 特征重要性存储
        self.feature_importance = {}
        self.feature_correlations = None
        
    def engineer_comprehensive_features(self, df: pd.DataFrame, symbol: str = 'SOLUSDT') -> pd.DataFrame:
        """
        ✅ 全面的特征工程 - 主要入口
        补充关键特征并优化现有特征集
        """
        try:
            self.logger.info("🔧 开始全面特征工程...")
            df_enhanced = df.copy()
            
            # 1. 补充基础技术指标
            df_enhanced = self._add_missing_technical_indicators(df_enhanced)
            
            # 2. 增强成交量特征
            df_enhanced = self._add_enhanced_volume_features(df_enhanced)
            
            # 3. 多时间尺度特征
            if self.feature_config['enable_multi_timeframe']:
                df_enhanced = self._add_multi_timeframe_features(df_enhanced)
            
            # 4. 时间周期特征
            if self.feature_config['enable_cyclical_features']:
                df_enhanced = self._add_cyclical_features(df_enhanced)
            
            # 5. 波动率制度特征
            if self.feature_config['enable_volatility_regime']:
                df_enhanced = self._add_volatility_regime_features(df_enhanced)
            
            # 6. 市场微结构特征
            df_enhanced = self._add_microstructure_features(df_enhanced)
            
            # 7. 特征标准化
            df_enhanced = self._normalize_features(df_enhanced)
            
            self.logger.info(f"✅ 特征工程完成，总特征数: {len(df_enhanced.columns)}")
            return df_enhanced
            
        except Exception as e:
            self.logger.exception(f"❌ 特征工程失败: {e}")
            return df
    
    def _add_missing_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """补充缺失的技术指标"""
        try:
            # 价格数据
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # 1. ATR系列（不同周期）
            df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['atr_21'] = talib.ATR(high, low, close, timeperiod=21)
            df['atr_ratio'] = df['atr_14'] / df['atr_21']  # ATR比率
            
            # 2. ADX趋势强度系列
            df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            df['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            df['dx'] = talib.DX(high, low, close, timeperiod=14)
            
            # 3. 布林带完整系列
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # 带宽
            df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # %B位置
            
            # 4. 随机振荡器
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close, 
                                                     fastk_period=14, slowk_period=3, slowd_period=3)
            
            # 5. 威廉指标
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # 6. 商品通道指数
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            
            # 7. 抛物线转向
            df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # 8. 真实强度指数
            df['trix'] = talib.TRIX(close, timeperiod=14)
            
            # 9. 资金流向指标
            df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # 10. 终极振荡器
            df['ultosc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            self.logger.debug("✅ 技术指标补充完成")
            
        except Exception as e:
            self.logger.error(f"❌ 补充技术指标失败: {e}")
        
        return df
    
    def _add_enhanced_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强成交量特征"""
        try:
            volume = df['volume'].values
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 1. 成交量移动平均
            df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            df['volume_sma_50'] = talib.SMA(volume, timeperiod=50)
            
            # 2. 成交量比率
            df['volume_ratio_10'] = volume / df['volume_sma_10']
            df['volume_ratio_20'] = volume / df['volume_sma_20']
            
            # 3. 成交量异常检测
            volume_std = pd.Series(volume).rolling(20).std()
            volume_mean = pd.Series(volume).rolling(20).mean()
            df['volume_zscore'] = (volume - volume_mean) / volume_std
            df['volume_spike'] = (df['volume_zscore'] > 2).astype(int)
            
            # 4. 价量关系
            price_change = np.diff(close, prepend=close[0])
            volume_change = np.diff(volume, prepend=volume[0])
            df['price_volume_corr'] = pd.Series(price_change).rolling(20).corr(pd.Series(volume_change))
            
            # 5. 资金流向
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            df['money_flow_14'] = pd.Series(money_flow).rolling(14).sum()
            
            # 6. 成交量加权平均价格(VWAP)
            df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
            df['vwap_ratio'] = close / df['vwap']
            
            # 7. 累积/派发线
            df['ad_line'] = talib.AD(high, low, close, volume)
            
            # 8. Chaikin资金流量
            df['chaikin_mf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            self.logger.debug("✅ 成交量特征增强完成")
            
        except Exception as e:
            self.logger.error(f"❌ 成交量特征增强失败: {e}")
        
        return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加多时间尺度特征"""
        try:
            # 模拟高级别时间框架（实际应该从API获取）
            # 这里用重采样方法近似
            
            # 1小时级别特征
            df_1h = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_1h) > 20:
                # 1小时RSI
                rsi_1h = talib.RSI(df_1h['close'].values, timeperiod=14)
                df['rsi_1h'] = self._interpolate_to_original_timeframe(df, df_1h, rsi_1h)
                
                # 1小时EMA
                ema_1h = talib.EMA(df_1h['close'].values, timeperiod=21)
                df['ema_1h'] = self._interpolate_to_original_timeframe(df, df_1h, ema_1h)
                
                # 1小时趋势
                df['trend_1h'] = (df['close'] > df['ema_1h']).astype(int)
            
            # 4小时级别特征
            df_4h = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_4h) > 20:
                # 4小时MACD
                macd_4h, macd_signal_4h, macd_hist_4h = talib.MACD(df_4h['close'].values)
                df['macd_4h'] = self._interpolate_to_original_timeframe(df, df_4h, macd_4h)
                
                # 4小时ADX
                adx_4h = talib.ADX(df_4h['high'].values, df_4h['low'].values, 
                                  df_4h['close'].values, timeperiod=14)
                df['adx_4h'] = self._interpolate_to_original_timeframe(df, df_4h, adx_4h)
            
            # 日线级别特征
            df_1d = df.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_1d) > 10:
                # 日线支撑阻力位
                df['daily_high'] = self._interpolate_to_original_timeframe(df, df_1d, df_1d['high'].values)
                df['daily_low'] = self._interpolate_to_original_timeframe(df, df_1d, df_1d['low'].values)
                
                # 距离日线高低点的距离
                df['distance_to_daily_high'] = (df['close'] - df['daily_high']) / df['daily_high']
                df['distance_to_daily_low'] = (df['close'] - df['daily_low']) / df['daily_low']
            
            self.logger.debug("✅ 多时间尺度特征添加完成")
            
        except Exception as e:
            self.logger.error(f"❌ 多时间尺度特征添加失败: {e}")
        
        return df
    
    def _interpolate_to_original_timeframe(self, df_original: pd.DataFrame, 
                                         df_resampled: pd.DataFrame, values: np.ndarray) -> pd.Series:
        """将高级别时间框架的数据插值到原始时间框架"""
        try:
            # 创建临时DataFrame进行插值
            temp_df = pd.DataFrame({'values': values}, index=df_resampled.index)
            temp_df = temp_df.reindex(df_original.index, method='ffill')
            return temp_df['values']
        except:
            return pd.Series(np.nan, index=df_original.index)
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间周期特征"""
        try:
            # 假设index是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                return df
            
            # 1. 小时特征（0-23）
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # 2. 星期特征（0-6）
            dayofweek = df.index.dayofweek
            df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)
            
            # 3. 月份特征（1-12）
            month = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * month / 12)
            df['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # 4. 交易时段特征
            # 亚洲时段：0-8 UTC
            # 欧洲时段：8-16 UTC
            # 美洲时段：16-24 UTC
            df['asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
            df['europe_session'] = ((hour >= 8) & (hour < 16)).astype(int)
            df['america_session'] = ((hour >= 16) & (hour < 24)).astype(int)
            
            # 5. 是否周末
            df['is_weekend'] = (dayofweek >= 5).astype(int)
            
            self.logger.debug("✅ 时间周期特征添加完成")
            
        except Exception as e:
            self.logger.error(f"❌ 时间周期特征添加失败: {e}")
        
        return df
    
    def _add_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率制度特征"""
        try:
            close = df['close'].values
            
            # 1. 多期间波动率
            returns = np.diff(np.log(close), prepend=np.log(close[0]))
            
            vol_5 = pd.Series(returns).rolling(5).std() * np.sqrt(288)  # 15分钟 -> 年化
            vol_20 = pd.Series(returns).rolling(20).std() * np.sqrt(288)
            vol_60 = pd.Series(returns).rolling(60).std() * np.sqrt(288)
            
            df['volatility_5'] = vol_5
            df['volatility_20'] = vol_20
            df['volatility_60'] = vol_60
            
            # 2. 波动率制度分类
            vol_20_mean = vol_20.rolling(100).mean()
            vol_20_std = vol_20.rolling(100).std()
            
            # 低波动率制度: < mean - 0.5*std
            # 中波动率制度: mean - 0.5*std <= vol <= mean + 0.5*std
            # 高波动率制度: > mean + 0.5*std
            low_vol_threshold = vol_20_mean - 0.5 * vol_20_std
            high_vol_threshold = vol_20_mean + 0.5 * vol_20_std
            
            df['vol_regime_low'] = (vol_20 < low_vol_threshold).astype(int)
            df['vol_regime_high'] = (vol_20 > high_vol_threshold).astype(int)
            df['vol_regime_normal'] = ((vol_20 >= low_vol_threshold) & 
                                     (vol_20 <= high_vol_threshold)).astype(int)
            
            # 3. 波动率变化率
            df['volatility_change'] = vol_20.pct_change()
            
            # 4. 波动率分位数
            df['volatility_percentile'] = vol_20.rolling(252).rank(pct=True)  # 252个15分钟周期约1个交易日
            
            # 5. GARCH波动率（简化版）
            squared_returns = returns ** 2
            df['garch_vol'] = pd.Series(squared_returns).ewm(alpha=0.1).mean()
            
            self.logger.debug("✅ 波动率制度特征添加完成")
            
        except Exception as e:
            self.logger.error(f"❌ 波动率制度特征添加失败: {e}")
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场微结构特征"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values
            
            # 1. 价差特征
            df['spread'] = high - low
            df['spread_pct'] = (high - low) / close
            
            # 2. 影线分析
            upper_shadow = high - np.maximum(open_price, close)
            lower_shadow = np.minimum(open_price, close) - low
            body_size = np.abs(close - open_price)
            
            df['upper_shadow'] = upper_shadow / close
            df['lower_shadow'] = lower_shadow / close
            df['body_size'] = body_size / close
            df['shadow_ratio'] = (upper_shadow + lower_shadow) / np.maximum(body_size, 1e-8)
            
            # 3. K线形态特征
            df['doji'] = (body_size / (high - low) < 0.1).astype(int)
            df['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
            df['shooting_star'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
            
            # 4. 价格位置
            df['price_position'] = (close - low) / (high - low)
            
            # 5. 成交量强度
            typical_price = (high + low + close) / 3
            volume_price = volume * typical_price
            df['volume_intensity'] = pd.Series(volume_price).rolling(20).mean()
            
            # 6. 订单流不平衡（近似）
            # 用成交量和价格变化近似估算买卖压力
            price_change = np.diff(close, prepend=close[0])
            buy_volume = np.where(price_change > 0, volume, 0)
            sell_volume = np.where(price_change < 0, volume, 0)
            
            df['buy_sell_ratio'] = (pd.Series(buy_volume).rolling(20).sum() / 
                                   pd.Series(sell_volume).rolling(20).sum().clip(lower=1))
            
            # 7. 流动性指标
            df['amihud_illiq'] = np.abs(price_change) / (volume * close)  # Amihud非流动性指标
            
            self.logger.debug("✅ 市场微结构特征添加完成")
            
        except Exception as e:
            self.logger.error(f"❌ 市场微结构特征添加失败: {e}")
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征标准化处理"""
        try:
            # 获取数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除一些不需要标准化的列
            exclude_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            for col in feature_columns:
                if col in df.columns:
                    # 使用robust标准化处理异常值
                    values = df[col].values.reshape(-1, 1)
                    
                    # 先处理无穷大和NaN
                    finite_mask = np.isfinite(values.flatten())
                    if finite_mask.sum() > 0:
                        # 计算分位数进行截尾
                        q1 = np.percentile(values[finite_mask], 1)
                        q99 = np.percentile(values[finite_mask], 99)
                        values = np.clip(values, q1, q99)
                        
                        # 标准化
                        scaler = RobustScaler()
                        try:
                            values_scaled = scaler.fit_transform(values)
                            df[col] = values_scaled.flatten()
                        except:
                            # 如果标准化失败，使用简单的z-score
                            mean_val = np.nanmean(values)
                            std_val = np.nanstd(values)
                            if std_val > 0:
                                df[col] = (values.flatten() - mean_val) / std_val
            
            # 处理剩余的NaN
            df = df.fillna(method='ffill').fillna(0)
            
            self.logger.debug("✅ 特征标准化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 特征标准化失败: {e}")
        
        return df
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_column: str = None) -> Dict:
        """分析特征重要性"""
        try:
            if target_column is None:
                # 如果没有目标列，创建一个基于未来收益的目标
                target = df['close'].pct_change(5).shift(-5)  # 5期后收益
            else:
                target = df[target_column]
            
            # 获取特征列
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_columns = ['open', 'high', 'low', 'close', 'volume'] + ([target_column] if target_column else [])
            feature_columns = [col for col in feature_columns if col not in exclude_columns]
            
            # 准备数据
            X = df[feature_columns].fillna(0)
            y = target.fillna(0)
            
            # 移除无效数据
            valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                self.logger.warning("数据量不足，跳过特征重要性分析")
                return {}
            
            # 1. 相关性分析
            correlations = {}
            for col in feature_columns:
                if col in X.columns:
                    corr, p_value = pearsonr(X[col], y)
                    correlations[col] = {'correlation': corr, 'p_value': p_value}
            
            # 2. 互信息分析
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_importance = dict(zip(feature_columns, mi_scores))
            except:
                mi_importance = {}
            
            # 3. 随机森林重要性
            try:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                rf_importance = dict(zip(feature_columns, rf.feature_importances_))
            except:
                rf_importance = {}
            
            # 综合重要性评分
            importance_summary = {}
            for col in feature_columns:
                score = 0
                if col in correlations:
                    score += abs(correlations[col]['correlation']) * 0.3
                if col in mi_importance:
                    score += mi_importance[col] * 0.3
                if col in rf_importance:
                    score += rf_importance[col] * 0.4
                
                importance_summary[col] = score
            
            # 保存结果
            self.feature_importance = {
                'correlations': correlations,
                'mutual_info': mi_importance,
                'random_forest': rf_importance,
                'combined_score': importance_summary
            }
            
            self.logger.info(f"✅ 特征重要性分析完成，分析了{len(feature_columns)}个特征")
            return self.feature_importance
            
        except Exception as e:
            self.logger.exception(f"❌ 特征重要性分析失败: {e}")
            return {}
    
    def select_best_features(self, df: pd.DataFrame, max_features: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """选择最佳特征"""
        try:
            if not self.feature_importance:
                self.logger.warning("未进行特征重要性分析，先执行分析")
                self.analyze_feature_importance(df)
            
            max_features = max_features or self.feature_config['max_features']
            
            # 获取综合评分
            combined_scores = self.feature_importance.get('combined_score', {})
            
            if not combined_scores:
                self.logger.warning("无法获取特征重要性评分")
                return df, []
            
            # 按重要性排序
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 选择top特征
            selected_features = [feat for feat, score in sorted_features[:max_features] 
                               if score > self.feature_config['importance_threshold']]
            
            # 保留基础列
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            all_selected = base_columns + selected_features
            
            # 过滤存在的列
            final_columns = [col for col in all_selected if col in df.columns]
            
            df_selected = df[final_columns].copy()
            
            self.logger.info(f"✅ 特征选择完成，从{len(df.columns)}个特征中选择了{len(final_columns)}个")
            
            return df_selected, selected_features
            
        except Exception as e:
            self.logger.exception(f"❌ 特征选择失败: {e}")
            return df, []
    
    def analyze_feature_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析特征相关性"""
        try:
            # 获取数值特征
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 计算相关性矩阵
            correlation_matrix = df[numeric_columns].corr()
            
            # 保存结果
            self.feature_correlations = correlation_matrix
            
            # 找出高相关性特征对
            high_corr_pairs = []
            threshold = self.feature_config['correlation_threshold']
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > threshold:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': correlation_matrix.iloc[i, j]
                        })
            
            self.logger.info(f"✅ 相关性分析完成，发现{len(high_corr_pairs)}对高相关特征")
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.exception(f"❌ 相关性分析失败: {e}")
            return pd.DataFrame()
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = None) -> Tuple[pd.DataFrame, List[str]]:
        """移除高相关性特征"""
        try:
            threshold = threshold or self.feature_config['correlation_threshold']
            
            if self.feature_correlations is None:
                self.analyze_feature_correlations(df)
            
            # 获取数值特征
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in numeric_columns if col not in base_columns]
            
            # 找出要移除的特征
            to_remove = set()
            corr_matrix = self.feature_correlations
            
            for i in range(len(feature_columns)):
                for j in range(i+1, len(feature_columns)):
                    col1, col2 = feature_columns[i], feature_columns[j]
                    
                    if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                        corr_value = abs(corr_matrix.loc[col1, col2])
                        
                        if corr_value > threshold:
                            # 选择重要性较低的特征移除
                            if self.feature_importance.get('combined_score', {}):
                                score1 = self.feature_importance['combined_score'].get(col1, 0)
                                score2 = self.feature_importance['combined_score'].get(col2, 0)
                                if score1 < score2:
                                    to_remove.add(col1)
                                else:
                                    to_remove.add(col2)
                            else:
                                # 如果没有重要性信息，随机移除一个
                                to_remove.add(col2)
            
            # 移除高相关特征
            remaining_columns = [col for col in df.columns if col not in to_remove]
            df_filtered = df[remaining_columns].copy()
            
            removed_features = list(to_remove)
            self.logger.info(f"✅ 移除了{len(removed_features)}个高相关特征")
            
            return df_filtered, removed_features
            
        except Exception as e:
            self.logger.exception(f"❌ 移除高相关特征失败: {e}")
            return df, []
    
    def apply_dimensionality_reduction(self, df: pd.DataFrame, n_components: int = None, method: str = 'pca') -> Tuple[pd.DataFrame, Any]:
        """应用降维处理"""
        try:
            # 获取特征列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in numeric_columns if col not in base_columns]
            
            X = df[feature_columns].fillna(0)
            n_components = n_components or min(20, len(feature_columns))
            
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)
                
                # 创建新的特征名
                pca_columns = [f'pca_{i+1}' for i in range(n_components)]
                
                # 保留基础列和PCA特征
                df_reduced = df[base_columns].copy()
                for i, col in enumerate(pca_columns):
                    df_reduced[col] = X_reduced[:, i]
                
                # 记录解释的方差比例
                explained_variance = reducer.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                self.logger.info(f"✅ PCA降维完成，{n_components}个主成分解释了{cumulative_variance[-1]*100:.2f}%的方差")
                
                return df_reduced, {
                    'reducer': reducer,
                    'explained_variance': explained_variance,
                    'cumulative_variance': cumulative_variance,
                    'feature_columns': feature_columns
                }
            
            else:
                self.logger.warning(f"不支持的降维方法: {method}")
                return df, None
                
        except Exception as e:
            self.logger.exception(f"❌ 降维处理失败: {e}")
            return df, None
    
    def get_feature_engineering_summary(self, df: pd.DataFrame) -> Dict:
        """获取特征工程摘要"""
        try:
            summary = {
                'total_features': len(df.columns),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'missing_values': df.isnull().sum().sum(),
                'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
                'feature_categories': {
                    'base_ohlcv': 5,
                    'technical_indicators': 0,
                    'volume_features': 0,
                    'cyclical_features': 0,
                    'volatility_features': 0,
                    'microstructure_features': 0,
                    'multi_timeframe_features': 0
                }
            }
            
            # 统计特征类别
            for col in df.columns:
                if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'ema', 'sma', 'bb', 'atr', 'adx']):
                    summary['feature_categories']['technical_indicators'] += 1
                elif any(vol_term in col.lower() for vol_term in ['volume', 'vwap', 'mfi', 'ad_line']):
                    summary['feature_categories']['volume_features'] += 1
                elif any(time_term in col.lower() for time_term in ['hour', 'day', 'month', 'session']):
                    summary['feature_categories']['cyclical_features'] += 1
                elif any(vol_term in col.lower() for vol_term in ['volatility', 'vol_regime', 'garch']):
                    summary['feature_categories']['volatility_features'] += 1
                elif any(micro_term in col.lower() for micro_term in ['spread', 'shadow', 'doji', 'hammer']):
                    summary['feature_categories']['microstructure_features'] += 1
                elif any(tf_term in col.lower() for tf_term in ['_1h', '_4h', '_1d', 'daily']):
                    summary['feature_categories']['multi_timeframe_features'] += 1
            
            # 添加重要性分析结果
            if self.feature_importance:
                top_features = sorted(self.feature_importance.get('combined_score', {}).items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                summary['top_10_features'] = top_features
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ 获取特征工程摘要失败: {e}")
            return {}

def main():
    """测试特征工程功能"""
    print("🔧 测试增强版特征工程系统")
    
    # 这里可以添加测试代码
    feature_engineer = EnhancedFeatureEngineer()
    print("✅ 特征工程器初始化完成")
    print("📋 主要功能:")
    print("  - 多时间尺度特征")
    print("  - 增强技术指标")
    print("  - 成交量分析特征")
    print("  - 时间周期特征")
    print("  - 波动率制度特征")
    print("  - 特征重要性分析")
    print("  - 特征选择和降维")

if __name__ == "__main__":
    main() 