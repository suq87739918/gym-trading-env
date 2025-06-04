"""
信号过滤器模块 - 增强版入场信号优化系统
实现多指标共振确认、分层过滤逻辑，提高信号质量和胜率
基于RSI、布林带、ADX、SMC等指标的综合分析
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from utils.config import get_config
from utils.logger import get_logger

class EnhancedSignalFilter:
    """增强版信号过滤器 - 多指标共振确认系统"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('SignalFilter', 'signal_filter.log')
        
        # ✅ 信号过滤配置参数
        self.filter_config = {
            # RSI过滤参数
            'rsi_overbought': self.config.get('RSI_OVERBOUGHT', 70),
            'rsi_oversold': self.config.get('RSI_OVERSOLD', 30),
            'rsi_neutral_min': self.config.get('RSI_NEUTRAL_MIN', 40),
            'rsi_neutral_max': self.config.get('RSI_NEUTRAL_MAX', 60),
            
            # 布林带过滤参数
            'bb_upper_threshold': self.config.get('BB_UPPER_THRESHOLD', 0.8),
            'bb_lower_threshold': self.config.get('BB_LOWER_THRESHOLD', 0.2),
            'bb_squeeze_threshold': self.config.get('BB_SQUEEZE_THRESHOLD', 0.8),
            'bb_expansion_threshold': self.config.get('BB_EXPANSION_THRESHOLD', 1.2),
            
            # ADX过滤参数
            'adx_trend_threshold': self.config.get('ADX_TREND_THRESHOLD', 25),
            'adx_strong_trend': self.config.get('ADX_STRONG_TREND', 40),
            'adx_no_trend': self.config.get('ADX_NO_TREND', 20),
            
            # 成交量过滤参数
            'volume_confirmation_threshold': self.config.get('VOLUME_CONFIRMATION_THRESHOLD', 1.5),
            'volume_spike_threshold': self.config.get('VOLUME_SPIKE_THRESHOLD', 2.0),
            
            # 🔧 调低信号质量阈值 - 避免过滤掉所有有效信号
            'min_signal_quality': self.config.get('MIN_SIGNAL_QUALITY', 0.01),      # 从0.6降到0.01
            'min_confluence_score': self.config.get('MIN_CONFLUENCE_SCORE', 0.5),   # 从0.7降到0.5
            'min_filter_score': self.config.get('MIN_FILTER_SCORE', 0.3),           # 从0.5降到0.3
        }
    
    def apply_enhanced_signal_filter(self, df: pd.DataFrame, smc_signal_col: str = 'enhanced_smc_signal') -> pd.DataFrame:
        """
        ✅ 应用增强信号过滤器 - 主要入口函数
        
        Args:
            df: 包含技术指标和SMC信号的DataFrame
            smc_signal_col: SMC信号列名
            
        Returns:
            添加了过滤后信号的DataFrame
        """
        try:
            df_filtered = df.copy()
            
            # 1. 基础信号过滤
            df_filtered = self._apply_basic_filters(df_filtered, smc_signal_col)
            
            # 2. 多指标共振确认
            df_filtered = self._apply_confluence_confirmation(df_filtered, smc_signal_col)
            
            # 3. 分层过滤逻辑
            df_filtered = self._apply_layered_filtering(df_filtered, smc_signal_col)
            
            # 4. 最终信号评分和筛选
            df_filtered = self._calculate_final_signal_score(df_filtered, smc_signal_col)
            
            # 5. 生成过滤后的交易信号
            df_filtered = self._generate_filtered_signals(df_filtered)
            
            self.logger.info("✅ 增强信号过滤器应用完成")
            return df_filtered
            
        except Exception as e:
            self.logger.exception(f"❌ 应用增强信号过滤器失败: {e}")
            return df
    
    def _apply_basic_filters(self, df: pd.DataFrame, smc_signal_col: str) -> pd.DataFrame:
        """
        ✅ 应用基础过滤条件
        实现用户需求：RSI、布林带、ADX等基础指标过滤
        """
        try:
            df['basic_filter_score'] = 0.0
            df['basic_filter_reasons'] = ''
            
            for i in range(len(df)):
                filter_score = 0.0
                filter_reasons = []
                
                current_data = df.iloc[i]
                smc_signal = current_data.get(smc_signal_col, 0)
                
                if abs(smc_signal) < 0.1:  # 无明显SMC信号，跳过
                    continue
                
                # ===== RSI过滤 =====
                rsi = current_data.get('rsi', 50)
                
                if smc_signal > 0:  # 看涨信号
                    if rsi < self.filter_config['rsi_oversold']:
                        filter_score += 0.4  # RSI超卖支持看涨
                        filter_reasons.append('RSI超卖确认')
                    elif rsi > self.filter_config['rsi_overbought']:
                        filter_score -= 0.3  # RSI超买反对看涨
                        filter_reasons.append('RSI超买警告')
                    elif self.filter_config['rsi_neutral_min'] <= rsi <= self.filter_config['rsi_neutral_max']:
                        filter_score += 0.1  # RSI中性区域中等支持
                        filter_reasons.append('RSI中性')
                
                elif smc_signal < 0:  # 看跌信号
                    if rsi > self.filter_config['rsi_overbought']:
                        filter_score += 0.4  # RSI超买支持看跌
                        filter_reasons.append('RSI超买确认')
                    elif rsi < self.filter_config['rsi_oversold']:
                        filter_score -= 0.3  # RSI超卖反对看跌
                        filter_reasons.append('RSI超卖警告')
                    elif self.filter_config['rsi_neutral_min'] <= rsi <= self.filter_config['rsi_neutral_max']:
                        filter_score += 0.1  # RSI中性区域中等支持
                        filter_reasons.append('RSI中性')
                
                # ===== 布林带过滤 =====
                bb_position = current_data.get('bb_position', 0.5)
                bb_width = current_data.get('bb_width', 0.02)
                bb_squeeze = current_data.get('bb_squeeze', 0)
                bb_expansion = current_data.get('bb_expansion', 0)
                
                if smc_signal > 0:  # 看涨信号
                    if bb_position < self.filter_config['bb_lower_threshold']:
                        filter_score += 0.3  # 接近下轨支持看涨
                        filter_reasons.append('布林带下轨确认')
                    elif bb_position > self.filter_config['bb_upper_threshold']:
                        filter_score -= 0.2  # 接近上轨谨慎看涨
                        filter_reasons.append('布林带上轨警告')
                    
                    # 布林带压缩后扩张 - 突破信号
                    if bb_expansion and bb_squeeze == 0:
                        filter_score += 0.2
                        filter_reasons.append('布林带扩张突破')
                
                elif smc_signal < 0:  # 看跌信号
                    if bb_position > self.filter_config['bb_upper_threshold']:
                        filter_score += 0.3  # 接近上轨支持看跌
                        filter_reasons.append('布林带上轨确认')
                    elif bb_position < self.filter_config['bb_lower_threshold']:
                        filter_score -= 0.2  # 接近下轨谨慎看跌
                        filter_reasons.append('布林带下轨警告')
                    
                    # 布林带压缩后扩张 - 突破信号
                    if bb_expansion and bb_squeeze == 0:
                        filter_score += 0.2
                        filter_reasons.append('布林带扩张突破')
                
                # ===== ADX趋势强度过滤 =====
                adx = current_data.get('adx', 25)
                di_plus = current_data.get('di_plus', 0)
                di_minus = current_data.get('di_minus', 0)
                
                if adx > self.filter_config['adx_strong_trend']:
                    # 强趋势环境
                    if smc_signal > 0 and di_plus > di_minus:
                        filter_score += 0.3  # 强上升趋势确认看涨
                        filter_reasons.append('ADX强上升趋势')
                    elif smc_signal < 0 and di_minus > di_plus:
                        filter_score += 0.3  # 强下降趋势确认看跌
                        filter_reasons.append('ADX强下降趋势')
                    else:
                        filter_score -= 0.2  # 趋势方向不一致
                        filter_reasons.append('ADX趋势冲突')
                
                elif adx > self.filter_config['adx_trend_threshold']:
                    # 中等趋势环境
                    if smc_signal > 0 and di_plus > di_minus:
                        filter_score += 0.2
                        filter_reasons.append('ADX中上升趋势')
                    elif smc_signal < 0 and di_minus > di_plus:
                        filter_score += 0.2
                        filter_reasons.append('ADX中下降趋势')
                
                elif adx < self.filter_config['adx_no_trend']:
                    # 无趋势环境 - 震荡市场
                    filter_score -= 0.1  # 震荡市场降低信号权重
                    filter_reasons.append('ADX无趋势环境')
                
                # ===== 成交量确认过滤 =====
                volume_ratio = current_data.get('volume_ratio', 1.0)
                volume_sma_ratio = current_data.get('volume_sma_ratio', 1.0)
                
                if volume_ratio > self.filter_config['volume_spike_threshold']:
                    filter_score += 0.3  # 成交量激增确认信号
                    filter_reasons.append('成交量激增确认')
                elif volume_ratio > self.filter_config['volume_confirmation_threshold']:
                    filter_score += 0.2  # 成交量放大确认
                    filter_reasons.append('成交量放大确认')
                elif volume_ratio < 0.8:
                    filter_score -= 0.1  # 成交量萎缩警告
                    filter_reasons.append('成交量萎缩警告')
                
                # 短期vs长期成交量比较
                if volume_sma_ratio > 1.2:
                    filter_score += 0.1  # 短期成交量活跃
                    filter_reasons.append('短期成交量活跃')
                
                # 记录过滤结果
                df.loc[df.index[i], 'basic_filter_score'] = np.clip(filter_score, -1.0, 1.0)
                df.loc[df.index[i], 'basic_filter_reasons'] = '; '.join(filter_reasons)
            
            self.logger.debug("✅ 基础过滤条件应用完成")
            
        except Exception as e:
            self.logger.error(f"❌ 应用基础过滤条件失败: {e}")
        
        return df
    
    def _apply_confluence_confirmation(self, df: pd.DataFrame, smc_signal_col: str) -> pd.DataFrame:
        """
        ✅ 应用多指标共振确认
        实现用户需求的信号融合策略：当多个独立信号同时指向同一方向时，胜率更高
        """
        try:
            df['confluence_score'] = 0.0
            df['confluence_signals'] = ''
            
            for i in range(len(df)):
                current_data = df.iloc[i]
                smc_signal = current_data.get(smc_signal_col, 0)
                
                if abs(smc_signal) < 0.1:
                    continue
                
                confluence_signals = []
                confluence_score = 0.0
                
                # ===== 收集各类信号 =====
                
                # 1. SMC基础信号
                if abs(smc_signal) > 0.3:
                    confluence_signals.append(f"SMC强信号({smc_signal:.2f})")
                    confluence_score += abs(smc_signal) * 0.4
                
                # 2. 趋势信号组合
                ema_cross = current_data.get('ema_cross_signal', 0)
                macd = current_data.get('macd', 0)
                adx = current_data.get('adx', 25)
                
                trend_alignment = 0
                if (smc_signal > 0 and ema_cross > 0 and macd > 0) or \
                   (smc_signal < 0 and ema_cross < 0 and macd < 0):
                    trend_alignment = 1
                    confluence_signals.append("趋势指标对齐")
                    confluence_score += 0.3
                
                # 3. 动量信号组合
                rsi = current_data.get('rsi', 50)
                stoch_k = current_data.get('stoch_k', 50)
                mfi = current_data.get('mfi', 50)
                
                momentum_signals = []
                if smc_signal > 0:  # 看涨信号
                    if rsi < 35:
                        momentum_signals.append("RSI超卖")
                    if stoch_k < 25:
                        momentum_signals.append("Stoch超卖")
                    if mfi < 30:
                        momentum_signals.append("MFI超卖")
                else:  # 看跌信号
                    if rsi > 65:
                        momentum_signals.append("RSI超买")
                    if stoch_k > 75:
                        momentum_signals.append("Stoch超买")
                    if mfi > 70:
                        momentum_signals.append("MFI超买")
                
                if len(momentum_signals) >= 2:
                    confluence_signals.append(f"动量共振({len(momentum_signals)}个)")
                    confluence_score += 0.25 * len(momentum_signals)
                
                # 4. 波动率和布林带信号
                bb_position = current_data.get('bb_position', 0.5)
                bb_squeeze = current_data.get('bb_squeeze', 0)
                atr_normalized = current_data.get('atr_normalized', 0.02)
                
                if smc_signal > 0 and bb_position < 0.2:
                    confluence_signals.append("布林带下轨支撑")
                    confluence_score += 0.2
                elif smc_signal < 0 and bb_position > 0.8:
                    confluence_signals.append("布林带上轨阻力")
                    confluence_score += 0.2
                
                # 布林带压缩后的突破
                if bb_squeeze and atr_normalized > 0.03:
                    confluence_signals.append("压缩后高波动突破")
                    confluence_score += 0.25
                
                # 5. 成交量确认信号
                volume_ratio = current_data.get('volume_ratio', 1.0)
                volume_sma_ratio = current_data.get('volume_sma_ratio', 1.0)
                
                if volume_ratio > 1.8 and volume_sma_ratio > 1.3:
                    confluence_signals.append("双重成交量确认")
                    confluence_score += 0.3
                elif volume_ratio > 1.5:
                    confluence_signals.append("成交量确认")
                    confluence_score += 0.2
                
                # 6. K线形态确认
                candle_pattern = current_data.get('candle_pattern', 0)
                if abs(candle_pattern) > 0.5:
                    if (smc_signal > 0 and candle_pattern > 0) or (smc_signal < 0 and candle_pattern < 0):
                        confluence_signals.append(f"K线形态确认({candle_pattern:.1f})")
                        confluence_score += 0.15
                
                # ===== 计算综合共振分数 =====
                # 信号数量奖励
                signal_count_bonus = min(len(confluence_signals) * 0.1, 0.5)
                final_confluence_score = confluence_score + signal_count_bonus
                
                # 归一化到0-1范围
                final_confluence_score = min(final_confluence_score, 1.0)
                
                # 记录结果
                df.loc[df.index[i], 'confluence_score'] = final_confluence_score
                df.loc[df.index[i], 'confluence_signals'] = '; '.join(confluence_signals)
            
            self.logger.debug("✅ 多指标共振确认应用完成")
            
        except Exception as e:
            self.logger.error(f"❌ 应用多指标共振确认失败: {e}")
        
        return df
    
    def _apply_layered_filtering(self, df: pd.DataFrame, smc_signal_col: str) -> pd.DataFrame:
        """
        ✅ 应用分层过滤逻辑
        实现用户需求：分层过滤的逻辑，首先用SMC/PO3识别潜在交易机会，然后用其他指标条件过滤
        """
        try:
            df['layer1_passed'] = False  # SMC信号强度层
            df['layer2_passed'] = False  # 基础指标过滤层
            df['layer3_passed'] = False  # 共振确认层
            df['final_filter_passed'] = False  # 最终通过标志
            
            for i in range(len(df)):
                current_data = df.iloc[i]
                smc_signal = current_data.get(smc_signal_col, 0)
                
                # ===== Layer 1: SMC信号强度筛选 =====
                if abs(smc_signal) > 0.2:  # SMC信号足够强
                    df.loc[df.index[i], 'layer1_passed'] = True
                else:
                    continue  # SMC信号太弱，直接跳过
                
                # ===== Layer 2: 基础指标过滤 =====
                basic_filter_score = current_data.get('basic_filter_score', 0)
                
                # 根据信号方向设置不同的基础过滤阈值
                if smc_signal > 0:  # 看涨信号
                    layer2_threshold = 0.3  # 看涨信号需要较高的基础过滤分数
                else:  # 看跌信号
                    layer2_threshold = 0.3  # 看跌信号同样需要较高分数
                
                if basic_filter_score > layer2_threshold:
                    df.loc[df.index[i], 'layer2_passed'] = True
                else:
                    continue  # 基础过滤未通过
                
                # ===== Layer 3: 共振确认层 =====
                confluence_score = current_data.get('confluence_score', 0)
                
                # 动态调整共振阈值
                if abs(smc_signal) > 0.7:  # 极强SMC信号
                    confluence_threshold = 0.5  # 降低共振要求
                elif abs(smc_signal) > 0.5:  # 强SMC信号
                    confluence_threshold = 0.6  # 中等共振要求
                else:  # 中等SMC信号
                    confluence_threshold = 0.7  # 提高共振要求
                
                if confluence_score > confluence_threshold:
                    df.loc[df.index[i], 'layer3_passed'] = True
                else:
                    continue  # 共振确认未通过
                
                # ===== 最终综合评估 =====
                # 所有层都通过的信号才被标记为最终通过
                df.loc[df.index[i], 'final_filter_passed'] = True
            
            # 统计各层通过情况
            layer1_count = df['layer1_passed'].sum()
            layer2_count = df['layer2_passed'].sum()
            layer3_count = df['layer3_passed'].sum()
            final_count = df['final_filter_passed'].sum()
            
            self.logger.info(f"📊 分层过滤结果: Layer1={layer1_count}, Layer2={layer2_count}, Layer3={layer3_count}, 最终通过={final_count}")
            
        except Exception as e:
            self.logger.error(f"❌ 应用分层过滤逻辑失败: {e}")
        
        return df
    
    def _calculate_final_signal_score(self, df: pd.DataFrame, smc_signal_col: str) -> pd.DataFrame:
        """
        ✅ 计算最终信号评分
        综合所有过滤层的结果，生成最终的信号强度评分
        """
        try:
            df['final_signal_score'] = 0.0
            df['signal_strength_level'] = 0  # 0: 无效, 1: 弱, 2: 中, 3: 强, 4: 极强
            
            for i in range(len(df)):
                if not df['final_filter_passed'].iloc[i]:
                    continue  # 未通过最终过滤的信号评分为0
                
                current_data = df.iloc[i]
                smc_signal = current_data.get(smc_signal_col, 0)
                basic_filter_score = current_data.get('basic_filter_score', 0)
                confluence_score = current_data.get('confluence_score', 0)
                
                # ===== 综合评分计算 =====
                # 权重分配：SMC 40%, 基础过滤 30%, 共振确认 30%
                final_score = (
                    abs(smc_signal) * 0.4 +
                    max(basic_filter_score, 0) * 0.3 +  # 基础过滤分数取正值
                    confluence_score * 0.3
                ) * np.sign(smc_signal)  # 保持信号方向
                
                # ===== 信号强度等级分类 =====
                abs_score = abs(final_score)
                if abs_score >= 0.8:
                    strength_level = 4  # 极强信号
                elif abs_score >= 0.6:
                    strength_level = 3  # 强信号
                elif abs_score >= 0.4:
                    strength_level = 2  # 中等信号
                elif abs_score >= 0.2:
                    strength_level = 1  # 弱信号
                else:
                    strength_level = 0  # 无效信号
                
                # 记录结果
                df.loc[df.index[i], 'final_signal_score'] = final_score
                df.loc[df.index[i], 'signal_strength_level'] = strength_level
            
            # 统计信号强度分布
            strength_counts = df['signal_strength_level'].value_counts().sort_index()
            self.logger.info(f"📊 信号强度分布: {dict(strength_counts)}")
            
        except Exception as e:
            self.logger.error(f"❌ 计算最终信号评分失败: {e}")
        
        return df
    
    def _generate_filtered_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 生成过滤后的交易信号
        基于最终评分生成具体的交易动作建议
        """
        try:
            df['filtered_action'] = 0  # 0: 持有, 1: 做多, -1: 做空
            df['action_confidence'] = 0.0  # 动作置信度
            df['suggested_position_size'] = 0.0  # 建议仓位大小
            
            for i in range(len(df)):
                final_score = df['final_signal_score'].iloc[i]
                strength_level = df['signal_strength_level'].iloc[i]
                
                if strength_level == 0:
                    continue  # 无效信号，保持持有
                
                # ===== 生成交易动作 =====
                if final_score > 0:
                    df.loc[df.index[i], 'filtered_action'] = 1  # 做多
                elif final_score < 0:
                    df.loc[df.index[i], 'filtered_action'] = -1  # 做空
                
                # ===== 计算动作置信度 =====
                confidence = min(abs(final_score), 1.0)
                df.loc[df.index[i], 'action_confidence'] = confidence
                
                # ===== 建议仓位大小 =====
                # 基于信号强度和置信度动态调整仓位
                base_position = 0.1  # 基础仓位10%
                
                if strength_level == 4:  # 极强信号
                    position_multiplier = 2.0
                elif strength_level == 3:  # 强信号
                    position_multiplier = 1.5
                elif strength_level == 2:  # 中等信号
                    position_multiplier = 1.0
                else:  # 弱信号
                    position_multiplier = 0.5
                
                suggested_position = base_position * position_multiplier * confidence
                suggested_position = min(suggested_position, 0.2)  # 限制最大仓位20%
                
                df.loc[df.index[i], 'suggested_position_size'] = suggested_position
            
            # 统计生成的信号
            action_counts = df['filtered_action'].value_counts()
            avg_confidence = df[df['filtered_action'] != 0]['action_confidence'].mean()
            avg_position = df[df['filtered_action'] != 0]['suggested_position_size'].mean()
            
            self.logger.info(f"📊 过滤后信号统计: {dict(action_counts)}")
            self.logger.info(f"📊 平均置信度: {avg_confidence:.3f}, 平均建议仓位: {avg_position:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ 生成过滤后交易信号失败: {e}")
        
        return df
    
    def get_filter_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取过滤器应用摘要"""
        try:
            total_signals = (abs(df.get('enhanced_smc_signal', 0)) > 0.1).sum()
            passed_layer1 = df.get('layer1_passed', False).sum()
            passed_layer2 = df.get('layer2_passed', False).sum()
            passed_layer3 = df.get('layer3_passed', False).sum()
            final_passed = df.get('final_filter_passed', False).sum()
            
            filter_efficiency = final_passed / max(total_signals, 1)
            
            summary = {
                'total_smc_signals': int(total_signals),
                'layer1_passed': int(passed_layer1),
                'layer2_passed': int(passed_layer2),
                'layer3_passed': int(passed_layer3),
                'final_passed': int(final_passed),
                'filter_efficiency': filter_efficiency,
                'avg_final_score': df[df['final_filter_passed']]['final_signal_score'].mean() if final_passed > 0 else 0,
                'avg_confluence_score': df[df['final_filter_passed']]['confluence_score'].mean() if final_passed > 0 else 0,
                'signal_strength_distribution': dict(df['signal_strength_level'].value_counts()),
                'action_distribution': dict(df['filtered_action'].value_counts()),
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取过滤器摘要失败: {e}")
            return {}

def main():
    """测试信号过滤器功能"""
    print("🔍 测试增强版信号过滤器")
    
    # 这里可以添加测试代码
    filter_system = EnhancedSignalFilter()
    print("✅ 信号过滤器初始化完成")
    print("📋 主要功能:")
    print("  - 多指标共振确认")
    print("  - 分层过滤逻辑")
    print("  - RSI + 布林带 + ADX 综合分析")
    print("  - 动态信号评分系统")

if __name__ == "__main__":
    main() 