"""
Gym交易环境模块 - 优化版
实现SOL/USDT合约交易的强化学习环境
包含动态杠杆、动态止盈、完整SMC信号状态空间
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import required modules
from data.data_collector import DataCollector
from data.technical_indicators import TechnicalIndicators
from data.smc_signals import SMCSignals
from data.signal_filter import EnhancedSignalFilter
from environment.enhanced_risk_control import (
    EnhancedRiskController, 
    RiskControlConfig, 
    PositionRiskState,
    StopLossType,
    TakeProfitType
)
from environment.dynamic_position_manager import (
    DynamicPositionManager,
    PositionSizingConfig,
    PositionSizingMethod
)
from environment.reward_config import get_reward_config
from environment.balanced_reward_function import BalancedRewardFunction, create_reward_config, RewardObjective

class EnhancedPositionManager:
    """增强版智能仓位管理器 - 集成动态仓位策略"""
    
    def __init__(self, initial_balance: float = 10000.0, position_config: Dict = None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # ✅ 集成动态仓位管理器
        self.position_config = position_config or {}
        
        # 创建动态仓位配置
        self.dynamic_config = PositionSizingConfig(
            method=PositionSizingMethod(self.position_config.get('method', 'kelly_volatility')),
            max_position_ratio=self.position_config.get('max_position_ratio', 0.5),
            min_position_ratio=self.position_config.get('min_position_ratio', 0.01),
            kelly_multiplier=self.position_config.get('kelly_multiplier', 0.5),
            risk_per_trade=self.position_config.get('risk_per_trade', 0.02),
            enable_adaptive=self.position_config.get('enable_adaptive', True)
        )
        
        # 初始化动态仓位管理器
        self.dynamic_manager = DynamicPositionManager(self.dynamic_config)
        
        # 传统风险控制参数（向后兼容）
        self.leverage_limits = (1.0, 15.0)
        self.max_drawdown_limit = 0.15
        self.daily_loss_limit = 0.05
        self.consecutive_loss_limit = 5
        
        # 统计数据
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.max_balance = initial_balance
        
        # 日志
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🎯 增强版仓位管理器初始化: 方法={self.dynamic_config.method.value}")
    
    def calculate_position_size(self, signal_strength: float, confidence: float, 
                              volatility: float, balance: float, 
                              current_price: float = None, stop_loss_price: float = None,
                              market_data: pd.Series = None) -> float:
        """
        ✅ 增强版仓位计算 - 集成Kelly公式和波动率目标策略
        
        Args:
            signal_strength: 信号强度 (0-1)
            confidence: 信号置信度 (0-1) 
            volatility: 市场波动率 (0-1)
            balance: 当前余额
            current_price: 当前价格
            stop_loss_price: 止损价格
            market_data: 市场数据
            
        Returns:
            仓位大小比例
        """
        try:
            # 使用动态仓位管理器计算
            if current_price and stop_loss_price:
                position_ratio, calc_info = self.dynamic_manager.calculate_position_size(
                    account_balance=balance,
                    current_price=current_price,
                    stop_loss_price=stop_loss_price,
                    signal_strength=signal_strength * confidence,  # 合并信号强度和置信度
                    market_data=market_data
                )
                
                # 记录计算信息
                self.logger.debug(f"📊 动态仓位计算: {calc_info}")
                
                return position_ratio
            
            else:
                # 回退到传统方法（向后兼容）
                return self._calculate_traditional_position_size(
                    signal_strength, confidence, volatility, balance)
        
        except Exception as e:
            self.logger.error(f"❌ 仓位计算失败: {e}")
            # 返回保守的默认仓位
            return self.dynamic_config.min_position_ratio
    
    def _calculate_traditional_position_size(self, signal_strength: float, confidence: float, 
                                           volatility: float, balance: float) -> float:
        """传统仓位计算方法（向后兼容）"""
        # 基础仓位大小（基于信号强度）
        base_size = signal_strength * 0.2  # 降低基础仓位到20%
        
        # 置信度调整
        confidence_adjusted_size = base_size * confidence
        
        # 波动率调整（高波动率降低仓位）
        volatility_factor = max(0.5, 1.0 - volatility * 0.5)
        volatility_adjusted_size = confidence_adjusted_size * volatility_factor
        
        # 风险控制调整
        risk_factor = self._calculate_risk_factor(balance)
        final_size = volatility_adjusted_size * risk_factor
        
        return min(max(final_size, self.dynamic_config.min_position_ratio), 
                  self.dynamic_config.max_position_ratio)
    
    def calculate_dynamic_leverage(self, signal_strength: float, market_regime: str, 
                                 volatility: float, risk_score: float) -> float:
        """
        ✅ 动态杠杆计算 - 结合Kelly公式优化
        """
        # 获取Kelly建议的基础杠杆
        kelly_info = self.dynamic_manager.get_position_sizing_summary()
        kelly_fraction = kelly_info.get('current_kelly_fraction', 0.1)
        
        # Kelly杠杆建议：基于Kelly分数调整杠杆
        kelly_leverage = 1.0 + kelly_fraction * 10  # Kelly分数越高，杠杆越大
        
        # 传统信号强度杠杆
        if signal_strength > 0.8:
            signal_leverage = 8.0
        elif signal_strength > 0.6:
            signal_leverage = 5.0
        elif signal_strength > 0.4:
            signal_leverage = 3.0
        elif signal_strength > 0.2:
            signal_leverage = 2.0
        else:
            signal_leverage = 1.0
        
        # 混合杠杆策略
        mixed_leverage = (kelly_leverage * 0.4 + signal_leverage * 0.6)
        
        # 市场状态调整
        if market_regime == 'trending':
            mixed_leverage *= 1.3
        elif market_regime == 'volatile':
            mixed_leverage *= 0.7
        
        # 波动率调整
        if volatility > 0.8:
            mixed_leverage *= 0.5
        elif volatility < 0.3:
            mixed_leverage *= 1.2
        
        # 风险评分调整
        mixed_leverage *= (1.0 - risk_score * 0.3)
        
        # 应用限制
        final_leverage = np.clip(mixed_leverage, self.leverage_limits[0], self.leverage_limits[1])
        
        self.logger.debug(f"🎯 动态杠杆: Kelly={kelly_leverage:.2f}, 信号={signal_leverage:.2f}, "
                         f"混合={mixed_leverage:.2f}, 最终={final_leverage:.2f}")
        
        return final_leverage
    
    def _calculate_risk_factor(self, current_balance: float) -> float:
        """计算风险系数"""
        # 回撤风险
        drawdown = (self.max_balance - current_balance) / self.max_balance
        if drawdown > self.max_drawdown_limit * 0.8:
            drawdown_factor = 0.5
        elif drawdown > self.max_drawdown_limit * 0.5:
            drawdown_factor = 0.7
        else:
            drawdown_factor = 1.0
        
        # 连续亏损风险
        if self.consecutive_losses >= self.consecutive_loss_limit * 0.8:
            consecutive_factor = 0.6
        elif self.consecutive_losses >= self.consecutive_loss_limit * 0.5:
            consecutive_factor = 0.8
        else:
            consecutive_factor = 1.0
        
        return min(drawdown_factor, consecutive_factor)
    
    def update_statistics(self, pnl: float, balance: float, trade_info: Dict = None):
        """✅ 更新统计数据 - 同时更新动态管理器"""
        # 更新传统统计
        self.current_balance = balance
        self.max_balance = max(self.max_balance, balance)
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # ✅ 更新动态仓位管理器
        self.dynamic_manager.update_trade_result(pnl, trade_info)
        
        # 定期输出统计信息
        if hasattr(self, '_update_counter'):
            self._update_counter += 1
        else:
            self._update_counter = 1
        
        if self._update_counter % 50 == 0:  # 每50次交易输出一次
            summary = self.dynamic_manager.get_position_sizing_summary()
            self.logger.info(f"📊 仓位管理统计更新: 胜率={summary['trading_statistics']['win_rate']:.1%}, "
                           f"Kelly分数={summary['current_kelly_fraction']:.3f}")
    
    def get_kelly_info(self) -> Dict:
        """获取Kelly公式相关信息"""
        return self.dynamic_manager.get_position_sizing_summary()
    
    def reset_for_new_episode(self):
        """新episode重置（保留学习到的Kelly参数）"""
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        # 注意：不重置dynamic_manager的历史数据，保持学习效果

class SolUsdtTradingEnv(gym.Env):
    """SOL/USDT交易环境 - 优化版"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df: pd.DataFrame = None, mode: str = 'train', **kwargs):
        """
        初始化SOL/USDT交易环境
        
        Args:
            df: 包含OHLCV数据的DataFrame
            mode: 'train', 'validate', 'test'
            **kwargs: 其他配置参数
        """
        super().__init__()
        
        # 环境配置
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # ✅ 添加配置字典
        self.config = kwargs.copy()  # 保存所有配置参数
        
        # ✅ 增强风险控制配置
        self.risk_control_config = RiskControlConfig(
            # ATR止损配置
            atr_period=kwargs.get('atr_period', 14),
            atr_multiplier_stop=kwargs.get('atr_multiplier_stop', 2.0),
            atr_multiplier_take_profit=kwargs.get('atr_multiplier_take_profit', 3.0),
            
            # 波动率止损配置
            volatility_lookback=kwargs.get('volatility_lookback', 20),
            volatility_multiplier=kwargs.get('volatility_multiplier', 2.0),
            
            # 移动止盈配置
            trailing_activation_profit=kwargs.get('trailing_activation_profit', 0.05),
            trailing_stop_distance=kwargs.get('trailing_stop_distance', 0.03),
            trailing_step_size=kwargs.get('trailing_step_size', 0.01),
            
            # 分批止盈配置
            partial_profit_levels=kwargs.get('partial_profit_levels', [0.03, 0.06, 0.10]),
            partial_profit_sizes=kwargs.get('partial_profit_sizes', [0.3, 0.3, 0.4]),
            
            # 技术位止损配置
            technical_buffer_atr=kwargs.get('technical_buffer_atr', 1.0),
            
            # 混合策略权重
            hybrid_weights=kwargs.get('hybrid_weights', {
                'atr': 0.4,
                'volatility': 0.3,
                'technical': 0.3
            })
        )
        
        # ✅ 初始化增强风险控制器
        self.risk_controller = EnhancedRiskController(self.risk_control_config)
        
        # ✅ 风险控制策略选择
        self.stop_loss_type = StopLossType(kwargs.get('stop_loss_type', 'hybrid'))
        self.take_profit_type = TakeProfitType(kwargs.get('take_profit_type', 'trailing_stop'))
        
        # ✅ 风险控制开关
        self.enable_atr_adaptive = kwargs.get('enable_atr_adaptive', True)
        self.enable_trailing_stop = kwargs.get('enable_trailing_stop', True)
        self.enable_partial_profit = kwargs.get('enable_partial_profit', True)
        self.enable_technical_levels = kwargs.get('enable_technical_levels', True)
        
        # ✅ 增强版Reward函数配置参数
        self.reward_config = {
            # 基础PnL奖励配置
            'pnl_scale_factor': kwargs.get('pnl_scale_factor', 100),  # PnL放大倍数
            
            # 胜负附加奖励配置
            'win_bonus_large': kwargs.get('win_bonus_large', 2.0),   # 大盈利奖励（>5%）
            'win_bonus_medium': kwargs.get('win_bonus_medium', 1.0), # 中盈利奖励（2-5%）
            'win_bonus_small': kwargs.get('win_bonus_small', 0.5),   # 小盈利奖励（0-2%）
            'loss_penalty_large': kwargs.get('loss_penalty_large', -3.0),   # 大亏损惩罚（>5%）
            'loss_penalty_medium': kwargs.get('loss_penalty_medium', -1.5), # 中亏损惩罚（2-5%）
            'loss_penalty_small': kwargs.get('loss_penalty_small', -0.8),   # 小亏损惩罚（0-2%）
            
            # 连胜奖励配置
            'consecutive_win_bonus': kwargs.get('consecutive_win_bonus', 0.2),  # 每次连胜的额外奖励
            'max_consecutive_bonus': kwargs.get('max_consecutive_bonus', 1.0),  # 连胜奖励上限
            
            # 风险调整配置
            'risk_adjustment_strength': kwargs.get('risk_adjustment_strength', 0.5),  # 风险调整强度
            'volatility_penalty_high': kwargs.get('volatility_penalty_high', -0.5),  # 高波动率惩罚
            'volatility_penalty_medium': kwargs.get('volatility_penalty_medium', -0.2), # 中波动率惩罚
            'drawdown_penalty_high': kwargs.get('drawdown_penalty_high', -2.0),      # 高回撤惩罚倍数
            'drawdown_penalty_medium': kwargs.get('drawdown_penalty_medium', -1.0),  # 中回撤惩罚倍数
            
            # 趋势对齐奖励配置
            'strong_trend_bonus': kwargs.get('strong_trend_bonus', 0.5),     # 强趋势对齐奖励
            'weak_trend_bonus': kwargs.get('weak_trend_bonus', 0.2),         # 弱趋势对齐奖励
            'counter_trend_penalty': kwargs.get('counter_trend_penalty', -0.5), # 逆势惩罚
            
            # 信号质量奖励配置
            'high_quality_bonus': kwargs.get('high_quality_bonus', 0.6),     # 高质量信号奖励
            'low_quality_penalty': kwargs.get('low_quality_penalty', -0.4),  # 低质量信号惩罚
            
            # 时间相关惩罚配置
            'time_penalty_base': kwargs.get('time_penalty_base', -0.001),    # 基础时间惩罚
            'holding_inefficiency_penalty': kwargs.get('holding_inefficiency_penalty', -0.5), # 无效长持仓惩罚
            
            # 组合表现奖励配置
            'sharpe_ratio_bonus_scale': kwargs.get('sharpe_ratio_bonus_scale', 2.0),  # 夏普比率奖励倍数
            'win_rate_bonus_scale': kwargs.get('win_rate_bonus_scale', 2.0),          # 胜率奖励倍数
            'return_bonus_scale': kwargs.get('return_bonus_scale', 2.0),              # 收益奖励倍数
            
            # 结构识别奖励配置
            'structure_signal_bonus': kwargs.get('structure_signal_bonus', 0.4),     # 结构信号奖励
            'structure_indicator_bonus': kwargs.get('structure_indicator_bonus', 0.3), # 技术指标结构奖励
            'reasonable_profit_bonus': kwargs.get('reasonable_profit_bonus', 0.2),   # 合理盈利奖励
            'excellent_profit_bonus': kwargs.get('excellent_profit_bonus', 0.5),     # 优秀盈利奖励
        }
        
        # 交易配置
        self.initial_balance = kwargs.get('initial_balance', 10000.0)
        self.max_trades_per_day = kwargs.get('max_trades_per_day', 20)
        self.commission = kwargs.get('commission', 0.001)  # 0.1%
        self.slippage = kwargs.get('slippage', 0.0005)    # 0.05%
        
        # 仓位管理配置  
        self.max_leverage = kwargs.get('max_leverage', 3.0)
        self.max_position_size = kwargs.get('max_position_size', 0.95)
        self.min_trade_size = kwargs.get('min_trade_size', 0.01)
        
        # 观察窗口配置
        self.lookback_window = kwargs.get('lookback_window', 50)
        self.max_steps = kwargs.get('max_steps', 10000)
        
        # 风险管理配置
        self.daily_loss_limit = kwargs.get('daily_loss_limit', 0.05)  # 5%
        self.max_drawdown_limit = kwargs.get('max_drawdown_limit', 0.15)  # 15%
        self.max_consecutive_losses = kwargs.get('max_consecutive_losses', 5)
        
        # 动作空间：0=持有, 1=做多, 2=做空, 3=平仓
        self.action_space = gym.spaces.Discrete(4)
        
        # 数据处理
        if df is not None:
            self.df = self._prepare_data(df.copy())
        else:
            self.df = self._load_data()
        
        # 确定观察空间维度
        self.observation_features = self._get_observation_features()
        
        # ✅ 修复：准确计算观测空间维度
        # 观测向量包含：
        # 1. 核心特征 (len(self.observation_features))
        # 2. 滑动窗口特征 (固定7个)
        # 3. 持仓状态特征 (固定7个: 3个position_type + 1个hold_duration + 1个pnl + 1个steps_since_trade + 1个drawdown)
        windowed_features_count = 7  # _get_windowed_features返回7个特征
        position_features_count = 7  # _get_position_state_features返回7个特征
        
        total_obs_dim = len(self.observation_features) + windowed_features_count + position_features_count
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )
        
        self.logger.info(f"🔍 观测空间维度: {total_obs_dim}")
        self.logger.info(f"📊 特征分布: 核心特征={len(self.observation_features)}, 滑动窗口={windowed_features_count}, 持仓状态={position_features_count}")
        
        # ✅ 初始化增强版仓位管理器 - 传入动态仓位配置
        position_config = {
            'method': kwargs.get('position_sizing_method', 'kelly_volatility'),
            'max_position_ratio': kwargs.get('max_position_ratio', 0.5),
            'min_position_ratio': kwargs.get('min_position_ratio', 0.01),
            'kelly_multiplier': kwargs.get('kelly_multiplier', 0.5),
            'risk_per_trade': kwargs.get('risk_per_trade', 0.02),
            'enable_adaptive': kwargs.get('enable_adaptive_position', True)
        }
        
        self.position_manager = EnhancedPositionManager(self.initial_balance, position_config)
        
        # 状态变量初始化在reset()中
        
        # ✅ 新增：平衡奖励函数选项
        self.use_balanced_reward = kwargs.get('use_balanced_reward', False)
        reward_objective_str = kwargs.get('reward_objective', 'balanced')
        
        if self.use_balanced_reward:
            # 创建平衡奖励函数
            reward_objective = getattr(RewardObjective, reward_objective_str.upper(), RewardObjective.BALANCED)
            reward_config = create_reward_config(reward_objective)
            self.balanced_reward_function = BalancedRewardFunction(reward_config, self.logger)
            self.logger.info(f"🎯 启用平衡奖励函数: 目标={reward_objective_str}")
        else:
            self.balanced_reward_function = None
            self.logger.info("📊 使用传统奖励函数")
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        collector = DataCollector()
        df = collector.load_data()
        
        if df.empty:
            # 如果没有本地数据，尝试获取历史数据
            self.logger.warning("本地数据不存在，开始获取历史数据...")
            df = collector.get_historical_data()
            if not df.empty:
                collector.save_data(df)
        
        return df
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理 - 使用优化的核心指标"""
        try:
            from data.technical_indicators import TechnicalIndicators
            from data.smc_signals import SMCSignals
            
            # ✅ 使用增强版技术指标计算
            indicator_calculator = TechnicalIndicators()
            df_with_indicators = indicator_calculator.calculate_enhanced_indicators(df)
            
            # 计算SMC信号（已集成信号过滤器）
            smc_calculator = SMCSignals()
            df_final = smc_calculator.calculate_all_smc_signals(df_with_indicators)
            
            # 验证关键特征
            validation_results = indicator_calculator.validate_features(df_final)
            if validation_results['invalid_features']:
                self.logger.warning(f"⚠️ 发现无效特征: {validation_results['invalid_features']}")
            
            self.logger.info(f"✅ 增强数据预处理完成: {len(df_final.columns)} 列, {len(df_final)} 行")
            self.logger.info(f"✅ 核心特征验证: {len(validation_results['valid_features'])} 有效, {len(validation_results['missing_features'])} 缺失")
            
            return df_final
            
        except Exception as e:
            self.logger.exception(f"❌ 数据预处理失败: {e}")
            return df
    
    def _get_observation_features(self) -> List[str]:
        """✅ 获取增强的观察特征列表 - 集成新增技术指标和信号过滤器"""
        if self.df.empty:
            return []
        
        # ✅ 核心技术指标特征（增强版）
        core_tech_features = [
            # 趋势类（增强版 - 新增ADX）
            'price_vs_ema_fast', 'price_vs_ema_slow', 'ema_cross_signal',
            'adx', 'adx_normalized', 'trend_strength',
            
            # 动量类（增强版 - 新增Stochastic）
            'rsi_normalized', 'macd_normalized', 'macd_histogram',
            'stoch_k', 'stoch_d', 'stoch_cross_signal',
            
            # 波动率类（增强版 - 新增压缩检测）
            'atr_normalized', 'bb_position', 'bb_width', 'bb_squeeze', 'bb_expansion',
            
            # 成交量类（增强版 - 新增MFI）
            'volume_ratio', 'volume_sma_ratio', 'obv_normalized', 'price_vs_vwap', 'mfi_normalized',
            
            # 价格行为类（增强版 - 新增K线形态）
            'price_change_5', 'price_change_10', 'close_position_in_range', 'candle_pattern'
        ]
        
        # ✅ SMC核心信号特征（增强版）
        smc_core_features = [
            'enhanced_smc_signal', 'enhanced_signal_quality', 'technical_indicator_confirmation',
            'bos_bullish', 'bos_bearish', 'po3_phase',
            'order_block_signal', 'liquidity_sweep_signal', 'market_structure'
        ]
        
        # ✅ 新增：信号过滤器特征
        signal_filter_features = [
            'trend_alignment', 'momentum_alignment', 'volatility_regime',
            'signal_confluence', 'basic_filter_score', 'confluence_score',
            'final_signal_score', 'signal_strength_level'
        ]
        
        # ✅ 新增：过滤后的交易信号
        filtered_signal_features = [
            'filtered_action', 'action_confidence', 'final_filter_passed'
        ]
        
        # 持仓和交易状态特征
        position_features = [
            'position_type', 'position_size_normalized', 'hold_duration_normalized',
            'unrealized_pnl_normalized', 'portfolio_value_normalized'
        ]
        
        # 动态特征
        dynamic_features = [
            'current_leverage', 'trend_strength', 'volatility_regime'
        ]
        
        # ✅ 组合所有特征
        all_features = (core_tech_features + smc_core_features + signal_filter_features + 
                       filtered_signal_features + position_features + dynamic_features)
        
        # 过滤存在的特征
        available_features = []
        
        for feature in all_features:
            if feature in self.df.columns or feature in position_features + dynamic_features:
                available_features.append(feature)
            else:
                self.logger.debug(f"特征 {feature} 不存在，跳过")
        
        self.logger.info(f"✅ 增强观察特征列表：{len(available_features)} 个特征")
        self.logger.info(f"📊 特征分布: 技术指标={len(core_tech_features)}, SMC={len(smc_core_features)}, 过滤器={len(signal_filter_features)}")
        return available_features
    
    def _initialize_sliding_window(self):
        """初始化滑动窗口缓存"""
        self.window_size = self.config.get('OBSERVATION_WINDOW_SIZE', 10)  # 滑动窗口大小
        self.feature_window_cache = {}  # 特征滑动窗口缓存
        self.price_window_cache = []    # 价格滑动窗口缓存
        
        self.logger.info(f"🔄 初始化滑动窗口，窗口大小: {self.window_size}")
    
    def _update_sliding_window(self, current_idx: int):
        """更新滑动窗口缓存"""
        try:
            if current_idx >= len(self.df):
                return
            
            current_data = self.df.iloc[current_idx]
            
            # 更新价格窗口
            price_data = {
                'close': current_data.get('close', 0),
                'high': current_data.get('high', 0),
                'low': current_data.get('low', 0),
                'volume': current_data.get('volume', 0)
            }
            self.price_window_cache.append(price_data)
            
            # 保持窗口大小
            if len(self.price_window_cache) > self.window_size:
                self.price_window_cache.pop(0)
            
            # 更新特征窗口（只缓存核心特征的历史值）
            for feature in self.observation_features:
                if feature in self.df.columns:
                    if feature not in self.feature_window_cache:
                        self.feature_window_cache[feature] = []
                    
                    feature_value = current_data.get(feature, 0)
                    self.feature_window_cache[feature].append(feature_value)
                    
                    # 保持窗口大小
                    if len(self.feature_window_cache[feature]) > self.window_size:
                        self.feature_window_cache[feature].pop(0)
            
        except Exception as e:
            self.logger.error(f"❌ 更新滑动窗口失败: {e}")
    
    def _get_windowed_features(self) -> List[float]:
        """获取滑动窗口特征（时序特征）"""
        windowed_features = []
        
        try:
            # 1. 价格变化序列特征
            if len(self.price_window_cache) >= 2:
                # 最近3步的收益率
                recent_returns = []
                for i in range(min(3, len(self.price_window_cache)-1)):
                    if i < len(self.price_window_cache)-1:
                        curr_price = self.price_window_cache[-(i+1)]['close']
                        prev_price = self.price_window_cache[-(i+2)]['close']
                        if prev_price > 0:
                            return_rate = (curr_price - prev_price) / prev_price
                            recent_returns.append(np.clip(return_rate, -0.1, 0.1))
                        else:
                            recent_returns.append(0.0)
                
                # 填充到固定长度
                while len(recent_returns) < 3:
                    recent_returns.append(0.0)
                windowed_features.extend(recent_returns)
            else:
                windowed_features.extend([0.0] * 3)
            
            # 2. 关键指标的短期趋势
            trend_indicators = ['rsi_normalized', 'bb_position', 'volume_ratio']
            for indicator in trend_indicators:
                if indicator in self.feature_window_cache and len(self.feature_window_cache[indicator]) >= 2:
                    values = self.feature_window_cache[indicator][-3:]  # 最近3个值
                    
                    # 计算趋势方向 (简单差分)
                    if len(values) >= 2:
                        trend = values[-1] - values[-2] if values[-2] != 0 else 0
                        windowed_features.append(np.clip(trend, -1.0, 1.0))
                    else:
                        windowed_features.append(0.0)
                else:
                    windowed_features.append(0.0)
            
            # 3. 成交量变化模式
            if len(self.price_window_cache) >= 3:
                volumes = [data['volume'] for data in self.price_window_cache[-3:]]
                # 成交量相对变化
                if volumes[-2] > 0:
                    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
                    windowed_features.append(np.clip(volume_change, -2.0, 2.0))
                else:
                    windowed_features.append(0.0)
            else:
                windowed_features.append(0.0)
            
        except Exception as e:
            self.logger.error(f"❌ 获取滑动窗口特征失败: {e}")
            # 如果出错，返回零特征
            windowed_features = [0.0] * 7  # 3个收益率 + 3个趋势 + 1个成交量变化
        
        return windowed_features

    def _calculate_dynamic_leverage(self, obs_data: pd.Series) -> float:
        """
        ✅ 核心优化：杠杆动态决策
        基于RSI、布林带宽度、SMC分数动态调整杠杆
        """
        try:
            # 提取关键指标
            rsi = obs_data.get('rsi', 50) / 100.0  # 归一化到0-1
            bb_width = obs_data.get('bb_width', 0.1)
            
            # 计算SMC综合分数
            smc_score = 0.0
            smc_score += obs_data.get('bos_bullish', 0) + obs_data.get('bos_bearish', 0)  # BOS信号
            smc_score += min(obs_data.get('order_block_strength', 0), 1.0)  # Order Block强度
            smc_score += obs_data.get('po3_phase', 0) / 3.0  # PO3阶段(归一化)
            smc_score += abs(obs_data.get('smc_signal', 0))  # SMC信号强度
            
            # 高信号强度 + 高波动率 = 最大杠杆
            if smc_score > 2.0 and bb_width > 0.05:
                leverage = 5.0
                self.logger.debug(f"最大杠杆触发: SMC={smc_score:.2f}, BB_width={bb_width:.3f}")
            
            # RSI极值区域 = 中等杠杆
            elif rsi > 0.7 or rsi < 0.3:
                leverage = 3.0
                self.logger.debug(f"RSI极值杠杆: RSI={rsi:.2f}")
            
            # 低波动率环境 = 最小杠杆
            elif bb_width < 0.02:
                leverage = 1.0
                self.logger.debug(f"低波动率杠杆: BB_width={bb_width:.3f}")
            
            # 默认情况 = 中等杠杆
            else:
                # 根据信号强度线性调整
                base_leverage = 2.0
                signal_bonus = min(smc_score / 2.0, 1.0) * 1.5  # 最多增加1.5倍
                volatility_adjustment = min(bb_width / 0.1, 1.0) * 0.5  # 波动率调整
                
                leverage = base_leverage + signal_bonus + volatility_adjustment
                leverage = min(max(leverage, 1.0), 5.0)  # 限制在1-5倍范围
            
            return round(leverage, 2)
            
        except Exception as e:
            self.logger.error(f"动态杠杆计算失败: {e}")
            return 2.0  # 默认杠杆

    def compute_leverage(self, current_idx: int) -> float:
        """
        智能杠杆调度系统 - 优化版
        调用新的动态杠杆计算方法
        """
        try:
            if current_idx >= len(self.df):
                return 1.0
            
            current_data = self.df.iloc[current_idx]
            
            # 使用新的动态杠杆计算方法
            leverage = self._calculate_dynamic_leverage(current_data)
            
            # 记录杠杆计算详情
            self.logger.log_signal(
                'leverage_calculation', 'SOLUSDT', leverage,
                {
                    'rsi': current_data.get('rsi', 50),
                    'bb_width': current_data.get('bb_width', 0.1),
                    'smc_signal': current_data.get('smc_signal', 0),
                    'final_leverage': leverage
                }
            )
            
            return leverage
            
        except Exception as e:
            self.logger.error(f"杠杆计算失败: {e}")
            return 1.0
    
    def compute_dynamic_stop_loss(self, entry_price: float, position_type: int, current_idx: int) -> float:
        """
        ✅ 增强止损机制 - 结构自适应 + ATR动态调整
        """
        try:
            if current_idx >= len(self.df):
                fallback_pct = 0.02
                return entry_price * (1 - fallback_pct if position_type == 1 else 1 + fallback_pct)
            
            current_data = self.df.iloc[current_idx]
            
            # 1. ATR基础止损
            atr = current_data.get('atr', entry_price * 0.015)
            base_atr_multiplier = 2.0
            
            # 2. ✅ 结构识别调整因子 - 增强版
            structure_factor = 1.0
            
            # 检查重要结构信号
            bos_bullish = current_data.get('bos_bullish', 0)
            bos_bearish = current_data.get('bos_bearish', 0)
            choch_signal = current_data.get('po3_phase', 0)  # PO3阶段作为CHoCH代理
            order_block_strength = current_data.get('order_block_strength', 0)
            
            # ✅ 新增：基于用户示例的结构调整逻辑
            # 检查是否有结构突破（BOS）
            if position_type == 1:  # 多头
                if bos_bullish or choch_signal == 3 or order_block_strength > 0.5:
                    structure_factor = 1.5  # BOS突破时放宽50%
                elif bos_bearish:
                    structure_factor = 0.7  # 反向信号时收紧30%
                # 特殊情况：强势CHoCH信号
                elif choch_signal == 2 and current_data.get('po3_strength', 0) > 0.6:
                    structure_factor = 1.3  # CHoCH确认时适度放宽
            else:  # 空头
                if bos_bearish or order_block_strength > 0.5:
                    structure_factor = 1.5  # 放宽50%
                elif bos_bullish or choch_signal == 3:
                    structure_factor = 0.7  # 收紧30%
                elif choch_signal == 2 and current_data.get('po3_strength', 0) > 0.6:
                    structure_factor = 1.3
            
            # 3. ✅ 波动率自适应调整 - 增强版
            atr_normalized = current_data.get('atr_normalized', 0.02)
            volatility_factor = 1.0
            
            if atr_normalized > 0.04:  # 高波动率(>4%)
                volatility_factor = 1.3  # 放宽止损1.3倍
            elif atr_normalized < 0.01:  # 低波动率(<1%)
                volatility_factor = 0.8  # 收紧止损0.8倍
            elif atr_normalized > 0.025:  # 中高波动率
                volatility_factor = 1.1
            elif atr_normalized < 0.015:  # 中低波动率
                volatility_factor = 0.9
            
            # 4. 综合调整系数
            adjusted_multiplier = base_atr_multiplier * structure_factor * volatility_factor
            
            # 5. 计算最终止损价位
            if position_type == 1:  # 多头
                stop_loss_price = entry_price - (atr * adjusted_multiplier)
                # ✅ 确保止损不超过最大限制（多头不超过5%）
                max_stop_loss = entry_price * 0.95
                stop_loss_price = max(stop_loss_price, max_stop_loss)
            else:  # 空头
                stop_loss_price = entry_price + (atr * adjusted_multiplier)
                # ✅ 确保止损不超过最大限制（空头不超过5%）
                max_stop_loss = entry_price * 1.05
                stop_loss_price = min(stop_loss_price, max_stop_loss)
            
            # ✅ 详细日志记录
            self.logger.debug(f"🛡️ 增强止损: 入场={entry_price:.4f}, 止损={stop_loss_price:.4f}, "
                            f"结构因子={structure_factor:.2f}, 波动率因子={volatility_factor:.2f}, "
                            f"ATR={atr:.4f}, 调整倍数={adjusted_multiplier:.2f}")
            
            return stop_loss_price
            
        except Exception as e:
            self.logger.error(f"增强止损计算失败: {e}")
            # 默认2%止损
            fallback_pct = 0.02
            if position_type == 1:
                return entry_price * (1 - fallback_pct)
            else:
                return entry_price * (1 + fallback_pct)

    def compute_dynamic_take_profit(self, entry_price: float, position_type: int, current_idx: int) -> float:
        """
        ✅ 增强止盈机制 - 结构自适应 + 多目标优化
        实现用户要求: adjusted_tp = price * (1 - base_tp_pct * structure_factor)
        """
        try:
            if current_idx >= len(self.df):
                return entry_price * (1.02 if position_type == 1 else 0.98)
            
            current_data = self.df.iloc[current_idx]
            
            # 1. ATR基础止盈
            atr = current_data.get('atr', entry_price * 0.015)
            base_atr_multiplier = 2.5  # 基础ATR倍数
            
            # 2. ✅ 结构识别调整 - 实现用户核心要求
            structure_factor = 1.0
            
            # 检查重要结构信号
            bos_bullish = current_data.get('bos_bullish', 0)
            bos_bearish = current_data.get('bos_bearish', 0)
            choch_signal = current_data.get('po3_phase', 0)  # PO3阶段
            order_block_strength = current_data.get('order_block_strength', 0)
            market_structure = current_data.get('market_structure', 0)
            
            # ✅ 趋势行情下扩大止盈目标，震荡时保持谨慎 - 核心算法
            if position_type == 1:  # 多头
                if (bos_bullish or choch_signal == 3) and market_structure > 0:
                    structure_factor = 1.8  # 趋势行情：扩大80%止盈目标
                elif order_block_strength > 0.3:
                    structure_factor = 1.4  # 有支撑：适度扩大
                elif market_structure == 0:  # 震荡市
                    structure_factor = 0.8  # 震荡时保持谨慎
                # 新增：强势突破后的动量延续
                elif bos_bullish and current_data.get('bos_strength', 0) > 0.7:
                    structure_factor = 2.0  # 强势突破时大幅扩大目标
            else:  # 空头
                if (bos_bearish or order_block_strength > 0.3) and market_structure < 0:
                    structure_factor = 1.8  # 趋势行情：扩大止盈
                elif market_structure == 0:  # 震荡市
                    structure_factor = 0.8  # 震荡时保持谨慎
                elif bos_bearish and current_data.get('bos_strength', 0) > 0.7:
                    structure_factor = 2.0  # 强势突破
            
            # 3. ✅ 多目标止盈计算
            targets = []
            
            # ATR目标
            if position_type == 1:
                atr_target = entry_price + (atr * base_atr_multiplier * structure_factor)
            else:
                atr_target = entry_price - (atr * base_atr_multiplier * structure_factor)
            targets.append(atr_target)
            
            # ✅ 实现用户建议的公式: adjusted_tp = price * (1 - base_tp_pct * structure_factor)
            base_tp_pct = 0.015  # 1.5%基础止盈
            adjusted_tp_pct = base_tp_pct * structure_factor
            
            if position_type == 1:
                # 多头: price * (1 + adjusted_tp_pct)
                formula_target = entry_price * (1 + adjusted_tp_pct)
            else:
                # 空头: price * (1 - adjusted_tp_pct)  
                formula_target = entry_price * (1 - adjusted_tp_pct)
            targets.append(formula_target)
            
            # 4. ✅ 结构位目标（布林带、摆动点等）
            if position_type == 1 and 'bb_upper' in self.df.columns:
                bb_upper = current_data.get('bb_upper', 0)
                if bb_upper > entry_price:
                    # 趋势行情中可以冲击布林带上轨
                    structure_adjustment = 0.995 if structure_factor < 1.2 else 0.998
                    targets.append(bb_upper * structure_adjustment)
            elif position_type == -1 and 'bb_lower' in self.df.columns:
                bb_lower = current_data.get('bb_lower', 0)
                if bb_lower < entry_price and bb_lower > 0:
                    structure_adjustment = 1.005 if structure_factor < 1.2 else 1.002
                    targets.append(bb_lower * structure_adjustment)
            
            # 新增：摆动点阻力/支撑位目标
            if position_type == 1:
                last_swing_high = current_data.get('last_swing_high', 0)
                if last_swing_high > entry_price:
                    swing_target = last_swing_high * 0.995  # 略低于摆动高点
                    targets.append(swing_target)
            else:
                last_swing_low = current_data.get('last_swing_low', 0)
                if last_swing_low < entry_price and last_swing_low > 0:
                    swing_target = last_swing_low * 1.005  # 略高于摆动低点
                    targets.append(swing_target)
            
            # 5. ✅ 智能目标选择
            if position_type == 1:
                # 多头：根据趋势强度选择目标
                if structure_factor > 1.5:  # 强趋势
                    take_profit_price = max(targets)  # 选择最远目标
                else:
                    # 去除极值后取中等目标
                    targets.sort()
                    mid_index = len(targets) // 2
                    take_profit_price = targets[mid_index] if len(targets) > 2 else min(targets)
            else:
                # 空头：相应调整
                if structure_factor > 1.5:  # 强趋势
                    take_profit_price = min(targets)  # 选择最远目标
                else:
                    targets.sort(reverse=True)
                    mid_index = len(targets) // 2
                    take_profit_price = targets[mid_index] if len(targets) > 2 else max(targets)
            
            # 6. ✅ 确保最小收益要求
            min_profit_pct = 0.008  # 最小0.8%收益
            if position_type == 1:
                min_take_profit = entry_price * (1 + min_profit_pct)
                take_profit_price = max(take_profit_price, min_take_profit)
            else:
                max_take_profit = entry_price * (1 - min_profit_pct)
                take_profit_price = min(take_profit_price, max_take_profit)
            
            # ✅ 详细日志记录
            expected_return = abs(take_profit_price - entry_price) / entry_price * 100
            self.logger.debug(f"🎯 增强止盈: 入场={entry_price:.4f}, 止盈={take_profit_price:.4f}, "
                            f"结构因子={structure_factor:.2f}, 预期收益={expected_return:.2f}%, "
                            f"目标数量={len(targets)}, 最终选择={'最远' if structure_factor > 1.5 else '中等'}目标")
            
            return take_profit_price
            
        except Exception as e:
            self.logger.error(f"增强止盈计算失败: {e}")
            # 默认1.5%止盈
            if position_type == 1:
                return entry_price * 1.015
            else:
                return entry_price * 0.985
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境 - 符合gymnasium标准，返回(observation, info)"""
        # ✅ 兼容gymnasium标准，接受seed和options参数
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 重置交易状态
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.position_size = 0.0  # 持仓数量（正数=多头，负数=空头）
        self.position_type = 0  # 0=无持仓, 1=多头, -1=空头
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0

        # 动态交易参数
        self.current_leverage = 1.0
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.entry_step = 0

        # ✅ 增强风控状态初始化
        self.position_risk_state = None  # 当前持仓的风险状态
        self.risk_events_history = []    # 风控事件历史
        
        # ✅ 添加缺失的移动止损相关属性初始化
        self.trailing_stop_activation = self.risk_control_config.trailing_activation_profit
        self.trailing_stop_active = False
        self.trailing_stop_distance = self.risk_control_config.trailing_stop_distance
        self.highest_profit_price = 0.0
        self.trailing_stop_price = 0.0
        
        # ✅ 添加其他风控相关属性
        self.daily_pnl = 0.0
        self.max_single_trade_loss = 0.0
        self.hold_duration = 0

        # ✅ 初始化历史数据列表
        self.action_history = []
        self.portfolio_history = []
        self.reward_breakdown_history = []
        self.trade_history = []
        self.last_trade_step = 0

        # ✅ 初始化交易费用
        self.trading_fee = self.commission  # 确保交易费用正确设置

        # ✅ 初始化连续损失统计
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # ✅ 计算总步数
        self.total_steps = len(self.df) - self.lookback_window if len(self.df) > self.lookback_window else 100

        # 添加动态特征到数据框
        self._update_dynamic_features()

        # 初始化滑动窗口
        self._initialize_sliding_window()

        observation = self._get_observation()

        # ✅ 符合gymnasium标准，始终返回(observation, info)元组
        info = {
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'position_type': self.position_type,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown
        }

        # ✅ 重置平衡奖励函数
        if self.use_balanced_reward and self.balanced_reward_function:
            self.balanced_reward_function.reset_for_new_episode()

        return observation, info
    
    def _update_dynamic_features(self):
        """更新动态特征到数据框"""
        current_idx = self.current_step + self.lookback_window
        
        if current_idx >= len(self.df):
            return
        
        # 添加动态特征列（如果不存在）
        dynamic_cols = ['current_leverage', 'dynamic_stop_loss', 'dynamic_take_profit',
                       'trend_strength', 'volatility_regime', 'signal_confluence',
                       'position_size', 'position_type', 'unrealized_pnl_pct',
                       'portfolio_value_normalized', 'drawdown', 'hold_duration']
        
        for col in dynamic_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0
        
        # 更新当前值
        self.df.loc[self.df.index[current_idx], 'current_leverage'] = self.current_leverage
        self.df.loc[self.df.index[current_idx], 'dynamic_stop_loss'] = self.dynamic_stop_loss
        self.df.loc[self.df.index[current_idx], 'dynamic_take_profit'] = self.dynamic_take_profit
        self.df.loc[self.df.index[current_idx], 'position_size'] = self.position_size
        self.df.loc[self.df.index[current_idx], 'position_type'] = self.position_type
        
        # 计算趋势强度
        if 'market_structure' in self.df.columns and 'structure_strength' in self.df.columns:
            market_structure = self.df.loc[self.df.index[current_idx], 'market_structure']
            structure_strength = self.df.loc[self.df.index[current_idx], 'structure_strength']
            trend_strength = abs(market_structure) * structure_strength
            self.df.loc[self.df.index[current_idx], 'trend_strength'] = trend_strength
        
        # 计算波动率状态
        if 'atr_normalized' in self.df.columns:
            atr = self.df.loc[self.df.index[current_idx], 'atr_normalized']
            avg_atr = self.df['atr_normalized'].iloc[:current_idx+1].mean() if current_idx > 0 else atr
            volatility_regime = atr / avg_atr if avg_atr > 0 else 1.0
            self.df.loc[self.df.index[current_idx], 'volatility_regime'] = volatility_regime
        
        # 计算信号汇聚度
        signal_confluence = self._calculate_signal_confluence(current_idx)
        self.df.loc[self.df.index[current_idx], 'signal_confluence'] = signal_confluence
        
        # 更新其他指标
        if self.position_size != 0:
            current_price = self.df['close'].iloc[current_idx]
            if self.position_type == 1:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
            
            hold_duration = self.current_step - self.entry_step
        else:
            unrealized_pnl_pct = 0.0
            hold_duration = 0
        
        self.df.loc[self.df.index[current_idx], 'unrealized_pnl_pct'] = unrealized_pnl_pct
        self.df.loc[self.df.index[current_idx], 'portfolio_value_normalized'] = self.portfolio_value / self.initial_balance
        self.df.loc[self.df.index[current_idx], 'drawdown'] = self.max_drawdown
        self.df.loc[self.df.index[current_idx], 'hold_duration'] = hold_duration
    
    def _calculate_signal_confluence(self, current_idx: int) -> float:
        """计算信号汇聚度"""
        try:
            if current_idx >= len(self.df):
                return 0.0
            
            current_data = self.df.iloc[current_idx]
            signals = []
            
            # 技术指标信号
            if current_data.get('ema_golden_cross', 0):
                signals.append(1)
            elif current_data.get('ema_death_cross', 0):
                signals.append(-1)
            
            if current_data.get('macd_golden_cross', 0):
                signals.append(1)
            elif current_data.get('macd_death_cross', 0):
                signals.append(-1)
            
            # RSI信号
            rsi = current_data.get('rsi_normalized', 0.5)
            if rsi < 0.3:
                signals.append(1)
            elif rsi > 0.7:
                signals.append(-1)
            
            # SMC信号
            if current_data.get('bos_bullish', 0):
                signals.append(1)
            elif current_data.get('bos_bearish', 0):
                signals.append(-1)
            
            if current_data.get('bullish_order_block', 0):
                signals.append(1)
            elif current_data.get('bearish_order_block', 0):
                signals.append(-1)
            
            # 计算一致性
            if signals:
                signal_sum = sum(signals)
                signal_consistency = abs(signal_sum) / len(signals)
                return signal_consistency
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"计算信号汇聚度失败: {e}")
            return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """✅ 增强的环境步进 - 整合新的平衡奖励函数"""
        # 检查是否已经到达数据末尾
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # 更新市场数据滑动窗口
        self._update_sliding_window(self.current_step)
        
        # 更新动态特征
        self._update_dynamic_features()
        
        # 记录前一步的组合价值（用于奖励计算）
        previous_portfolio_value = self.balance + (
            self.position_size * current_price if self.position_type != 0 else 0
        )
        
        # 执行交易动作
        self._execute_trading_action(action, current_price, self.current_step)
        
        # 检查止损止盈
        force_exit = self._check_stop_loss_take_profit(current_price)
        
        # 检查增强风控
        should_force_exit, exit_reason, exit_info = self._check_enhanced_risk_control(
            current_price, self.current_step)
        
        if should_force_exit and not force_exit:
            self._close_position(current_price, self.current_step)
            force_exit = True
        
        # 更新风险管理统计
        self._update_risk_management_stats(current_price, force_exit, exit_reason if should_force_exit else "")
        
        # 更新投资组合价值
        self._update_portfolio_value()
        
        # ✅ 计算奖励 - 使用新的平衡奖励函数或传统函数
        if self.use_balanced_reward and self.balanced_reward_function:
            # 使用新的平衡奖励函数
            current_portfolio_value = self.balance + (
                self.position_size * current_price if self.position_type != 0 else 0
            )
            
            # 判断是否完成了交易
            trade_completed = (action == 3 and hasattr(self, '_last_trade_pnl'))
            trade_pnl_pct = getattr(self, '_last_trade_pnl', None) if trade_completed else None
            
            reward, reward_breakdown = self.balanced_reward_function.calculate_reward(
                current_portfolio_value=current_portfolio_value,
                previous_portfolio_value=previous_portfolio_value,
                action=action,
                trade_completed=trade_completed,
                trade_pnl_pct=trade_pnl_pct
            )
            
            # 重置交易PnL标记
            if hasattr(self, '_last_trade_pnl'):
                delattr(self, '_last_trade_pnl')
        else:
            # 使用传统奖励函数
            reward, reward_breakdown = self._calculate_optimized_reward(action, current_price, self.current_step)
        
        # 添加风险控制奖励调整
        if should_force_exit:
            risk_control_reward = self._calculate_risk_control_reward(exit_reason, exit_info)
            reward += risk_control_reward
            reward_breakdown['risk_control_reward'] = risk_control_reward
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        terminated = self._check_done()
        truncated = self.current_step >= len(self.df) - 1
        
        # 构建信息字典
        info = self._get_info()
        info['reward_breakdown'] = reward_breakdown
        
        if should_force_exit:
            info['risk_control_triggered'] = True
            info['exit_reason'] = exit_reason
            info['exit_info'] = exit_info
        else:
            info['risk_control_triggered'] = False
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_risk_control_reward(self, exit_reason: str, exit_info: Dict) -> float:
        """计算风控事件相关的奖励调整"""
        from utils.config import ENHANCED_REWARD_CONFIG
        
        if exit_reason == 'stop_loss':
            return ENHANCED_REWARD_CONFIG['STOP_LOSS_TRIGGER_PENALTY']
        elif exit_reason == 'take_profit':
            return ENHANCED_REWARD_CONFIG['TAKE_PROFIT_TRIGGER_BONUS']
        elif exit_reason == 'trailing_stop':
            return ENHANCED_REWARD_CONFIG['TRAILING_STOP_BONUS']
        elif exit_reason in ['time_stop', 'max_single_loss', 'daily_loss_limit']:
            return ENHANCED_REWARD_CONFIG['STOP_LOSS_TRIGGER_PENALTY'] * 1.5  # 更严重的惩罚
        else:
            return ENHANCED_REWARD_CONFIG['RISK_CONTROL_BONUS']  # 一般风控奖励
    
    def _update_risk_management_stats(self, current_price: float, force_exit: bool, exit_reason: str):
        """更新风险管理统计"""
        if self.position_size != 0:
            # 更新移动止损（如果启用）
            if self.enable_trailing_stop:
                current_pnl_pct = self._calculate_current_pnl_pct(current_price)
                if current_pnl_pct >= self.trailing_stop_activation and not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.logger.info(f"🔄 移动止损激活: 当前盈利={current_pnl_pct*100:.2f}%")
                
                if self.trailing_stop_active:
                    self._update_trailing_stop(current_price)
        
        # 更新日内盈亏
        if force_exit and self.position_size != 0:
            exit_pnl = self._calculate_position_pnl(current_price)
            self.daily_pnl += exit_pnl
            
            # 更新连续亏损统计
            if exit_pnl < 0:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.max_single_trade_loss = min(self.max_single_trade_loss, exit_pnl)
            else:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
    
    def _calculate_position_pnl(self, current_price: float) -> float:
        """计算当前持仓的绝对盈亏"""
        if self.position_size == 0:
            return 0.0
        
        if self.position_type == 1:  # 多头
            return self.position_size * (current_price - self.entry_price)
        else:  # 空头
            return abs(self.position_size) * (self.entry_price - current_price)
    
    def _calculate_optimized_reward(self, action: int, current_price: float, current_idx: int) -> Tuple[float, Dict]:
        """
        ✅ 增强版奖励函数 - 基于用户需求的综合优化设计
        
        核心改进：
        1. 盈亏为基础的奖励：将每笔交易的实际盈亏作为奖励基础
        2. 胜负附加奖励：在盈亏基础上增加胜率导向的激励
        3. 风险调整因素：引入对收益波动或回撤的惩罚
        4. 终局奖励与分段奖励：结合阶段性奖励和最终奖励
        5. 多目标平衡：同时优化收益、胜率和风险控制
        """
        reward_breakdown = {
            'base_pnl_reward': 0.0,           # 基础盈亏奖励
            'win_loss_bonus': 0.0,            # 胜负附加奖励
            'risk_adjusted_reward': 0.0,      # 风险调整后奖励
            'leverage_multiplier': 1.0,       # 杠杆倍数
            'structure_bonus': 0.0,           # 结构识别奖励
            'time_penalty': 0.0,              # 时间惩罚
            'volatility_penalty': 0.0,        # 波动率惩罚
            'trend_alignment_bonus': 0.0,     # 趋势对齐奖励
            'signal_quality_bonus': 0.0,      # 信号质量奖励
            'portfolio_performance_bonus': 0.0,# 组合表现奖励
            'consecutive_win_bonus': 0.0,     # 连胜奖励
            'drawdown_penalty': 0.0           # 回撤惩罚
        }
        
        total_reward = 0.0
        
        # ====================== 核心逻辑：平仓时的增强奖励计算 ======================
        if action == 3 and self.position_type != 0:  # 平仓动作
            # 1. ✅ 盈亏为基础的奖励 - 核心设计
            pnl_pct = self._calculate_current_pnl_pct(current_price)
            leverage = self.current_leverage if hasattr(self, 'current_leverage') else 1.0
            
            # 基础盈亏奖励：直接使用实际盈亏百分比，乘以杠杆放大
            base_pnl_reward = pnl_pct * leverage * self.reward_config['pnl_scale_factor']
            reward_breakdown['base_pnl_reward'] = base_pnl_reward
            reward_breakdown['leverage_multiplier'] = leverage
            
            # 2. ✅ 胜负附加奖励 - 提高胜率导向
            win_loss_bonus = 0.0
            if pnl_pct > 0:  # 盈利交易
                # 根据盈利幅度给予不同的胜利奖励
                if pnl_pct > 0.05:  # 大盈利（>5%）
                    win_loss_bonus = self.reward_config['win_bonus_large']
                elif pnl_pct > 0.02:  # 中等盈利（2%-5%）
                    win_loss_bonus = self.reward_config['win_bonus_medium']
                else:  # 小盈利（0%-2%）
                    win_loss_bonus = self.reward_config['win_bonus_small']
                
                # 连胜奖励机制
                consecutive_wins = getattr(self, 'consecutive_wins', 0) + 1
                if consecutive_wins >= 3:
                    consecutive_bonus = min(consecutive_wins * self.reward_config['consecutive_win_bonus'], 
                                          self.reward_config['max_consecutive_bonus'])
                    reward_breakdown['consecutive_win_bonus'] = consecutive_bonus
                    win_loss_bonus += consecutive_bonus
                
            else:  # 亏损交易
                # 根据亏损幅度给予不同的失败惩罚
                if pnl_pct < -0.05:  # 大亏损（>5%）
                    win_loss_bonus = self.reward_config['loss_penalty_large']
                elif pnl_pct < -0.02:  # 中等亏损（2%-5%）
                    win_loss_bonus = self.reward_config['loss_penalty_medium']
                else:  # 小亏损（0%-2%）
                    win_loss_bonus = self.reward_config['loss_penalty_small']
                
                # 重置连胜计数
                if hasattr(self, 'consecutive_wins'):
                    self.consecutive_wins = 0
            
            reward_breakdown['win_loss_bonus'] = win_loss_bonus
            
            # 3. ✅ 风险调整因素 - 综合风险控制
            volatility_adjustment = self._calculate_volatility_adjustment(current_idx)
            drawdown_penalty = self._calculate_drawdown_penalty()
            risk_score = self._calculate_position_risk_score(current_idx, pnl_pct)
            
            # 风险调整后的奖励
            risk_adjusted_multiplier = max(0.3, 1.0 - risk_score * self.reward_config['risk_adjustment_strength'])
            risk_adjusted_reward = base_pnl_reward * risk_adjusted_multiplier
            
            reward_breakdown['risk_adjusted_reward'] = risk_adjusted_reward
            reward_breakdown['volatility_penalty'] = volatility_adjustment
            reward_breakdown['drawdown_penalty'] = drawdown_penalty
            
            # 4. ✅ 结构识别奖励（鼓励在正确时机平仓）
            structure_bonus = self._calculate_enhanced_structure_bonus(current_idx, pnl_pct)
            reward_breakdown['structure_bonus'] = structure_bonus
            
            # 5. ✅ 组合表现奖励 - 终局奖励机制
            portfolio_bonus = self._calculate_portfolio_performance_bonus()
            reward_breakdown['portfolio_performance_bonus'] = portfolio_bonus
            
            # 主要奖励组合
            total_reward = risk_adjusted_reward + win_loss_bonus + structure_bonus + portfolio_bonus + volatility_adjustment + drawdown_penalty
            
            # 更新连胜计数
            if pnl_pct > 0:
                self.consecutive_wins = getattr(self, 'consecutive_wins', 0) + 1
            else:
                self.consecutive_wins = 0
            
            # 记录详细交易信息
            self.logger.info(f"📊 交易完成奖励: PnL={pnl_pct*100:.2f}%, 杠杆={leverage:.2f}x, "
                           f"基础奖励={base_pnl_reward:.3f}, 胜负奖励={win_loss_bonus:.3f}, "
                           f"风险调整后={risk_adjusted_reward:.3f}, 总奖励={total_reward:.3f}")
        
        # ==================== 持仓过程中的分段奖励计算 ====================
        else:
            # 1. 基础时间惩罚 - 鼓励及时决策
            time_penalty = self.reward_config['time_penalty_base'] * (1 + getattr(self, 'hold_duration', 0) * 0.1)
            reward_breakdown['time_penalty'] = time_penalty
            total_reward += time_penalty
            
            # 2. ✅ 持仓过程中的风险监控奖励
            if self.position_type != 0:
                current_pnl_pct = self._calculate_current_pnl_pct(current_price)
                leverage = self.current_leverage if hasattr(self, 'current_leverage') else 1.0
                
                # 浮动盈亏的分段奖励
                if current_pnl_pct > 0:
                    # 盈利中给予小额鼓励奖励，但不能太大以免过早平仓
                    floating_profit_reward = min(current_pnl_pct * leverage * 5, 0.2)
                    total_reward += floating_profit_reward
                else:
                    # 浮动亏损的风险警告惩罚
                    floating_loss_penalty = current_pnl_pct * leverage * 10  # 更强的亏损惩罚
                    total_reward += floating_loss_penalty
                
                # 持仓风险控制奖励
                hold_duration = getattr(self, 'hold_duration', 0)
                if hold_duration > 30:  # 超过30步的长期持仓
                    if abs(current_pnl_pct) < 0.005:  # 无明显盈亏的长期持仓
                        inefficiency_penalty = self.reward_config['holding_inefficiency_penalty'] - (hold_duration - 30) * 0.02
                        total_reward += inefficiency_penalty
                        reward_breakdown['time_penalty'] += inefficiency_penalty
        
        # ==================== 开仓时的信号质量奖励 ====================
        if action in [1, 2]:  # 开仓动作
            current_data = self.df.iloc[current_idx]
            
            # 1. ✅ 趋势对齐奖励增强
            enhanced_smc_signal = current_data.get('enhanced_smc_signal', current_data.get('smc_signal', 0))
            market_structure = current_data.get('market_structure', 0)
            signal_quality = current_data.get('signal_quality_score', 0)
            
            # 趋势对齐检查
            if (action == 1 and enhanced_smc_signal > 0.3 and market_structure >= 0) or \
               (action == 2 and enhanced_smc_signal < -0.3 and market_structure <= 0):
                # 强趋势对齐奖励
                trend_bonus = self.reward_config['strong_trend_bonus'] + signal_quality * 0.5
                reward_breakdown['trend_alignment_bonus'] = trend_bonus
                total_reward += trend_bonus
                self.logger.debug(f"📈 强趋势对齐奖励: {trend_bonus:.3f}, 信号质量: {signal_quality:.3f}")
            elif (action == 1 and enhanced_smc_signal > 0) or (action == 2 and enhanced_smc_signal < 0):
                # 一般趋势对齐奖励
                trend_bonus = self.reward_config['weak_trend_bonus'] + signal_quality * 0.2
                reward_breakdown['trend_alignment_bonus'] = trend_bonus
                total_reward += trend_bonus
            else:
                # 逆势开仓惩罚
                trend_penalty = self.reward_config['counter_trend_penalty'] - abs(enhanced_smc_signal) * 0.5
                reward_breakdown['trend_alignment_bonus'] = trend_penalty
                total_reward += trend_penalty
                self.logger.debug(f"📉 逆势开仓惩罚: {trend_penalty:.3f}")
            
            # 2. ✅ 多信号汇聚质量奖励
            signal_confluence = current_data.get('signal_confluence_enhanced', 
                                               current_data.get('signal_confluence', 0))
            combined_quality = signal_quality * signal_confluence
            
            # 高质量信号组合奖励
            if combined_quality > 0.7:
                quality_bonus = self.reward_config['high_quality_bonus'] * combined_quality
                reward_breakdown['signal_quality_bonus'] = quality_bonus
                total_reward += quality_bonus
                self.logger.debug(f"🎯 高质量信号组合奖励: {quality_bonus:.3f}")
            elif combined_quality < 0.3:
                # 低质量信号惩罚
                quality_penalty = self.reward_config['low_quality_penalty']
                reward_breakdown['signal_quality_bonus'] = quality_penalty
                total_reward += quality_penalty
                self.logger.debug(f"⚠️ 低质量信号惩罚: {quality_penalty:.3f}")
        
        # ==================== 最终奖励处理和限制 ====================
        
        # 限制奖励范围，防止异常值
        total_reward = np.clip(total_reward, -30.0, 30.0)
        
        # ✅ 详细奖励分解日志 (重要时刻记录)
        if abs(total_reward) > 0.5 or action == 3:
            self.logger.debug(f"🎁 奖励分解详情: 基础PnL={reward_breakdown['base_pnl_reward']:.3f}, "
                            f"胜负奖励={reward_breakdown['win_loss_bonus']:.3f}, "
                            f"风险调整={reward_breakdown['risk_adjusted_reward']:.3f}, "
                            f"结构奖励={reward_breakdown['structure_bonus']:.3f}, "
                            f"组合奖励={reward_breakdown['portfolio_performance_bonus']:.3f}, "
                            f"总奖励={total_reward:.3f}")
        
        return total_reward, reward_breakdown
    
    def _calculate_volatility_adjustment(self, current_idx: int) -> float:
        """计算波动率调整惩罚"""
        try:
            current_data = self.df.iloc[current_idx]
            atr_normalized = current_data.get('atr_normalized', 0.02)
            
            # 高波动率环境惩罚
            if atr_normalized > 0.05:  # 超高波动率
                return self.reward_config['volatility_penalty_high']
            elif atr_normalized > 0.03:  # 高波动率
                return self.reward_config['volatility_penalty_medium']
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_drawdown_penalty(self) -> float:
        """计算回撤惩罚"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            # 计算最近的最大回撤
            recent_portfolio = self.portfolio_history[-20:]  # 最近20步
            peak = max(recent_portfolio)
            current = recent_portfolio[-1]
            
            if peak > 0:
                drawdown = (peak - current) / peak
                if drawdown > 0.10:  # 回撤超过10%
                    return self.reward_config['drawdown_penalty_high'] * drawdown
                elif drawdown > 0.05:  # 回撤超过5%
                    return self.reward_config['drawdown_penalty_medium'] * drawdown
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_position_risk_score(self, current_idx: int, pnl_pct: float) -> float:
        """计算持仓风险评分 (0-1，越高风险越大)"""
        try:
            risk_factors = []
            current_data = self.df.iloc[current_idx]
            
            # 1. 持仓时间风险
            hold_duration = getattr(self, 'hold_duration', 0)
            time_risk = min(hold_duration / 50.0, 0.8)  # 最高0.8
            risk_factors.append(time_risk)
            
            # 2. 杠杆风险
            leverage = getattr(self, 'current_leverage', 1.0)
            leverage_risk = min((leverage - 1.0) / 4.0, 0.8)  # 5倍杠杆以上高风险
            risk_factors.append(leverage_risk)
            
            # 3. 市场波动率风险
            atr_normalized = current_data.get('atr_normalized', 0.02)
            volatility_risk = min(atr_normalized / 0.06, 0.8)  # 高于6%为高风险
            risk_factors.append(volatility_risk)
            
            # 4. 浮动亏损风险
            if pnl_pct < 0:
                loss_risk = min(abs(pnl_pct) / 0.05, 0.8)  # 5%以上亏损高风险
            else:
                loss_risk = 0.0
            risk_factors.append(loss_risk)
            
            # 5. 信号质量风险（信号质量越低风险越高）
            signal_quality = current_data.get('signal_quality_score', 0.5)
            signal_risk = max(0, (0.5 - signal_quality) * 1.6)  # 低于0.5质量增加风险
            risk_factors.append(signal_risk)
            
            return min(np.mean(risk_factors), 1.0)
        except:
            return 0.5  # 默认中等风险
    
    def _calculate_enhanced_structure_bonus(self, current_idx: int, pnl_pct: float) -> float:
        """增强版结构化平仓奖励"""
        try:
            if pnl_pct <= 0:  # 只对盈利平仓给予结构奖励
                return 0.0
            
            current_data = self.df.iloc[current_idx]
            structure_bonus = 0.0
            
            # 1. SMC结构位置奖励
            enhanced_smc_signal = current_data.get('enhanced_smc_signal', 0)
            if (self.position_type == 1 and enhanced_smc_signal < -0.4) or \
               (self.position_type == -1 and enhanced_smc_signal > 0.4):
                structure_bonus += self.reward_config['structure_signal_bonus']
            
            # 2. 技术指标确认奖励
            bb_position = current_data.get('bb_position', 0.5)
            if (self.position_type == 1 and bb_position > 0.8) or \
               (self.position_type == -1 and bb_position < 0.2):
                structure_bonus += self.reward_config['structure_indicator_bonus']
            
            # 3. 盈利幅度合理性奖励
            if 0.015 <= pnl_pct <= 0.04:  # 1.5%-4%的合理盈利范围
                structure_bonus += self.reward_config['reasonable_profit_bonus']
            elif pnl_pct > 0.06:  # 超过6%的优秀盈利
                structure_bonus += self.reward_config['excellent_profit_bonus']
            
            # 4. 持仓时间效率奖励
            hold_duration = getattr(self, 'hold_duration', 0)
            if 5 <= hold_duration <= 20:  # 合理持仓时间
                structure_bonus += 0.1
            elif hold_duration <= 4:  # 过快平仓略微惩罚
                structure_bonus -= 0.1
            
            return min(structure_bonus, 1.0)  # 限制最高奖励
            
        except Exception as e:
            self.logger.error(f"结构平仓奖励计算失败: {e}")
            return 0.0
    
    def _calculate_portfolio_performance_bonus(self) -> float:
        """计算组合表现奖励 - 终局奖励机制"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            # 1. 夏普比率奖励
            portfolio_returns = np.diff(self.portfolio_history[-50:]) if len(self.portfolio_history) > 50 else np.diff(self.portfolio_history)
            if len(portfolio_returns) > 5:
                sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)
                if sharpe_ratio > 0.1:
                    sharpe_bonus = min(sharpe_ratio * self.reward_config['sharpe_ratio_bonus_scale'], 0.5)
                elif sharpe_ratio < -0.1:
                    sharpe_bonus = max(sharpe_ratio * self.reward_config['sharpe_ratio_bonus_scale'], -0.5)
                else:
                    sharpe_bonus = 0.0
            else:
                sharpe_bonus = 0.0
            
            # 2. 胜率奖励
            recent_trades = [trade for trade in self.trade_history[-20:] if 'pnl' in trade]
            if len(recent_trades) >= 5:
                win_count = sum(1 for trade in recent_trades if trade['pnl'] > 0)
                win_rate = win_count / len(recent_trades)
                if win_rate > 0.6:
                    win_rate_bonus = (win_rate - 0.5) * self.reward_config['win_rate_bonus_scale']
                elif win_rate < 0.4:
                    win_rate_bonus = (win_rate - 0.5) * self.reward_config['win_rate_bonus_scale']
                else:
                    win_rate_bonus = 0.0
            else:
                win_rate_bonus = 0.0
            
            # 3. 总收益稳定性奖励
            total_return = (self.balance - 10000) / 10000  # 相对初始资金的收益率
            if total_return > 0.05:  # 总收益超过5%
                return_bonus = min(total_return * self.reward_config['return_bonus_scale'], 0.8)
            elif total_return < -0.05:  # 总收益低于-5%
                return_bonus = max(total_return * 3, -1.0)  # 加重惩罚
            else:
                return_bonus = 0.0
            
            return sharpe_bonus + win_rate_bonus + return_bonus
            
        except Exception as e:
            self.logger.error(f"组合表现奖励计算失败: {e}")
            return 0.0
    
    def _calculate_risk_adjustment_factor(self, current_idx: int, pnl_pct: float) -> float:
        """
        ✅ 计算风险调整因子 - 核心创新
        基于市场状态、波动率、信号质量等计算综合风险调整
        """
        try:
            current_data = self.df.iloc[current_idx]
            risk_factors = []
            
            # 1. 波动率风险调整
            atr_normalized = current_data.get('atr_normalized', 0.02)
            if atr_normalized > 0.04:  # 高波动率
                volatility_risk = 0.8  # 降低奖励
            elif atr_normalized < 0.01:  # 低波动率
                volatility_risk = 1.2  # 提升奖励
            else:
                volatility_risk = 1.0
            risk_factors.append(volatility_risk)
            
            # 2. 信号质量风险调整
            signal_quality = current_data.get('signal_quality_score', 0.5)
            quality_risk = 0.7 + signal_quality * 0.6  # 0.7-1.3范围
            risk_factors.append(quality_risk)
            
            # 3. 市场结构风险调整
            market_structure = current_data.get('market_structure', 0)
            if abs(market_structure) > 0.5:  # 明确趋势
                structure_risk = 1.1
            else:  # 震荡市场
                structure_risk = 0.9
            risk_factors.append(structure_risk)
            
            # 4. 持仓时间风险调整
            hold_duration = getattr(self, 'hold_duration', 0)
            if hold_duration > 20:  # 长时间持仓增加风险
                time_risk = max(0.8, 1.0 - (hold_duration - 20) * 0.01)
            else:
                time_risk = 1.0
            risk_factors.append(time_risk)
            
            # 5. ✅ 盈亏状态风险调整 - 增强版
            if pnl_pct > 0:  # 盈利时
                pnl_risk = 1.0 + min(pnl_pct * 2, 0.5)  # 盈利奖励加成
            else:  # 亏损时
                pnl_risk = max(0.5, 1.0 + pnl_pct * 3)  # 亏损惩罚加重
            risk_factors.append(pnl_risk)
            
            # ✅ 新增6. 组合信号汇聚度风险调整
            signal_confluence = current_data.get('signal_confluence_enhanced', 0.5)
            if signal_confluence > 0.8:  # 高汇聚度
                confluence_risk = 1.2  # 信号一致性高，提升奖励
            elif signal_confluence < 0.3:  # 低汇聚度
                confluence_risk = 0.8  # 信号混乱，降低奖励
            else:
                confluence_risk = 1.0
            risk_factors.append(confluence_risk)
            
            # ✅ 新增7. BOS/CHoCH结构确认风险调整
            bos_bullish = current_data.get('bos_bullish', 0)
            bos_bearish = current_data.get('bos_bearish', 0)
            choch_signal = current_data.get('po3_phase', 0)
            
            if bos_bullish or bos_bearish or choch_signal == 3:  # 有明确结构信号
                structure_confirmation_risk = 1.15  # 结构确认提升奖励
            elif choch_signal == 2:  # 操控阶段（风险较高）
                structure_confirmation_risk = 0.9
            else:
                structure_confirmation_risk = 1.0
            risk_factors.append(structure_confirmation_risk)
            
            # ✅ 新增8. 成交量确认风险调整
            if 'volume' in self.df.columns and current_idx > 0:
                current_volume = current_data.get('volume', 0)
                avg_volume = self.df['volume'].iloc[max(0, current_idx-19):current_idx+1].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 1.5:  # 高成交量确认
                    volume_risk = 1.1
                elif volume_ratio < 0.7:  # 低成交量（不够确认）
                    volume_risk = 0.9
                else:
                    volume_risk = 1.0
                risk_factors.append(volume_risk)
            
            # ✅ 新增9. 趋势强度风险调整
            enhanced_smc_signal = current_data.get('enhanced_smc_signal', 0)
            if abs(enhanced_smc_signal) > 0.7:  # 强信号
                signal_strength_risk = 1.1
            elif abs(enhanced_smc_signal) < 0.2:  # 弱信号
                signal_strength_risk = 0.9
            else:
                signal_strength_risk = 1.0
            risk_factors.append(signal_strength_risk)
            
            # ✅ 新增10. 止盈止损比例风险调整
            if hasattr(self, 'entry_price') and self.entry_price > 0:
                current_price = current_data.get('close', 0)
                price_move_pct = abs(current_price - self.entry_price) / self.entry_price
                
                if price_move_pct > 0.03:  # 价格移动超过3%
                    price_move_risk = 1.1  # 大幅移动提升风险调整
                elif price_move_pct < 0.005:  # 价格移动小于0.5%
                    price_move_risk = 0.95  # 小幅移动略降低奖励
                else:
                    price_move_risk = 1.0
                risk_factors.append(price_move_risk)
            
            # 综合风险调整因子计算
            risk_adjustment = np.mean(risk_factors)
            risk_adjustment = np.clip(risk_adjustment, 0.3, 2.0)  # 限制范围
            
            # ✅ 详细日志记录（仅在交易关键时刻）
            if pnl_pct != 0:  # 有盈亏时记录详细信息
                self.logger.debug(f"🔧 风险调整因子详情: 总计={risk_adjustment:.3f}")
                self.logger.debug(f"   波动率={volatility_risk:.2f}, 信号质量={quality_risk:.2f}, "
                                f"市场结构={structure_risk:.2f}, 持仓时间={time_risk:.2f}")
                self.logger.debug(f"   盈亏状态={pnl_risk:.2f}, 信号汇聚={confluence_risk:.2f}, "
                                f"结构确认={structure_confirmation_risk:.2f}")
            
            return risk_adjustment
            
        except Exception as e:
            self.logger.error(f"风险调整因子计算失败: {e}")
            return 1.0  # 默认无调整
    
    def _calculate_structure_exit_bonus(self, current_idx: int, pnl_pct: float) -> float:
        """
        ✅ 计算结构化平仓奖励
        奖励在合适的结构位置平仓的行为
        """
        try:
            if pnl_pct <= 0:  # 只对盈利平仓给予结构奖励
                return 0.0
            
            current_data = self.df.iloc[current_idx]
            structure_bonus = 0.0
            
            # 1. 在结构阻力/支撑位附近平仓奖励
            if self.position_type == 1:  # 多头平仓
                bb_upper = current_data.get('bb_upper', 0)
                current_price = current_data.get('close', 0)
                
                if bb_upper > 0:
                    distance_to_resistance = abs(current_price - bb_upper) / bb_upper
                    if distance_to_resistance < 0.01:  # 在1%范围内
                        structure_bonus += 0.3
            
            elif self.position_type == -1:  # 空头平仓
                bb_lower = current_data.get('bb_lower', 0)
                current_price = current_data.get('close', 0)
                
                if bb_lower > 0:
                    distance_to_support = abs(current_price - bb_lower) / bb_lower
                    if distance_to_support < 0.01:  # 在1%范围内
                        structure_bonus += 0.3
            
            # 2. 在信号反转时平仓奖励
            enhanced_smc_signal = current_data.get('enhanced_smc_signal', 0)
            if (self.position_type == 1 and enhanced_smc_signal < -0.3) or \
               (self.position_type == -1 and enhanced_smc_signal > 0.3):
                structure_bonus += 0.2  # 及时平仓奖励
            
            # 3. 盈利幅度结构奖励
            if 0.01 <= pnl_pct <= 0.03:  # 1%-3%的合理盈利
                structure_bonus += 0.1
            elif pnl_pct > 0.05:  # 超过5%的大盈利
                structure_bonus += 0.2
            
            return structure_bonus
            
        except Exception as e:
            self.logger.error(f"结构平仓奖励计算失败: {e}")
            return 0.0
    
    def _calculate_current_pnl_pct(self, current_price: float) -> float:
        """计算当前持仓盈亏百分比"""
        if self.position_type == 0:
            return 0.0
        
        price_change = (current_price - self.entry_price) / self.entry_price
        direction_multiplier = self.position_type
        return price_change * direction_multiplier

    def _execute_trading_action(self, action: int, current_price: float, current_idx: int):
        """
        ✅ 优化的交易动作执行 - 集成动态杠杆和仓位管理
        """
        if action == 1 and self.position_type == 0:  # 开多
            self._open_long_position_optimized(current_price, current_idx)
        elif action == 2 and self.position_type == 0:  # 开空
            self._open_short_position_optimized(current_price, current_idx)
        elif action == 3 and self.position_type != 0:  # 平仓
            self._close_position(current_price, current_idx)
        
        # 更新最后交易步数
        if action in [1, 2, 3]:
            self.last_trade_step = self.current_step
    
    def _open_long_position_optimized(self, current_price: float, current_idx: int):
        """✅ 优化的开多仓位逻辑 - 集成动态仓位管理器"""
        current_data = self.df.iloc[current_idx]
        
        # 计算信号强度和置信度
        signal_strength = max(0, current_data.get('smc_signal', 0))
        signal_confluence = self._calculate_signal_confluence(current_idx)
        volatility = current_data.get('atr_normalized', 0.5)
        
        # 🔍 记录开仓前的市场状态
        self.logger.info(f"📈 开始多头开仓分析:")
        self.logger.info(f"   ├─ 当前价格: ${current_price:.4f}")
        self.logger.info(f"   ├─ SMC信号强度: {signal_strength:.4f}")
        self.logger.info(f"   ├─ 信号共振度: {signal_confluence:.4f}")
        self.logger.info(f"   ├─ 市场波动率: {volatility:.4f}")
        self.logger.info(f"   ├─ 当前余额: ${self.balance:.2f}")
        self.logger.info(f"   └─ 市场环境: {self._get_market_regime(current_idx)}")
        
        # ✅ 计算动态止损价格
        stop_loss_price = self._calculate_enhanced_stop_loss(current_price, 1, current_idx)
        
        # ✅ 使用增强仓位管理器计算最优仓位大小
        position_size_ratio = self.position_manager.calculate_position_size(
            signal_strength=signal_strength,
            confidence=signal_confluence,
            volatility=volatility,
            balance=self.balance,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            market_data=current_data
        )
        
        # 计算动态杠杆
        leverage = self.position_manager.calculate_dynamic_leverage(
            signal_strength=signal_strength,
            market_regime=self._get_market_regime(current_idx),
            volatility=volatility,
            risk_score=self._calculate_current_risk_score()
        )
        
        # 计算实际仓位大小
        position_value = self.balance * position_size_ratio
        leveraged_value = position_value * leverage
        position_size = leveraged_value / current_price
        
        # 🔍 记录详细的仓位计算过程
        self.logger.info(f"🧮 多头仓位计算详情:")
        self.logger.info(f"   ├─ 仓位比例: {position_size_ratio:.4f}")
        self.logger.info(f"   ├─ 投资金额: ${position_value:.2f}")
        self.logger.info(f"   ├─ 动态杠杆: {leverage:.2f}x")
        self.logger.info(f"   ├─ 杠杆后价值: ${leveraged_value:.2f}")
        self.logger.info(f"   ├─ 实际仓位大小: {position_size:.6f}")
        self.logger.info(f"   ├─ 止损价格: ${stop_loss_price:.4f}")
        self.logger.info(f"   └─ 资金验证: {'✅ 通过' if position_size > 0 and leveraged_value <= self.balance * 10 else '❌ 失败'}")
        
        # ✅ 仓位安全检查
        if position_size <= 0:
            self.logger.error(f"❌ 多头仓位计算异常: position_size={position_size:.6f}, 跳过开仓")
            return
        
        if leveraged_value > self.balance * 10:  # 最大10倍杠杆限制
            self.logger.warning(f"⚠️ 杠杆后价值过高: ${leveraged_value:.2f}, 限制为10倍杠杆")
            leveraged_value = self.balance * 10
            position_size = leveraged_value / current_price
        
        # ✅ 使用增强风控计算止盈
        take_profit_price = self._calculate_enhanced_take_profit(current_price, 1, current_idx)
        
        # 更新仓位
        self.position_size = position_size
        self.position_type = 1
        self.entry_price = current_price
        self.current_leverage = leverage
        self.dynamic_stop_loss = stop_loss_price
        self.dynamic_take_profit = take_profit_price
        self.entry_step = self.current_step
        self.hold_duration = 0
        
        # 🔍 保存信号强度用于后续分析
        self._last_signal_strength = signal_strength
        
        # ✅ 创建持仓风险状态
        self.position_risk_state = PositionRiskState(
            entry_price=current_price,
            entry_time=current_idx,
            position_type=1,
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        # ✅ 激活移动止盈（如果启用）
        if self.enable_trailing_stop:
            self.risk_controller.setup_trailing_take_profit(self.position_risk_state, current_price)
        
        # ✅ 记录交易信息（包含Kelly信息）
        kelly_info = self.position_manager.get_kelly_info()
        self._log_trade("开多", current_price, position_size, 
                       leverage=leverage, signal_strength=signal_strength,
                       stop_loss=stop_loss_price, take_profit=take_profit_price,
                       position_ratio=position_size_ratio,
                       kelly_fraction=kelly_info.get('current_kelly_fraction', 0.1),
                       volatility=volatility)
        
        # 🔍 记录开仓完成状态
        self.logger.info(f"✅ 多头开仓完成:")
        self.logger.info(f"   ├─ 仓位大小: {position_size:.6f}")
        self.logger.info(f"   ├─ 入场价格: ${current_price:.4f}")
        self.logger.info(f"   ├─ 止损价格: ${stop_loss_price:.4f}")
        self.logger.info(f"   ├─ 止盈价格: ${take_profit_price:.4f}")
        self.logger.info(f"   ├─ 使用杠杆: {leverage:.2f}x")
        self.logger.info(f"   └─ Kelly分数: {kelly_info.get('current_kelly_fraction', 0.1):.4f}")
    
    def _open_short_position_optimized(self, current_price: float, current_idx: int):
        """✅ 优化的开空仓位逻辑 - 集成动态仓位管理器"""
        current_data = self.df.iloc[current_idx]
        
        # 计算信号强度和置信度
        signal_strength = abs(min(0, current_data.get('smc_signal', 0)))
        signal_confluence = self._calculate_signal_confluence(current_idx)
        volatility = current_data.get('atr_normalized', 0.5)
        
        # 🔍 记录开仓前的市场状态
        self.logger.info(f"📉 开始空头开仓分析:")
        self.logger.info(f"   ├─ 当前价格: ${current_price:.4f}")
        self.logger.info(f"   ├─ SMC信号强度: {signal_strength:.4f}")
        self.logger.info(f"   ├─ 信号共振度: {signal_confluence:.4f}")
        self.logger.info(f"   ├─ 市场波动率: {volatility:.4f}")
        self.logger.info(f"   ├─ 当前余额: ${self.balance:.2f}")
        self.logger.info(f"   └─ 市场环境: {self._get_market_regime(current_idx)}")
        
        # ✅ 计算动态止损价格
        stop_loss_price = self._calculate_enhanced_stop_loss(current_price, -1, current_idx)
        
        # ✅ 使用增强仓位管理器计算最优仓位大小
        position_size_ratio = self.position_manager.calculate_position_size(
            signal_strength=signal_strength,
            confidence=signal_confluence,
            volatility=volatility,
            balance=self.balance,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            market_data=current_data
        )
        
        # 计算动态杠杆
        leverage = self.position_manager.calculate_dynamic_leverage(
            signal_strength=signal_strength,
            market_regime=self._get_market_regime(current_idx),
            volatility=volatility,
            risk_score=self._calculate_current_risk_score()
        )
        
        # 计算实际仓位大小
        position_value = self.balance * position_size_ratio
        leveraged_value = position_value * leverage
        position_size = leveraged_value / current_price
        
        # 🔍 记录详细的仓位计算过程
        self.logger.info(f"🧮 空头仓位计算详情:")
        self.logger.info(f"   ├─ 仓位比例: {position_size_ratio:.4f}")
        self.logger.info(f"   ├─ 投资金额: ${position_value:.2f}")
        self.logger.info(f"   ├─ 动态杠杆: {leverage:.2f}x")
        self.logger.info(f"   ├─ 杠杆后价值: ${leveraged_value:.2f}")
        self.logger.info(f"   ├─ 实际仓位大小: {position_size:.6f}")
        self.logger.info(f"   ├─ 止损价格: ${stop_loss_price:.4f}")
        self.logger.info(f"   └─ 资金验证: {'✅ 通过' if position_size > 0 and leveraged_value <= self.balance * 10 else '❌ 失败'}")
        
        # ✅ 仓位安全检查
        if position_size <= 0:
            self.logger.error(f"❌ 空头仓位计算异常: position_size={position_size:.6f}, 跳过开仓")
            return
        
        if leveraged_value > self.balance * 10:  # 最大10倍杠杆限制
            self.logger.warning(f"⚠️ 杠杆后价值过高: ${leveraged_value:.2f}, 限制为10倍杠杆")
            leveraged_value = self.balance * 10
            position_size = leveraged_value / current_price
        
        # ✅ 使用增强风控计算止盈
        take_profit_price = self._calculate_enhanced_take_profit(current_price, -1, current_idx)
        
        # 更新仓位
        self.position_size = -position_size
        self.position_type = -1
        self.entry_price = current_price
        self.current_leverage = leverage
        self.dynamic_stop_loss = stop_loss_price
        self.dynamic_take_profit = take_profit_price
        self.entry_step = self.current_step
        self.hold_duration = 0
        
        # 🔍 保存信号强度用于后续分析
        self._last_signal_strength = signal_strength
        
        # ✅ 创建持仓风险状态
        self.position_risk_state = PositionRiskState(
            entry_price=current_price,
            entry_time=current_idx,
            position_type=-1,
            position_size=position_size,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        # ✅ 激活移动止盈（如果启用）
        if self.enable_trailing_stop:
            self.risk_controller.setup_trailing_take_profit(self.position_risk_state, current_price)
        
        # ✅ 记录交易信息（包含Kelly信息）
        kelly_info = self.position_manager.get_kelly_info()
        self._log_trade("开空", current_price, position_size, 
                       leverage=leverage, signal_strength=signal_strength,
                       stop_loss=stop_loss_price, take_profit=take_profit_price,
                       position_ratio=position_size_ratio,
                       kelly_fraction=kelly_info.get('current_kelly_fraction', 0.1),
                       volatility=volatility)
        
        # 🔍 记录开仓完成状态
        self.logger.info(f"✅ 空头开仓完成:")
        self.logger.info(f"   ├─ 仓位大小: {position_size:.6f}")
        self.logger.info(f"   ├─ 入场价格: ${current_price:.4f}")
        self.logger.info(f"   ├─ 止损价格: ${stop_loss_price:.4f}")
        self.logger.info(f"   ├─ 止盈价格: ${take_profit_price:.4f}")
        self.logger.info(f"   ├─ 使用杠杆: {leverage:.2f}x")
        self.logger.info(f"   └─ Kelly分数: {kelly_info.get('current_kelly_fraction', 0.1):.4f}")
    
    def _close_position(self, current_price: float, current_idx: int):
        """✅ 增强平仓逻辑 - 集成动态仓位管理器统计更新"""
        try:
            if self.position_type == 0:
                return
            
            # 🔍 记录平仓前状态
            pre_close_balance = self.balance
            pre_close_portfolio = self.portfolio_value
            
            self.logger.info(f"🔄 开始平仓操作:")
            self.logger.info(f"   ├─ 平仓前余额: ${pre_close_balance:.2f}")
            self.logger.info(f"   ├─ 平仓前组合价值: ${pre_close_portfolio:.2f}")
            self.logger.info(f"   ├─ 持仓类型: {'多头' if self.position_type == 1 else '空头'}")
            self.logger.info(f"   └─ 仓位大小: {abs(self.position_size):.6f}")
            
            # 计算盈亏
            if self.position_type == 1:  # 平多头
                exit_price = current_price * (1 - self.slippage)
                pnl = self.position_size * (exit_price - self.entry_price)
            else:  # 平空头
                exit_price = current_price * (1 + self.slippage)
                pnl = abs(self.position_size) * (self.entry_price - exit_price)
            
            # 计算费用
            position_value = abs(self.position_size) * current_price
            fee = position_value * self.trading_fee
            net_pnl = pnl - fee
            
            # ✅ 计算实际投入资金
            initial_margin = position_value / self.current_leverage
            
            # 🔍 记录详细的盈亏计算过程
            self.logger.info(f"💰 盈亏计算详情:")
            self.logger.info(f"   ├─ 入场价格: ${self.entry_price:.4f}")
            self.logger.info(f"   ├─ 出场价格: ${exit_price:.4f} (含滑点)")
            self.logger.info(f"   ├─ 原始盈亏: ${pnl:.2f}")
            self.logger.info(f"   ├─ 交易费用: ${fee:.2f}")
            self.logger.info(f"   ├─ 净盈亏: ${net_pnl:.2f}")
            self.logger.info(f"   ├─ 初始保证金: ${initial_margin:.2f}")
            self.logger.info(f"   └─ 当前杠杆: {self.current_leverage:.2f}x")
            
            # ✅ 更新余额
            old_balance = self.balance
            self.balance = self.balance - initial_margin + initial_margin + net_pnl
            
            # 安全检查：防止余额异常
            if self.balance <= 0 or not np.isfinite(self.balance):
                self.logger.error(f"❌ 余额异常: {self.balance:.2f}, 重置为最小值")
                self.balance = max(self.initial_balance * 0.01, 100)
            elif self.balance > self.initial_balance * 100:
                self.logger.warning(f"⚠️ 余额过高: {self.balance:.2f}, 限制增长")
                self.balance = min(self.balance, self.initial_balance * 100)
            
            # 🔍 记录余额变化
            balance_change = self.balance - old_balance
            self.logger.info(f"💳 资金结算结果:")
            self.logger.info(f"   ├─ 余额变化: ${old_balance:.2f} → ${self.balance:.2f}")
            self.logger.info(f"   ├─ 净变化: ${balance_change:.2f}")
            self.logger.info(f"   ├─ 收益率: {(balance_change/old_balance)*100:.2f}%" if old_balance > 0 else "   ├─ 收益率: N/A")
            self.logger.info(f"   └─ 余额检查: {'✅ 正常' if self.balance > 0 and np.isfinite(self.balance) else '❌ 异常'}")
            
            self.total_fees += fee
            
            # ✅ 更新基础统计
            self.total_trades += 1
            
            if net_pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            # ✅ 计算盈亏百分比
            pnl_pct = net_pnl / initial_margin if initial_margin > 0 else 0
            
            # ✅ 构建交易信息给动态仓位管理器
            trade_info = {
                'timestamp': current_idx,
                'position_ratio': position_value / self.balance if self.balance > 0 else 0,
                'leverage': self.current_leverage,
                'hold_duration': self.hold_duration,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position_type': self.position_type,
                'signal_strength': getattr(self, '_last_signal_strength', 0.5),
                'market_regime': self._get_market_regime(current_idx)
            }
            
            # ✅ 更新动态仓位管理器统计
            self.position_manager.update_statistics(net_pnl, self.balance, trade_info)
            
            # 记录交易
            self._log_trade("平仓", current_price, self.position_size, 
                          pnl=net_pnl, pnl_pct=pnl_pct, hold_duration=self.hold_duration,
                          trade_info=trade_info)
            
            # ✅ 输出Kelly信息（定期）
            if self.total_trades % 10 == 0:  # 每10次交易输出一次Kelly信息
                kelly_summary = self.position_manager.get_kelly_info()
                self.logger.info(f"📊 Kelly公式更新 (交易#{self.total_trades}): "
                               f"胜率={kelly_summary['trading_statistics']['win_rate']:.1%}, "
                               f"盈亏比={kelly_summary['trading_statistics']['profit_factor']:.2f}, "
                               f"Kelly分数={kelly_summary['current_kelly_fraction']:.3f}")
            
            # 重置仓位
            self.position_size = 0.0
            self.position_type = 0
            self.entry_price = 0.0
            self.current_leverage = 1.0
            self.dynamic_stop_loss = 0.0
            self.dynamic_take_profit = 0.0
            self.entry_step = 0
            self.hold_duration = 0
            
            # ✅ 重置风控状态
            self.stop_loss_price = 0.0
            self.take_profit_price = 0.0
            self.trailing_stop_price = 0.0
            self.trailing_stop_active = False
            self.highest_profit = 0.0
            self.position_entry_time = 0
            self.position_risk_state = None
            
            # 🔍 记录平仓后状态
            self.logger.info(f"✅ 平仓完成:")
            self.logger.info(f"   ├─ 最终余额: ${self.balance:.2f}")
            self.logger.info(f"   ├─ 累计交易: {self.total_trades}")
            self.logger.info(f"   ├─ 胜率: {(self.winning_trades/self.total_trades)*100:.1f}%" if self.total_trades > 0 else "   ├─ 胜率: N/A")
            self.logger.info(f"   └─ 连续亏损: {self.consecutive_losses}")
            
            # 记录交易PnL供奖励函数使用
            self._last_trade_pnl = pnl_pct
            
        except Exception as e:
            self.logger.error(f"❌ 平仓失败: {e}")
            # 异常情况下强制重置仓位
            self.position_type = 0
            self.position_size = 0.0
    
    def _check_stop_loss_take_profit(self, current_price: float):
        """
        ✅ 优化止盈止损机制检查 - 添加详细日志
        """
        if self.position_type == 0:
            return False  # 无仓位
        
        should_close = False
        close_reason = ""
        
        # 🔍 定期记录止盈止损价位状态（每50步记录一次）
        if self.current_step % 50 == 0 and (self.dynamic_stop_loss > 0 or self.dynamic_take_profit > 0):
            self.logger.debug(f"🎯 止盈止损监控 (Step {self.current_step}):")
            self.logger.debug(f"   ├─ 当前价格: ${current_price:.4f}")
            self.logger.debug(f"   ├─ 止损价位: ${self.dynamic_stop_loss:.4f}")
            self.logger.debug(f"   ├─ 止盈价位: ${self.dynamic_take_profit:.4f}")
            self.logger.debug(f"   └─ 持仓类型: {'多头' if self.position_type == 1 else '空头'}")
        
        # ✅ 检查止损 - 添加详细日志
        if self.dynamic_stop_loss > 0:
            if self.position_type == 1 and current_price <= self.dynamic_stop_loss:
                should_close = True
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                close_reason = f"多头止损触发: 当前价=${current_price:.4f} <= 止损价=${self.dynamic_stop_loss:.4f}, 亏损={pnl_pct*100:.2f}%"
                
                # 🔍 额外的止损分析
                stop_distance = (self.entry_price - current_price) / self.entry_price
                self.logger.warning(f"🛑 多头止损详细分析:")
                self.logger.warning(f"   ├─ 入场价格: ${self.entry_price:.4f}")
                self.logger.warning(f"   ├─ 当前价格: ${current_price:.4f}")
                self.logger.warning(f"   ├─ 止损价格: ${self.dynamic_stop_loss:.4f}")
                self.logger.warning(f"   ├─ 止损距离: {stop_distance*100:.2f}%")
                self.logger.warning(f"   ├─ 持仓时长: {self.hold_duration} 步")
                self.logger.warning(f"   └─ 预期亏损: ${abs(self.position_size) * (self.entry_price - current_price):.2f}")
                
            elif self.position_type == -1 and current_price >= self.dynamic_stop_loss:
                should_close = True
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                close_reason = f"空头止损触发: 当前价=${current_price:.4f} >= 止损价=${self.dynamic_stop_loss:.4f}, 亏损={pnl_pct*100:.2f}%"
                
                # 🔍 额外的止损分析
                stop_distance = (current_price - self.entry_price) / self.entry_price
                self.logger.warning(f"🛑 空头止损详细分析:")
                self.logger.warning(f"   ├─ 入场价格: ${self.entry_price:.4f}")
                self.logger.warning(f"   ├─ 当前价格: ${current_price:.4f}")
                self.logger.warning(f"   ├─ 止损价格: ${self.dynamic_stop_loss:.4f}")
                self.logger.warning(f"   ├─ 止损距离: {stop_distance*100:.2f}%")
                self.logger.warning(f"   ├─ 持仓时长: {self.hold_duration} 步")
                self.logger.warning(f"   └─ 预期亏损: ${abs(self.position_size) * (current_price - self.entry_price):.2f}")
        
        # ✅ 检查止盈 - 添加详细日志
        if not should_close and self.dynamic_take_profit > 0:
            if self.position_type == 1 and current_price >= self.dynamic_take_profit:
                should_close = True
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                close_reason = f"多头止盈触发: 当前价=${current_price:.4f} >= 止盈价=${self.dynamic_take_profit:.4f}, 盈利={pnl_pct*100:.2f}%"
                
                # 🔍 额外的止盈分析
                profit_distance = (current_price - self.entry_price) / self.entry_price
                self.logger.info(f"🎯 多头止盈详细分析:")
                self.logger.info(f"   ├─ 入场价格: ${self.entry_price:.4f}")
                self.logger.info(f"   ├─ 当前价格: ${current_price:.4f}")
                self.logger.info(f"   ├─ 止盈价格: ${self.dynamic_take_profit:.4f}")
                self.logger.info(f"   ├─ 盈利距离: {profit_distance*100:.2f}%")
                self.logger.info(f"   ├─ 持仓时长: {self.hold_duration} 步")
                self.logger.info(f"   └─ 预期盈利: ${abs(self.position_size) * (current_price - self.entry_price):.2f}")
                
            elif self.position_type == -1 and current_price <= self.dynamic_take_profit:
                should_close = True
                pnl_pct = (self.entry_price - current_price) / self.entry_price
                close_reason = f"空头止盈触发: 当前价=${current_price:.4f} <= 止盈价=${self.dynamic_take_profit:.4f}, 盈利={pnl_pct*100:.2f}%"
                
                # 🔍 额外的止盈分析
                profit_distance = (self.entry_price - current_price) / self.entry_price
                self.logger.info(f"🎯 空头止盈详细分析:")
                self.logger.info(f"   ├─ 入场价格: ${self.entry_price:.4f}")
                self.logger.info(f"   ├─ 当前价格: ${current_price:.4f}")
                self.logger.info(f"   ├─ 止盈价格: ${self.dynamic_take_profit:.4f}")
                self.logger.info(f"   ├─ 盈利距离: {profit_distance*100:.2f}%")
                self.logger.info(f"   ├─ 持仓时长: {self.hold_duration} 步")
                self.logger.info(f"   └─ 预期盈利: ${abs(self.position_size) * (self.entry_price - current_price):.2f}")
        
        if should_close:
            self.logger.info(f"🎯 自动止盈止损执行: {close_reason}")
            self._close_position(current_price, self.current_step + self.lookback_window)
            return True
        
        return False

    def _log_trade(self, action: str, price: float, size: float, pnl: float = None, **kwargs):
        """记录交易日志"""
        trade = {
            'action': action,
            'price': price,
            'size': size,
            'pnl': pnl,
            'leverage': kwargs.get('leverage'),
            'stop_loss': kwargs.get('stop_loss'),
            'take_profit': kwargs.get('take_profit'),
            'hold_duration': kwargs.get('hold_duration'),
            'pnl_pct': kwargs.get('pnl_pct')
        }
        self.trade_history.append(trade)
    
    def _update_portfolio_value(self):
        """✅ 修复组合价值计算 - 基于杠杆正确计算"""
        current_price = self.df['close'].iloc[self.current_step + self.lookback_window]
        
        if self.position_type != 0:
            # 计算未实现盈亏
            if self.position_type == 1:  # 多头
                position_pnl = self.position_size * (current_price - self.entry_price)
            else:  # 空头
                position_pnl = abs(self.position_size) * (self.entry_price - current_price)
            
            # ✅ 修复：组合价值 = 余额 + 未实现盈亏
            # 余额中已经扣除了保证金，所以直接加未实现盈亏即可
            self.portfolio_value = self.balance + position_pnl
            
            # 安全检查
            if not np.isfinite(self.portfolio_value) or self.portfolio_value < 0:
                self.portfolio_value = max(0, self.balance)
            
            # 防止异常增长
            if self.portfolio_value > self.initial_balance * 50:
                self.portfolio_value = self.initial_balance * 50
        else:
            self.portfolio_value = self.balance
        
        # 更新最大值和回撤
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        # 数据结束
        if self.current_step >= self.total_steps - 1:
            return True
        
        # 破产
        if self.portfolio_value <= self.initial_balance * 0.1:
            return True
        
        # 最大回撤限制
        if self.max_drawdown > self.config.get('MAX_DRAWDOWN', 0.2):
            return True
        
        return False
    
    def _check_risk_limits(self) -> bool:
        """检查风险限制"""
        # 单日最大亏损
        daily_loss_pct = (self.initial_balance - self.portfolio_value) / self.initial_balance
        if daily_loss_pct > self.config.get('MAX_DAILY_LOSS', 0.05):
            return True
        
        # 连续亏损次数
        if self.consecutive_losses > 5:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """✅ 优化的观测函数 - 支持滑动窗口和改进的归一化"""
        features = []
        
        # 获取当前数据点
        current_idx = self.current_step + self.lookback_window - 1
        if current_idx >= len(self.df):
            current_idx = len(self.df) - 1
        
        current_data = self.df.iloc[current_idx]
        
        # 1. 核心技术指标特征（改进归一化）
        for feature_name in self.observation_features:
            if feature_name in self.df.columns:
                feature_value = current_data[feature_name]
                
                # 智能归一化处理
                if pd.isna(feature_value):
                    normalized_value = 0.0
                else:
                    normalized_value = self._normalize_feature(feature_name, feature_value, current_idx)
                
                features.append(float(normalized_value))
                
            elif feature_name in ['position_type', 'position_size_normalized', 'hold_duration_normalized',
                                  'unrealized_pnl_normalized', 'portfolio_value_normalized',
                                  'current_leverage', 'signal_confluence', 'trend_strength', 'volatility_regime']:
                # 动态特征，需要计算
                dynamic_value = self._get_dynamic_feature_value(feature_name, current_data, current_idx)
                features.append(float(dynamic_value))
            else:
                # 不存在的特征，填充0
                features.append(0.0)
        
        # 2. 滑动窗口时序特征
        windowed_features = self._get_windowed_features()
        features.extend(windowed_features)
        
        # 3. 持仓状态特征（增强版）
        position_state_features = self._get_position_state_features()
        features.extend(position_state_features)
        
        # 确保特征向量长度一致
        expected_length = len(self.observation_features) + len(windowed_features) + len(position_state_features)
        actual_expected_length = self.observation_space.shape[0]  # 使用观测空间定义的维度
        
        # ✅ 修复：使用观测空间定义的维度作为标准
        if len(features) != actual_expected_length:
            # 调整到正确长度
            if len(features) < actual_expected_length:
                features.extend([0.0] * (actual_expected_length - len(features)))
                self.logger.debug(f"🔧 观测向量长度不足，填充零值: {len(features)} -> {actual_expected_length}")
            else:
                features = features[:actual_expected_length]
                self.logger.debug(f"🔧 观测向量长度过长，截断: {len(features)} -> {actual_expected_length}")
        
        observation = np.array(features, dtype=np.float32)
        
        # ✅ 添加维度检查日志（仅在调试模式）
        if self.current_step < 5:  # 只在前5步记录，避免日志过多
            self.logger.debug(f"🔍 观测向量构成: 核心特征={len(self.observation_features)}, "
                            f"滑动窗口={len(windowed_features)}, 持仓状态={len(position_state_features)}, "
                            f"总计={len(features)}, 期望={actual_expected_length}")
        
        # 最终数值检查和处理
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -10.0, 10.0)  # 防止极值
        
        return observation
    
    def _normalize_feature(self, feature_name: str, feature_value: float, current_idx: int) -> float:
        """智能特征归一化"""
        try:
            # 已经归一化的特征
            if feature_name in ['rsi_normalized', 'bb_position', 'close_position_in_range']:
                return np.clip(feature_value, 0.0, 1.0)
            
            # 相对百分比特征
            elif feature_name in ['price_vs_ema_fast', 'price_vs_ema_slow', 'price_vs_vwap', 
                                'price_change_5', 'price_change_10']:
                return np.clip(feature_value, -0.2, 0.2)  # 限制在±20%
            
            # 波动率特征
            elif feature_name in ['atr_normalized', 'bb_width']:
                return np.clip(feature_value, 0.0, 0.2)  # 限制在20%以内
            
            # 成交量比率
            elif feature_name == 'volume_ratio':
                return np.clip(np.log1p(feature_value) / 3.0, -1.0, 2.0)  # 对数归一化
            
            # MACD归一化
            elif feature_name in ['macd_normalized', 'macd_histogram']:
                return np.clip(feature_value * 10000, -5.0, 5.0)  # 放大并限制
            
            # OBV归一化（已经在技术指标中处理）
            elif feature_name == 'obv_normalized':
                return np.clip(feature_value, -1.0, 1.0)
            
            # 分类特征
            elif feature_name in ['ema_cross_signal', 'po3_phase', 'market_structure']:
                return float(feature_value)  # 保持原值
            
            # 布尔特征
            elif feature_name in ['bos_bullish', 'bos_bearish']:
                return float(feature_value)
            
            # SMC信号
            elif feature_name in ['smc_signal', 'order_block_signal', 'liquidity_sweep_signal']:
                return np.clip(feature_value, -2.0, 2.0)
            
            # 默认归一化
            else:
                # 使用滚动标准化
                if current_idx > 50:
                    recent_values = self.df[feature_name].iloc[max(0, current_idx-50):current_idx+1]
                    mean_val = recent_values.mean()
                    std_val = recent_values.std()
                    if std_val > 0:
                        normalized = (feature_value - mean_val) / std_val
                        return np.clip(normalized, -3.0, 3.0)
                
                return np.clip(feature_value, -10.0, 10.0)
                
        except Exception as e:
            self.logger.error(f"❌ 特征 {feature_name} 归一化失败: {e}")
            return 0.0
    
    def _get_dynamic_feature_value(self, feature_name: str, current_data: pd.Series, current_idx: int) -> float:
        """获取动态特征值"""
        try:
            if feature_name == 'position_type':
                return float(self.position_type)
            
            elif feature_name == 'position_size_normalized':
                # 相对于初始余额的仓位大小
                max_position_value = self.initial_balance * 0.2  # 假设最大仓位是20%
                position_value = abs(self.position_size * current_data.get('close', 1))
                return np.clip(position_value / max_position_value, 0.0, 1.0)
            
            elif feature_name == 'hold_duration_normalized':
                if self.position_type == 0:
                    return 0.0
                max_hold_duration = 100  # 最大持仓100步
                hold_duration = self.current_step - self.entry_step
                return np.clip(hold_duration / max_hold_duration, 0.0, 1.0)
            
            elif feature_name == 'unrealized_pnl_normalized':
                if self.position_type == 0:
                    return 0.0
                # 未实现盈亏相对于入场价值的比例
                entry_value = abs(self.position_size * self.entry_price)
                if entry_value > 0:
                    pnl_ratio = self.unrealized_pnl / entry_value
                    return np.clip(pnl_ratio, -1.0, 1.0)
                return 0.0
            
            elif feature_name == 'portfolio_value_normalized':
                # 投资组合价值相对于初始资金的对数变化
                if self.portfolio_value > 0 and self.initial_balance > 0:
                    ratio = self.portfolio_value / self.initial_balance
                    return np.clip(np.log(ratio), -1.0, 1.0)
                return 0.0
            
            elif feature_name == 'current_leverage':
                return np.clip(self.current_leverage / 5.0, 0.0, 1.0)  # 最大5倍杠杆
            
            elif feature_name == 'signal_confluence':
                return np.clip(self._calculate_signal_confluence(current_idx), 0.0, 1.0)
            
            elif feature_name == 'trend_strength':
                return current_data.get('trend_strength', 0.0)
            
            elif feature_name == 'volatility_regime':
                return np.clip(current_data.get('volatility_regime', 1.0) / 2.0, 0.0, 1.0)
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ 动态特征 {feature_name} 计算失败: {e}")
            return 0.0
    
    def _get_position_state_features(self) -> List[float]:
        """获取持仓状态特征"""
        position_features = []
        
        try:
            # 1. 持仓方向 (One-hot编码)
            position_features.extend([
                1.0 if self.position_type == 1 else 0.0,   # 多头
                1.0 if self.position_type == -1 else 0.0,  # 空头
                1.0 if self.position_type == 0 else 0.0    # 空仓
            ])
            
            # 2. 持仓时长归一化
            if self.position_type != 0:
                hold_duration = (self.current_step - self.entry_step) / 50.0  # 50步为基准
                position_features.append(np.clip(hold_duration, 0.0, 2.0))
            else:
                position_features.append(0.0)
            
            # 3. 未实现盈亏状态
            if self.position_type != 0 and self.entry_price > 0:
                current_idx = self.current_step + self.lookback_window - 1
                if current_idx < len(self.df):
                    current_price = self.df['close'].iloc[current_idx]
                    if self.position_type == 1:  # 多头
                        pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:  # 空头
                        pnl_pct = (self.entry_price - current_price) / self.entry_price
                    position_features.append(np.clip(pnl_pct, -0.5, 0.5))
                else:
                    position_features.append(0.0)
            else:
                position_features.append(0.0)
            
            # 4. 连续交易状态
            steps_since_last_trade = self.current_step - self.last_trade_step
            position_features.append(np.clip(steps_since_last_trade / 20.0, 0.0, 2.0))
            
            # 5. 风险水平指示
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            position_features.append(np.clip(current_drawdown, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ 获取持仓状态特征失败: {e}")
            position_features = [0.0] * 7  # ✅ 修复：确保返回7个特征
        
        # ✅ 确保特征数量正确
        while len(position_features) < 7:
            position_features.append(0.0)
        if len(position_features) > 7:
            position_features = position_features[:7]
        
        return position_features
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_fees': self.total_fees,
            'position_type': self.position_type,
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl,
            'current_step': self.current_step
        }
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Total Return: {info['total_return']:.2%}")
            print(f"Max Drawdown: {info['max_drawdown']:.2%}")
            print(f"Total Trades: {info['total_trades']}")
            print(f"Win Rate: {info['win_rate']:.2%}")
            print(f"Position: {info['position_type']} ({info['position_size']:.4f})")
            print("-" * 50)
    
    def get_trade_summary(self) -> Dict:
        """获取交易总结 - 增加新奖励函数统计"""
        summary = {
            'trade_history': self.trade_history,
            'portfolio_history': self.portfolio_history,
            'action_history': self.action_history,
            'reward_breakdown_history': self.reward_breakdown_history,
            'final_portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_fees': self.total_fees,
            'avg_leverage': self._calculate_avg_leverage(),
            'max_leverage_used': self._calculate_max_leverage(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'profit_factor': self._calculate_profit_factor(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses()
        }
        
        # ✅ 添加平衡奖励函数统计
        if self.use_balanced_reward and self.balanced_reward_function:
            reward_summary = self.balanced_reward_function.get_performance_summary()
            summary['balanced_reward_stats'] = reward_summary
            summary['reward_parameters'] = reward_summary.get('current_parameters', {})
        
        return summary
    
    def _calculate_avg_leverage(self) -> float:
        """安全计算平均杠杆"""
        if not self.trade_history:
            return 1.0
        
        leverages = []
        for trade in self.trade_history:
            leverage = trade.get('leverage', 1.0)
            if leverage is None:
                leverage = 1.0
            leverages.append(leverage)
        
        return np.mean(leverages) if leverages else 1.0
    
    def _calculate_max_leverage(self) -> float:
        """安全计算最大杠杆"""
        if not self.trade_history:
            return 1.0
        
        leverages = []
        for trade in self.trade_history:
            leverage = trade.get('leverage', 1.0)
            if leverage is None:
                leverage = 1.0
            leverages.append(leverage)
        
        return max(leverages) if leverages else 1.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # 年化
    
    def _calculate_sortino_ratio(self) -> float:
        """计算索提诺比率"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        return np.mean(returns) / (np.std(negative_returns) + 1e-8) * np.sqrt(252)  # 年化
    
    def _calculate_profit_factor(self) -> float:
        """计算盈利因子"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = 0
        gross_loss = 0
        
        for trade in self.trade_history:
            pnl = trade.get('pnl', 0)
            if pnl is None:
                pnl = 0
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)
        
        return gross_profit / max(gross_loss, 1e-8)
    
    def _calculate_max_consecutive_losses(self) -> int:
        """计算最大连续亏损次数"""
        if not self.trade_history:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trade_history:
            pnl = trade.get('pnl', 0)
            if pnl is None:
                pnl = 0
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def get_detailed_performance_metrics(self) -> Dict:
        """获取详细性能指标"""
        metrics = {
            'total_return': (self.portfolio_value - self.config.get('INITIAL_BALANCE', 10000)) / self.config.get('INITIAL_BALANCE', 10000),
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade_pnl': self.unrealized_pnl / max(self.total_trades, 1),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            
            # ✅ 新增指标
            'trade_frequency_penalty': 0.0,
            'hold_duration_penalty': 0.0,
            'avg_leverage': self._calculate_avg_leverage(),
            'max_leverage_used': self._calculate_max_leverage(),
            'position_manager_stats': {
                'consecutive_losses': self.consecutive_losses,
                'daily_pnl': self.unrealized_pnl,
                'max_balance': self.max_portfolio_value
            }
        }
        
        return metrics

    def _calculate_leverage_efficiency(self, current_idx: int) -> float:
        """计算杠杆使用效率"""
        if self.position_type == 0:
            return 0.0
        
        current_data = self.df.iloc[current_idx]
        signal_strength = abs(current_data.get('smc_signal', 0))
        volatility = current_data.get('atr_normalized', 0.5)
        
        # 理想杠杆：基于信号强度和波动率
        ideal_leverage = self.position_manager.calculate_dynamic_leverage(
            signal_strength=signal_strength,
            market_regime=self._get_market_regime(current_idx),
            volatility=volatility,
            risk_score=self._calculate_current_risk_score()
        )
        
        # 计算当前杠杆与理想杠杆的差异
        leverage_diff = abs(self.current_leverage - ideal_leverage)
        efficiency = max(0, 1.0 - leverage_diff / ideal_leverage)
        
        return efficiency
    
    def _get_market_regime(self, current_idx: int) -> str:
        """判断市场状态"""
        current_data = self.df.iloc[current_idx]
        
        # 基于ATR和趋势强度判断
        atr_normalized = current_data.get('atr_normalized', 0.5)
        market_structure = current_data.get('market_structure', 0)
        
        if atr_normalized > 0.8:
            return 'volatile'
        elif abs(market_structure) > 0.5:
            return 'trending'
        else:
            return 'ranging'
    
    def _calculate_current_risk_score(self) -> float:
        """计算当前风险评分"""
        # 基于回撤、连续亏损等计算风险评分
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # 连续亏损风险
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            pnl = trade.get('pnl', 0)
            # 确保pnl是数字类型，如果是None则视为0
            if pnl is None:
                pnl = 0
            if pnl < 0:
                consecutive_losses += 1
            else:
                break
        
        risk_score = min(1.0, current_drawdown * 2 + consecutive_losses * 0.1)
        return risk_score

    def _calculate_risk_penalty(self, current_idx: int) -> float:
        """计算风险惩罚"""
        # 实现风险惩罚逻辑
        return 0.0  # 临时返回，需要根据实际风险管理逻辑实现

    def _check_enhanced_risk_control(self, current_price: float, current_idx: int) -> Tuple[bool, str, Dict]:
        """
        ✅ 增强风控检查 - 集成新的风控机制
        
        Returns:
            (should_exit, exit_reason, exit_info)
        """
        if self.position_size == 0 or self.position_risk_state is None:
            return False, "", {}
        
        exit_info = {}
        
        # 1. ✅ 基础止损检查（原有逻辑保持兼容性）
        if self.position_risk_state.stop_loss_price > 0:
            if ((self.position_type == 1 and current_price <= self.position_risk_state.stop_loss_price) or
                (self.position_type == -1 and current_price >= self.position_risk_state.stop_loss_price)):
                
                exit_info = {
                    'exit_type': 'stop_loss',
                    'exit_price': current_price,
                    'stop_loss_price': self.position_risk_state.stop_loss_price,
                    'stop_loss_type': self.stop_loss_type.value
                }
                return True, "enhanced_stop_loss", exit_info
        
        # 2. ✅ 基础止盈检查
        if self.position_risk_state.take_profit_price > 0:
            if ((self.position_type == 1 and current_price >= self.position_risk_state.take_profit_price) or
                (self.position_type == -1 and current_price <= self.position_risk_state.take_profit_price)):
                
                exit_info = {
                    'exit_type': 'take_profit',
                    'exit_price': current_price,
                    'take_profit_price': self.position_risk_state.take_profit_price,
                    'take_profit_type': self.take_profit_type.value
                }
                return True, "enhanced_take_profit", exit_info
        
        # 3. ✅ 移动止盈检查
        if self.enable_trailing_stop:
            # 更新移动止盈状态
            self.risk_controller.setup_trailing_take_profit(self.position_risk_state, current_price)
            
            # 检查移动止损触发
            if self.risk_controller.update_trailing_stop(self.position_risk_state, current_price):
                exit_info = {
                    'exit_type': 'trailing_stop',
                    'exit_price': current_price,
                    'trailing_stop_price': self.position_risk_state.trailing_stop_price,
                    'highest_profit': self.position_risk_state.highest_profit
                }
                return True, "trailing_stop", exit_info
        
        # 4. ✅ 分批止盈检查
        if self.enable_partial_profit:
            partial_executed, close_ratio, level = self.risk_controller.execute_partial_take_profit(
                self.position_risk_state, current_price)
            
            if partial_executed:
                # 执行部分平仓
                self._execute_partial_close(close_ratio, current_price, level)
                
                exit_info = {
                    'exit_type': 'partial_profit',
                    'exit_price': current_price,
                    'close_ratio': close_ratio,
                    'profit_level': level,
                    'remaining_position': self.position_risk_state.remaining_position_ratio
                }
                
                # 如果仓位全部平完，则退出
                if self.position_risk_state.remaining_position_ratio <= 0:
                    return True, "partial_profit_complete", exit_info
                else:
                    # 记录分批平仓事件但不退出
                    self._log_risk_event("partial_profit", exit_info)
                    return False, "", {}
        
        return False, "", {}
    
    def _update_trailing_stop(self, current_price: float):
        """更新移动止损线"""
        if not self.trailing_stop_active or self.position_size == 0:
            return
        
        # 激活移动止损的条件检查
        current_pnl_pct = self._calculate_current_pnl_pct(current_price)
        if not self.trailing_stop_active and current_pnl_pct >= self.trailing_stop_activation:
            self.trailing_stop_active = True
            self.logger.info(f"🔄 移动止损激活: 当前盈利={current_pnl_pct*100:.2f}%, 激活阈值={self.trailing_stop_activation*100:.1f}%")
        
        if self.trailing_stop_active:
            # 计算新的移动止损线
            if self.position_type == 1:  # 多头
                new_trailing_stop = current_price * (1 - self.trailing_stop_distance)
                if new_trailing_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing_stop
                    self.logger.debug(f"📈 多头移动止损更新: {new_trailing_stop:.4f}")
            else:  # 空头
                new_trailing_stop = current_price * (1 + self.trailing_stop_distance)
                if new_trailing_stop < self.trailing_stop_price or self.trailing_stop_price == 0:
                    self.trailing_stop_price = new_trailing_stop
                    self.logger.debug(f"📉 空头移动止损更新: {new_trailing_stop:.4f}")
    
    def _set_initial_stop_loss_take_profit(self, entry_price: float, current_idx: int):
        """设置初始止盈止损价位"""
        if self.enable_dynamic_sl_tp:
            # 使用动态ATR计算
            self.stop_loss_price = self.compute_dynamic_stop_loss(entry_price, self.position_type, current_idx)
            self.take_profit_price = self.compute_dynamic_take_profit(entry_price, self.position_type, current_idx)
        else:
            # 使用固定比例
            if self.position_type == 1:  # 多头
                self.stop_loss_price = entry_price * (1 - self.base_stop_loss)
                self.take_profit_price = entry_price * (1 + self.base_take_profit)
            else:  # 空头
                self.stop_loss_price = entry_price * (1 + self.base_stop_loss)
                self.take_profit_price = entry_price * (1 - self.base_take_profit)
        
        # ✅ 风险收益比检查
        if self.risk_config.get('RISK_REWARD_ADJUSTMENT', False):
            self._adjust_risk_reward_ratio(entry_price)
        
        # 初始化移动止损
        if self.enable_trailing_stop:
            self.trailing_stop_price = self.stop_loss_price
            self.trailing_stop_active = False
            self.highest_profit = 0.0
        
        # 记录设置
        self.position_entry_time = self.current_step
        self.last_stop_loss_update = self.current_step
        
        self.logger.info(f"🎯 设置止盈止损: 入场={entry_price:.4f}, 止损={self.stop_loss_price:.4f}, 止盈={self.take_profit_price:.4f}")
    
    def _adjust_risk_reward_ratio(self, entry_price: float):
        """调整风险收益比"""
        if self.position_type == 1:  # 多头
            risk = entry_price - self.stop_loss_price
            reward = self.take_profit_price - entry_price
        else:  # 空头
            risk = self.stop_loss_price - entry_price
            reward = entry_price - self.take_profit_price
        
        if risk > 0:
            current_ratio = reward / risk
            if current_ratio < self.min_risk_reward_ratio:
                # 调整止盈线以满足最小风险收益比
                if self.position_type == 1:
                    self.take_profit_price = entry_price + (risk * self.min_risk_reward_ratio)
                else:
                    self.take_profit_price = entry_price - (risk * self.min_risk_reward_ratio)
                
                self.logger.info(f"📊 风险收益比调整: 目标比例={self.min_risk_reward_ratio:.1f}, 新止盈={self.take_profit_price:.4f}")
    
    def _log_risk_event(self, event_type: str, details: Dict):
        """记录风控事件"""
        risk_event = {
            'step': self.current_step,
            'event_type': event_type,
            'timestamp': self.current_step,
            'details': details
        }
        self.risk_events_history.append(risk_event)
        
        # 只保留最近的100个风控事件
        if len(self.risk_events_history) > 100:
            self.risk_events_history.pop(0)

    def _calculate_enhanced_stop_loss(self, entry_price: float, position_type: int, current_idx: int) -> float:
        """
        ✅ 增强止损计算 - 根据配置选择不同策略
        """
        try:
            if self.stop_loss_type == StopLossType.ATR_ADAPTIVE:
                return self.risk_controller.calculate_atr_adaptive_stop_loss(
                    self.df, current_idx, entry_price, position_type)
                    
            elif self.stop_loss_type == StopLossType.VOLATILITY_PERCENTAGE:
                return self.risk_controller.calculate_volatility_adaptive_stop_loss(
                    self.df, current_idx, entry_price, position_type)
                    
            elif self.stop_loss_type == StopLossType.TECHNICAL_LEVEL:
                stop_price, _ = self.risk_controller.calculate_technical_level_stop_loss(
                    self.df, current_idx, entry_price, position_type)
                return stop_price
                
            elif self.stop_loss_type == StopLossType.HYBRID:
                stop_price, _ = self.risk_controller.calculate_hybrid_stop_loss(
                    self.df, current_idx, entry_price, position_type)
                return stop_price
                
            else:  # FIXED_PERCENTAGE
                fallback_pct = 0.025  # 2.5%
                if position_type == 1:
                    return entry_price * (1 - fallback_pct)
                else:
                    return entry_price * (1 + fallback_pct)
                    
        except Exception as e:
            self.logger.error(f"❌ 增强止损计算失败: {e}")
            # 回退到固定比例
            fallback_pct = 0.025
            if position_type == 1:
                return entry_price * (1 - fallback_pct)
            else:
                return entry_price * (1 + fallback_pct)
    
    def _calculate_enhanced_take_profit(self, entry_price: float, position_type: int, current_idx: int) -> float:
        """
        ✅ 增强止盈计算 - 根据配置选择不同策略
        """
        try:
            if self.take_profit_type in [TakeProfitType.DYNAMIC_ATR, TakeProfitType.TECHNICAL_TARGET]:
                return self.risk_controller.calculate_dynamic_take_profit(
                    self.df, current_idx, entry_price, position_type)
            else:
                # 默认固定比例止盈（移动止盈和分批止盈在运行时处理）
                fallback_pct = 0.03  # 3%
                if position_type == 1:
                    return entry_price * (1 + fallback_pct)
                else:
                    return entry_price * (1 - fallback_pct)
                    
        except Exception as e:
            self.logger.error(f"❌ 增强止盈计算失败: {e}")
            fallback_pct = 0.03
            if position_type == 1:
                return entry_price * (1 + fallback_pct)
            else:
                return entry_price * (1 - fallback_pct)

    def _execute_partial_close(self, close_ratio: float, current_price: float, level: int):
        """
        ✅ 执行分批平仓
        """
        try:
            # 计算需要平仓的数量
            close_size = abs(self.position_size) * close_ratio
            
            # 计算分批平仓的盈亏
            if self.position_type == 1:  # 多头
                partial_pnl = close_size * (current_price - self.entry_price)
            else:  # 空头  
                partial_pnl = close_size * (self.entry_price - current_price)
            
            # 计算费用
            close_value = close_size * current_price
            fee = close_value * self.commission
            net_partial_pnl = partial_pnl - fee
            
            # 更新余额
            initial_margin = close_value / self.current_leverage
            self.balance += net_partial_pnl
            
            # 更新仓位大小
            if self.position_type == 1:
                self.position_size -= close_size
            else:
                self.position_size += close_size
            
            # 更新统计
            self.total_trades += 1
            if net_partial_pnl > 0:
                self.winning_trades += 1
            
            self.total_fees += fee
            
            # 记录分批平仓
            self._log_trade(f"分批平仓-{level+1}", current_price, close_size, 
                          pnl=net_partial_pnl, pnl_pct=net_partial_pnl/initial_margin)
            
            self.logger.info(f"💰 分批平仓执行: 级别{level+1}, 平仓={close_ratio:.1%}, "
                           f"盈亏={net_partial_pnl:.2f}, 剩余仓位={abs(self.position_size):.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ 分批平仓执行失败: {e}")

def make_env(df: pd.DataFrame = None, mode: str = 'train', **kwargs):
    """创建环境的工厂函数"""
    return SolUsdtTradingEnv(df=df, mode=mode, **kwargs)

def main():
    """主函数，用于测试交易环境"""
    # 创建环境
    env = SolUsdtTradingEnv()
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"特征数量: {len(env.observation_features)}")
    print(f"数据点数量: {env.total_steps}")
    
    # 简单测试
    obs, info = env.reset()
    print(f"初始观察维度: {obs.shape}")
    
    # 随机测试几步
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {i}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            print("Episode finished!")
            break
    
    # 显示交易总结
    summary = env.get_trade_summary()
    print(f"\n交易总结:")
    print(f"总收益率: {summary['total_return']:.2%}")
    print(f"最大回撤: {summary['max_drawdown']:.2%}")
    print(f"交易次数: {summary['total_trades']}")
    print(f"胜率: {summary['win_rate']:.2%}")
    print(f"夏普比率: {summary['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    main() 