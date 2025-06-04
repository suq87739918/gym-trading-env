"""
配置管理模块
统一管理项目中的所有配置参数
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置管理类"""
    
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置参数"""
        return {
            # Binance API 配置
            'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY', ''),
            'BINANCE_SECRET_KEY': os.getenv('BINANCE_SECRET_KEY', ''),
            'BINANCE_TESTNET': os.getenv('BINANCE_TESTNET', 'True').lower() == 'true',
            
            # 数据配置
            'SYMBOL': 'SOLUSDT',
            'DATA_START_DATE': os.getenv('DATA_START_DATE', '2023-06-02'),  # 2年前
            'DATA_END_DATE': os.getenv('DATA_END_DATE', '2025-06-02'),      # 当前时间
            'TIMEFRAME': os.getenv('TIMEFRAME', '15m'),
            'LOOKBACK_WINDOW': 50,  # 观察窗口大小
            
            # 训练测试数据划分配置
            'TRAIN_TEST_SPLIT_RATIO': float(os.getenv('TRAIN_TEST_SPLIT_RATIO', '0.8')),  # 80%训练，20%测试
            'VALIDATION_SPLIT_RATIO': float(os.getenv('VALIDATION_SPLIT_RATIO', '0.1')),  # 10%验证集
            'USE_TIME_SPLIT': os.getenv('USE_TIME_SPLIT', 'True').lower() == 'true',      # 按时间顺序划分
            'PREVENT_DATA_LEAKAGE': os.getenv('PREVENT_DATA_LEAKAGE', 'True').lower() == 'true',  # 防止数据泄露
            
            # 交易配置
            'INITIAL_BALANCE': float(os.getenv('INITIAL_BALANCE', '10000')),
            'MAX_POSITION_SIZE': float(os.getenv('MAX_POSITION_SIZE', '0.5')),
            'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', '0.20')),
            'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', '0.04')),
            'TRADING_FEE': float(os.getenv('TRADING_FEE', '0.0004')),
            'SLIPPAGE': 0.001,  # 滑点
            
            # 技术指标参数
            'RSI_PERIOD': 14,
            'BB_PERIOD': 20,
            'BB_STD': 2.0,
            'EMA_FAST': 20,
            'EMA_SLOW': 50,
            'VOLUME_MA_PERIOD': 20,
            
            # SMC 信号参数
            'PO3_LOOKBACK': 20,
            'BOS_THRESHOLD': 0.002,  # 突破阈值
            'ORDER_BLOCK_MIN_SIZE': 10,  # Order Block 最小大小
            'LIQUIDITY_THRESHOLD': 1.5,  # 流动性阈值
            
            # 强化学习配置
            'MODEL_NAME': os.getenv('MODEL_NAME', 'sol_usdt_ppo'),
            'TOTAL_TIMESTEPS': int(os.getenv('TOTAL_TIMESTEPS', '500000')),
            'LEARNING_RATE': float(os.getenv('LEARNING_RATE', '0.00025')),
            'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '512')),
            'GAMMA': 0.99,
            'CLIP_RANGE': 0.2,
            'N_EPOCHS': 10,
            'ENT_COEF': 0.01,
            'VF_COEF': 0.5,
            'MAX_GRAD_NORM': 0.5,
            
            # ✅ 新增：优化的模型配置 - LSTM和改进的PPO参数
            'MODEL_CONFIGS': {
                'PPO_LSTM': {
                    'algorithm': 'RecurrentPPO',  # 使用RecurrentPPO支持LSTM
                    'policy': 'MlpLstmPolicy',  # 修正：RecurrentPPO使用MlpLstmPolicy
                    'learning_rate': 1e-4,  # 降低学习率提高稳定性
                    'gamma': 0.95,  # 调整折扣因子
                    'gae_lambda': 0.9,  # GAE参数优化
                    'n_steps': 2048,  # 增加步数覆盖更多3小时周期
                    'batch_size': 256,  # 增大批量大小
                    'n_epochs': 4,  # 减少epoch避免过拟合
                    'ent_coef': 0.005,  # 降低熵系数减少随机性
                    'clip_range': 0.1,  # 降低裁剪范围
                    'policy_kwargs': {
                        'net_arch': [64, 64],  # 网络架构
                        'enable_critic_lstm': True,  # 启用LSTM critic
                        'lstm_hidden_size': 64,  # LSTM隐藏层大小
                    },
                    'verbose': 1,
                    'target_kl': 0.02,  # KL散度目标
                },
                'PPO_STANDARD': {
                    'policy': 'MlpPolicy',
                    'learning_rate': 1e-4,
                    'gamma': 0.95,
                    'gae_lambda': 0.9,
                    'clip_range': 0.2,
                    'n_steps': 2048,
                    'batch_size': 256,
                    'n_epochs': 10,
                    'ent_coef': 0.005,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'policy_kwargs': {
                        'net_arch': [128, 128, 64]  # 增强的网络架构
                    },
                    'verbose': 1,
                    'device': 'auto'
                },
                'DQN': {
                    'learning_rate': 1e-4,
                    'buffer_size': 100000,        # 经验回放缓冲区大小
                    'learning_starts': 50000,     # 开始学习的步数
                    'batch_size': 128,            # DQN批次大小
                    'tau': 1.0,                   # 软更新参数
                    'gamma': 0.95,
                    'train_freq': 4,              # 训练频率
                    'gradient_steps': 1,
                    'target_update_interval': 10000,  # 目标网络更新间隔
                    'exploration_fraction': 0.1,  # 探索阶段比例
                    'exploration_initial_eps': 1.0,   # 初始ε值
                    'exploration_final_eps': 0.05,    # 最终ε值
                    'policy_kwargs': {
                        'net_arch': [128, 128, 64]
                    },
                    'verbose': 1,
                    'device': 'auto'
                }
            },
            
            # ✅ 新增：训练监控配置
            'TRAINING_MONITOR': {
                'SAVE_BEST_MODEL': True,         # 保存最佳模型
                'BEST_MODEL_METRIC': 'win_rate', # 最佳模型评判标准 ('win_rate', 'total_return', 'sharpe_ratio')
                'EVAL_EPISODES': 100,            # 评估集交易次数
                'EVAL_FREQ': 25000,              # 评估频率
                'WIN_RATE_THRESHOLD': 0.55,      # 胜率阈值
                'MAX_DRAWDOWN_THRESHOLD': 0.15,  # 最大回撤阈值
                'MIN_TOTAL_RETURN': 0.10,        # 最小总收益率
                'SAVE_FREQ': 50000,              # 检查点保存频率
                'LOG_INTERVAL': 1000,            # 日志记录间隔
                'PERFORMANCE_WINDOW': 100,       # 性能统计窗口
                'MODEL_COMPARISON': True,        # 启用模型对比
            },
            
            # ✅ 新增：模型选择配置
            'ACTIVE_MODEL': os.getenv('ACTIVE_MODEL', 'PPO_LSTM'),  # 'PPO_LSTM', 'PPO_STANDARD', 'DQN'
            'ENABLE_MODEL_ENSEMBLE': False,    # 是否启用模型集成
            'MODEL_WEIGHTS': {                 # 模型权重（用于集成）
                'PPO_LSTM': 0.6,
                'PPO_STANDARD': 0.3,
                'DQN': 0.1
            },
            
            # ✅ 新增：持仓时间优化配置
            'POSITION_TIMING': {
                'TARGET_HOLD_HOURS': 3,         # 目标持仓时间（3小时）
                'MIN_HOLD_MINUTES': 15,         # 最小持仓时间（15分钟）
                'MAX_HOLD_HOURS': 12,           # 最大持仓时间（12小时）
                'EARLY_EXIT_PENALTY': 0.001,   # 过早平仓惩罚
                'OVERTIME_PENALTY': 0.0005,    # 超时持仓惩罚
                'TIME_DECAY_START': 6,          # 时间衰减开始（小时）
            },
            
            # 环境配置
            'ACTION_SPACE_SIZE': 4,  # 0: 观望, 1: 开多/加多, 2: 开空/加空, 3: 平仓
            'OBSERVATION_FEATURES': 30,  # 观察空间特征数量
            
            # 风险管理
            'MAX_DAILY_LOSS': float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            'MAX_CONCURRENT_POSITIONS': int(os.getenv('MAX_CONCURRENT_POSITIONS', '3')),
            'MAX_DRAWDOWN': 0.2,  # 最大回撤限制
            'POSITION_SIZE_SCALING': True,  # 是否启用仓位缩放
            
            # 文件路径
            'DATA_DIR': 'data/raw',
            'MODEL_DIR': 'models/saved',
            'LOG_DIR': 'logs',
            'RESULTS_DIR': 'results',
            
            # 日志配置
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            
            # 可视化配置
            'PLOT_STYLE': 'darkgrid',
            'FIGURE_SIZE': (15, 10),
            'DPI': 300,
        }
    
    def get(self, key: str, default=None):
        """获取配置项"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """批量更新配置"""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self._config.copy()
    
    def validate(self) -> bool:
        """验证关键配置项"""
        required_keys = [
            'BINANCE_API_KEY', 'BINANCE_SECRET_KEY', 'SYMBOL',
            'INITIAL_BALANCE', 'TRADING_FEE'
        ]
        
        for key in required_keys:
            if not self._config.get(key):
                print(f"警告: 缺少必要配置项 {key}")
                return False
        
        # 验证数值范围
        if self._config['INITIAL_BALANCE'] <= 0:
            print("错误: 初始余额必须大于0")
            return False
            
        if not (0 <= self._config['MAX_POSITION_SIZE'] <= 1):
            print("错误: 最大仓位比例必须在0-1之间")
            return False
            
        return True

# 全局配置实例
config = Config()

def get_config() -> Config:
    """获取配置实例"""
    return config 

# 动态杠杆配置
LEVERAGE_CONFIG = {
    'MIN_LEVERAGE': 1.0,
    'MAX_LEVERAGE': 5.0,
    'BASE_LEVERAGE': 1.0,
    'SIGNAL_STRENGTH_THRESHOLD': 0.3,
    'CONFLUENCE_BONUS_THRESHOLD': 0.5,
    'HIGH_VOLATILITY_THRESHOLD': 0.7,
    'LOW_VOLATILITY_THRESHOLD': 0.3,
    'VOLATILITY_LEVERAGE_ADJUSTMENT': 0.8
}

# 动态止盈止损风控配置
RISK_CONTROL_CONFIG = {
    # 基础止盈止损参数
    'STOP_LOSS': 0.05,          # 固定止损比例 5%
    'TAKE_PROFIT': 0.03,        # 固定止盈比例 3%
    'ENABLE_DYNAMIC_SL_TP': True, # 启用动态止盈止损
    
    # ATR动态调整参数
    'ATR_STOP_LOSS_MULTIPLIER': 2.5,  # ATR止损倍数
    'ATR_TAKE_PROFIT_MULTIPLIER': 4.0, # ATR止盈倍数
    'MIN_ATR_PERCENTAGE': 0.008,  # 最小ATR止损比例 0.8%
    'MAX_ATR_PERCENTAGE': 0.08,   # 最大ATR止损比例 8%
    
    # 移动止损参数
    'ENABLE_TRAILING_STOP': True,     # 启用移动止损
    'TRAILING_STOP_DISTANCE': 0.02,   # 移动止损距离 2%
    'TRAILING_STOP_ACTIVATION': 0.015, # 移动止损激活阈值 1.5%盈利时开始
    'TRAILING_STOP_STEP': 0.005,      # 移动止损步长 0.5%
    
    # 波动率适应性调整
    'VOLATILITY_LOW_THRESHOLD': 0.02,   # 低波动率阈值
    'VOLATILITY_HIGH_THRESHOLD': 0.06,  # 高波动率阈值
    'LOW_VOL_SL_MULTIPLIER': 0.8,      # 低波动率时止损倍数调整
    'HIGH_VOL_SL_MULTIPLIER': 1.3,     # 高波动率时止损倍数调整
    
    # 时间止损参数
    'ENABLE_TIME_STOP': True,          # 启用时间止损
    'MAX_POSITION_HOLD_HOURS': 24,     # 最大持仓时间（小时）
    'TIME_DECAY_PENALTY': 0.001,       # 时间衰减惩罚系数
    
    # 风险收益比控制
    'MIN_RISK_REWARD_RATIO': 1.2,      # 最小风险收益比
    'RISK_REWARD_ADJUSTMENT': True,     # 启用风险收益比调整
    
    # 连续亏损保护
    'MAX_CONSECUTIVE_LOSSES': 3,        # 最大连续亏损次数
    'CONSECUTIVE_LOSS_MULTIPLIER': 0.5, # 连续亏损后仓位缩减倍数
    
    # 资金管理
    'MAX_SINGLE_LOSS': 0.02,           # 单笔最大亏损比例 2%
    'DAILY_LOSS_LIMIT': 0.08,          # 日内最大亏损比例 8%
    'PORTFOLIO_HEAT': 0.1,             # 组合风险敞口 10%
    
    # 市场状态适应
    'BULL_MARKET_TP_MULTIPLIER': 1.2,  # 牛市止盈倍数
    'BEAR_MARKET_SL_MULTIPLIER': 0.9,  # 熊市止损倍数
    'RANGING_MARKET_TIGHTER_STOPS': True, # 震荡市场收紧止损
}

# 动态止盈止损配置（保持向后兼容）
STOP_LOSS_TAKE_PROFIT_CONFIG = {
    'DEFAULT_ATR_MULTIPLIER_SL': 2.5,
    'DEFAULT_ATR_MULTIPLIER_TP': 4.0,
    'MIN_ATR_PERCENTAGE': 0.008,
    'MAX_ATR_PERCENTAGE': 0.08,
    'BOLLINGER_BAND_ADJUSTMENT': 0.99,
    'USE_STRUCTURE_LEVELS': True,
    'RISK_REWARD_RATIO': 1.5
}

# 奖励函数增强配置
ENHANCED_REWARD_CONFIG = {
    'BASE_REWARD_MULTIPLIER': 1.0,
    'TREND_ALIGNMENT_WEIGHT': 0.2,
    'SIGNAL_CONFLUENCE_WEIGHT': 0.15,
    'LEVERAGE_BONUS_WEIGHT': 0.1,
    'RISK_PENALTY_WEIGHT': 0.5,
    'REVERSAL_PENALTY_WEIGHT': 0.2,
    'CONSECUTIVE_LOSS_PENALTY': 0.05,
    'PROFIT_BONUS_THRESHOLD': 0.02,
    'PNL_SENSITIVITY': 15,
    # 新增风控奖励参数
    'STOP_LOSS_TRIGGER_PENALTY': -0.5,    # 止损触发惩罚
    'TAKE_PROFIT_TRIGGER_BONUS': 0.3,     # 止盈触发奖励
    'TRAILING_STOP_BONUS': 0.1,           # 移动止损奖励
    'RISK_CONTROL_BONUS': 0.05            # 风控合规奖励
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'PLOT_STYLE': 'darkgrid',
    'FIGURE_SIZE': (15, 10),
    'DPI': 300,
    'COLOR_PALETTE': 'husl',
    'SAVE_FORMAT': 'png',
    'INTERACTIVE_PLOTS': True,
    'SHOW_GRID': True,
    'FONT_SIZE': 12
}

# 实时交易配置
LIVE_TRADING_CONFIG = {
    'MAX_DAILY_TRADES': 20,
    'MAX_DAILY_LOSS': 0.05,  # 5%
    'MIN_TRADE_INTERVAL': 300,  # 5分钟
    'WEBSOCKET_TIMEOUT': 30,
    'RECONNECT_DELAY': 5,
    'BUFFER_SIZE': 200,
    'MIN_ANALYSIS_BARS': 50,
    'ORDER_TIMEOUT': 30,
    'SLIPPAGE_TOLERANCE': 0.001  # 0.1%
}

# 风险管理配置
RISK_MANAGEMENT_CONFIG = {
    'MAX_PORTFOLIO_DRAWDOWN': 0.20,  # 20%
    'MAX_DAILY_VAR': 0.03,  # 3% 日度VaR
    'POSITION_SIZE_LIMITS': {
        'MIN_POSITION_SIZE': 0.01,
        'MAX_POSITION_SIZE': 0.15,  # 15%的组合价值
        'DEFAULT_POSITION_SIZE': 0.1
    },
    'CORRELATION_LIMIT': 0.8,  # 相关性限制
    'CONCENTRATION_LIMIT': 0.25,  # 单一资产集中度限制
    'VOLATILITY_SCALING': True,  # 基于波动率调整仓位
    'KELLY_CRITERION': False  # 是否使用Kelly公式计算仓位
}

# 性能监控配置
MONITORING_CONFIG = {
    'PERFORMANCE_UPDATE_INTERVAL': 3600,  # 1小时
    'ALERT_THRESHOLDS': {
        'DRAWDOWN_ALERT': 0.10,  # 10%回撤预警
        'LOSS_ALERT': 0.05,  # 5%亏损预警
        'CORRELATION_ALERT': 0.9,  # 高相关性预警
        'VOLATILITY_ALERT': 2.0  # 波动率异常预警
    },
    'METRICS_RETENTION_DAYS': 30,
    'LOG_LEVEL': 'INFO',
    'ENABLE_TELEGRAM_ALERTS': False,
    'ENABLE_EMAIL_ALERTS': False
}

# 数据质量配置
DATA_QUALITY_CONFIG = {
    'MAX_MISSING_DATA_PCT': 0.05,  # 最大5%缺失数据
    'OUTLIER_DETECTION_METHOD': 'iqr',  # 异常值检测方法
    'OUTLIER_THRESHOLD': 3.0,  # 异常值阈值
    'DATA_VALIDATION_ENABLED': True,
    'AUTO_CLEAN_DATA': True,
    'FORWARD_FILL_LIMIT': 3,  # 前向填充限制
    'INTERPOLATION_METHOD': 'linear'
}

# 模型配置增强
ENHANCED_MODEL_CONFIG = {
    'MODEL_TYPE': 'PPO',
    'POLICY_TYPE': 'MlpPolicy',
    'ACTIVATION_FUNCTION': 'tanh',
    'NET_ARCH': [64, 64],  # 网络架构
    'OPTIMIZER': 'adam',
    'LEARNING_RATE_SCHEDULE': 'constant',  # 学习率调度
    'USE_SDE': False,  # 是否使用状态依赖噪声
    'SDE_SAMPLE_FREQ': -1,
    'TARGET_KL': None,  # 目标KL散度
    'DEVICE': 'auto',  # 'cpu', 'cuda', 'auto'
    'SEED': 42
}

# 回测增强配置
ENHANCED_BACKTEST_CONFIG = {
    'TRANSACTION_COSTS': {
        'MAKER_FEE': 0.0002,  # 0.02%
        'TAKER_FEE': 0.0004,  # 0.04%
        'FUNDING_RATE': 0.0001,  # 资金费率
        'SLIPPAGE_MODEL': 'linear',  # 滑点模型
        'IMPACT_COST': 0.0001  # 市场冲击成本
    },
    'MARKET_CONDITIONS': {
        'CONSIDER_HOLIDAYS': True,
        'WEEKEND_TRADING': True,
        'LOW_LIQUIDITY_HOURS': [22, 23, 0, 1, 2, 3, 4, 5],  # UTC时间
        'HIGH_VOLATILITY_ADJUSTMENT': True
    },
    'BENCHMARK': {
        'USE_BENCHMARK': True,
        'BENCHMARK_SYMBOL': 'BTCUSDT',  # 基准对比
        'CORRELATION_ANALYSIS': True,
        'BETA_CALCULATION': True
    }
}

# SMC信号增强配置
ENHANCED_SMC_CONFIG = {
    'PO3_SETTINGS': {
        'PHASE_DETECTION_SENSITIVITY': 0.02,  # 2%
        'MIN_PHASE_DURATION': 5,  # 最小阶段持续期
        'TREND_CONFIRMATION_BARS': 3
    },
    'BOS_SETTINGS': {
        'MIN_STRUCTURE_SIZE': 0.01,  # 1%
        'CONFIRMATION_BARS': 2,
        'STRENGTH_CALCULATION_PERIOD': 14
    },
    'ORDER_BLOCK_SETTINGS': {
        'MIN_BLOCK_SIZE': 0.005,  # 0.5%
        'MAX_BLOCK_AGE': 50,  # 最大50根K线
        'MITIGATION_THRESHOLD': 0.8  # 80%触及认为失效
    },
    'LIQUIDITY_SETTINGS': {
        'SWEEP_DETECTION_THRESHOLD': 0.002,  # 0.2%
        'VOLUME_CONFIRMATION': True,
        'MIN_LIQUIDITY_SIZE': 1000  # 最小流动性大小
    }
}

# 获取所有配置的函数
def get_all_configs():
    """获取所有配置参数"""
    return {
        'LEVERAGE_CONFIG': LEVERAGE_CONFIG,
        'STOP_LOSS_TAKE_PROFIT_CONFIG': STOP_LOSS_TAKE_PROFIT_CONFIG,
        'ENHANCED_REWARD_CONFIG': ENHANCED_REWARD_CONFIG,
        'VISUALIZATION_CONFIG': VISUALIZATION_CONFIG,
        'LIVE_TRADING_CONFIG': LIVE_TRADING_CONFIG,
        'RISK_MANAGEMENT_CONFIG': RISK_MANAGEMENT_CONFIG,
        'MONITORING_CONFIG': MONITORING_CONFIG,
        'DATA_QUALITY_CONFIG': DATA_QUALITY_CONFIG,
        'ENHANCED_MODEL_CONFIG': ENHANCED_MODEL_CONFIG,
        'ENHANCED_BACKTEST_CONFIG': ENHANCED_BACKTEST_CONFIG,
        'ENHANCED_SMC_CONFIG': ENHANCED_SMC_CONFIG
    } 