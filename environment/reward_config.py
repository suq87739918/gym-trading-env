# -*- coding: utf-8 -*-
"""
增强版Reward函数配置文件
提供不同策略导向的reward参数配置
"""

# 基础配置 - 平衡收益和胜率
BALANCED_REWARD_CONFIG = {
    # 基础PnL奖励配置
    'pnl_scale_factor': 100,  # PnL放大倍数
    
    # 胜负附加奖励配置 - 平衡设计
    'win_bonus_large': 2.0,   # 大盈利奖励（>5%）
    'win_bonus_medium': 1.0,  # 中盈利奖励（2-5%）
    'win_bonus_small': 0.5,   # 小盈利奖励（0-2%）
    'loss_penalty_large': -3.0,   # 大亏损惩罚（>5%）
    'loss_penalty_medium': -1.5,  # 中亏损惩罚（2-5%）
    'loss_penalty_small': -0.8,   # 小亏损惩罚（0-2%）
    
    # 连胜奖励配置
    'consecutive_win_bonus': 0.2,  # 每次连胜的额外奖励
    'max_consecutive_bonus': 1.0,  # 连胜奖励上限
    
    # 风险调整配置
    'risk_adjustment_strength': 0.5,  # 风险调整强度
    'volatility_penalty_high': -0.5,  # 高波动率惩罚
    'volatility_penalty_medium': -0.2, # 中波动率惩罚
    'drawdown_penalty_high': -2.0,    # 高回撤惩罚倍数
    'drawdown_penalty_medium': -1.0,  # 中回撤惩罚倍数
    
    # 趋势对齐奖励配置
    'strong_trend_bonus': 0.5,     # 强趋势对齐奖励
    'weak_trend_bonus': 0.2,       # 弱趋势对齐奖励
    'counter_trend_penalty': -0.5, # 逆势惩罚
    
    # 信号质量奖励配置
    'high_quality_bonus': 0.6,     # 高质量信号奖励
    'low_quality_penalty': -0.4,   # 低质量信号惩罚
    
    # 时间相关惩罚配置
    'time_penalty_base': -0.001,    # 基础时间惩罚
    'holding_inefficiency_penalty': -0.5, # 无效长持仓惩罚
    
    # 组合表现奖励配置
    'sharpe_ratio_bonus_scale': 2.0,  # 夏普比率奖励倍数
    'win_rate_bonus_scale': 2.0,      # 胜率奖励倍数
    'return_bonus_scale': 2.0,        # 收益奖励倍数
    
    # 结构识别奖励配置
    'structure_signal_bonus': 0.4,     # 结构信号奖励
    'structure_indicator_bonus': 0.3,  # 技术指标结构奖励
    'reasonable_profit_bonus': 0.2,    # 合理盈利奖励
    'excellent_profit_bonus': 0.5,     # 优秀盈利奖励
}

# 高胜率导向配置 - 强化胜负奖励
HIGH_WINRATE_REWARD_CONFIG = {
    **BALANCED_REWARD_CONFIG,
    
    # 强化胜负导向
    'win_bonus_large': 3.0,      # 增强大盈利奖励
    'win_bonus_medium': 1.8,     # 增强中盈利奖励
    'win_bonus_small': 1.0,      # 增强小盈利奖励
    'loss_penalty_large': -5.0,  # 加重大亏损惩罚
    'loss_penalty_medium': -3.0, # 加重中亏损惩罚
    'loss_penalty_small': -1.5,  # 加重小亏损惩罚
    
    # 增强连胜奖励
    'consecutive_win_bonus': 0.3,
    'max_consecutive_bonus': 1.5,
    
    # 更强的风险控制
    'risk_adjustment_strength': 0.6,
    'volatility_penalty_high': -0.8,
    'volatility_penalty_medium': -0.4,
    
    # 更严格的信号质量要求
    'high_quality_bonus': 0.8,
    'low_quality_penalty': -0.6,
}

# 高收益导向配置 - 强化收益幅度
HIGH_RETURN_REWARD_CONFIG = {
    **BALANCED_REWARD_CONFIG,
    
    # 基础PnL权重增加
    'pnl_scale_factor': 150,
    
    # 收益幅度导向奖励
    'win_bonus_large': 3.5,      # 大幅增强大盈利奖励
    'win_bonus_medium': 1.2,     # 略微增强中盈利奖励
    'win_bonus_small': 0.3,      # 降低小盈利奖励（鼓励更大盈利）
    'loss_penalty_large': -4.0,  # 适度增加大亏损惩罚
    'loss_penalty_medium': -2.0, # 适度增加中亏损惩罚
    'loss_penalty_small': -0.5,  # 降低小亏损惩罚
    
    # 优秀盈利结构奖励增强
    'excellent_profit_bonus': 1.0,  # 翻倍优秀盈利奖励
    'reasonable_profit_bonus': 0.1, # 降低一般盈利奖励
    
    # 组合表现奖励增强
    'return_bonus_scale': 3.0,      # 增强收益奖励
    'sharpe_ratio_bonus_scale': 1.5, # 降低夏普比率权重
}

# 稳健导向配置 - 强化风险控制
CONSERVATIVE_REWARD_CONFIG = {
    **BALANCED_REWARD_CONFIG,
    
    # 温和的胜负奖励
    'win_bonus_large': 1.5,
    'win_bonus_medium': 0.8,
    'win_bonus_small': 0.4,
    'loss_penalty_large': -2.0,  # 降低亏损惩罚，避免过度保守
    'loss_penalty_medium': -1.0,
    'loss_penalty_small': -0.5,
    
    # 强化风险调整
    'risk_adjustment_strength': 0.8,  # 更强的风险调整
    'volatility_penalty_high': -1.0,
    'volatility_penalty_medium': -0.5,
    'drawdown_penalty_high': -3.0,
    'drawdown_penalty_medium': -1.5,
    
    # 强化稳定性奖励
    'sharpe_ratio_bonus_scale': 3.0,  # 夏普比率权重增加
    'win_rate_bonus_scale': 2.5,      # 胜率权重增加
    'return_bonus_scale': 1.5,        # 收益权重适度降低
    
    # 强化信号质量要求
    'high_quality_bonus': 0.8,
    'low_quality_penalty': -0.8,  # 更严格的信号质量惩罚
    
    # 更强的时间控制
    'time_penalty_base': -0.002,
    'holding_inefficiency_penalty': -1.0,
}

# 激进配置 - 追求极致收益
AGGRESSIVE_REWARD_CONFIG = {
    **BALANCED_REWARD_CONFIG,
    
    # 极大化盈利奖励
    'pnl_scale_factor': 200,
    'win_bonus_large': 5.0,      # 极高的大盈利奖励
    'win_bonus_medium': 2.0,
    'win_bonus_small': 0.8,
    'loss_penalty_large': -2.0,  # 降低亏损惩罚（允许更大风险）
    'loss_penalty_medium': -1.0,
    'loss_penalty_small': -0.3,
    
    # 降低风险调整强度
    'risk_adjustment_strength': 0.3,
    'volatility_penalty_high': -0.2,  # 降低波动率惩罚
    'volatility_penalty_medium': -0.1,
    'drawdown_penalty_high': -1.0,    # 降低回撤惩罚
    'drawdown_penalty_medium': -0.5,
    
    # 极大化收益奖励
    'excellent_profit_bonus': 1.5,
    'return_bonus_scale': 4.0,
    'sharpe_ratio_bonus_scale': 1.0,  # 降低稳定性权重
}

# 预定义配置字典
REWARD_CONFIGS = {
    'balanced': BALANCED_REWARD_CONFIG,
    'high_winrate': HIGH_WINRATE_REWARD_CONFIG,
    'high_return': HIGH_RETURN_REWARD_CONFIG,
    'conservative': CONSERVATIVE_REWARD_CONFIG,
    'aggressive': AGGRESSIVE_REWARD_CONFIG,
}

def get_reward_config(config_name: str = 'balanced'):
    """
    获取指定的reward配置
    
    Args:
        config_name: 配置名称 ('balanced', 'high_winrate', 'high_return', 'conservative', 'aggressive')
    
    Returns:
        dict: reward配置参数
    """
    if config_name not in REWARD_CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(REWARD_CONFIGS.keys())}")
    
    return REWARD_CONFIGS[config_name].copy()

def print_config_comparison():
    """打印不同配置的对比"""
    print("="*80)
    print("Reward函数配置对比")
    print("="*80)
    
    key_params = [
        'pnl_scale_factor',
        'win_bonus_large', 'loss_penalty_large',
        'risk_adjustment_strength',
        'volatility_penalty_high',
        'return_bonus_scale', 'sharpe_ratio_bonus_scale'
    ]
    
    print(f"{'参数':<25}", end="")
    for config_name in REWARD_CONFIGS.keys():
        print(f"{config_name:<15}", end="")
    print()
    
    print("-" * 100)
    
    for param in key_params:
        print(f"{param:<25}", end="")
        for config_name, config in REWARD_CONFIGS.items():
            value = config.get(param, 'N/A')
            print(f"{value:<15}", end="")
        print()

if __name__ == "__main__":
    print_config_comparison()
    
    # 示例用法
    print("\n" + "="*50)
    print("示例用法:")
    print("="*50)
    
    # 获取平衡配置
    balanced_config = get_reward_config('balanced')
    print("平衡配置中的大盈利奖励:", balanced_config['win_bonus_large'])
    
    # 获取高胜率配置
    winrate_config = get_reward_config('high_winrate')
    print("高胜率配置中的大盈利奖励:", winrate_config['win_bonus_large']) 