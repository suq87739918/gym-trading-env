"""
快速开始示例脚本
演示如何使用SOL/USDT交易策略系统
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config
from data.data_collector import DataCollector
from data.technical_indicators import TechnicalIndicators
from data.smc_signals import SMCSignals
from environment.trading_env import SolUsdtTradingEnv

def generate_sample_data(days=365, freq='15T'):
    """
    生成示例数据用于演示
    
    Args:
        days: 数据天数
        freq: 数据频率 ('15T' = 15分钟)
    
    Returns:
        包含OHLCV数据的DataFrame
    """
    print("📊 生成示例数据...")
    
    # 创建时间索引
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 生成15分钟频率的时间索引
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # 设置初始价格（模拟SOL价格）
    initial_price = 100.0
    num_periods = len(dates)
    
    # 生成价格数据（使用几何布朗运动）
    np.random.seed(42)  # 固定随机种子确保可重复性
    
    # 价格变化参数
    drift = 0.001  # 漂移率
    volatility = 0.02  # 波动率
    
    # 生成收益率
    returns = np.random.normal(drift, volatility, num_periods)
    
    # 添加一些趋势和周期性
    trend = np.linspace(0, 0.5, num_periods)  # 长期上升趋势
    cyclical = 0.1 * np.sin(np.arange(num_periods) * 2 * np.pi / (4 * 24 * 7))  # 周周期
    
    returns += trend / num_periods + cyclical / num_periods
    
    # 计算价格
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # 生成OHLC数据
    data = []
    for i, price in enumerate(prices):
        # 为每个时间点生成合理的OHLC
        noise = np.random.normal(0, volatility * 0.1, 4)
        
        open_price = price + noise[0]
        close_price = price + noise[1]
        
        # 确保high >= max(open, close), low <= min(open, close)
        high_price = max(open_price, close_price) + abs(noise[2])
        low_price = min(open_price, close_price) - abs(noise[3])
        
        # 生成成交量（与价格变化相关）
        volume_base = 10000
        volume_multiplier = 1 + abs(noise[1]) * 10  # 价格变化大时成交量增加
        volume = volume_base * volume_multiplier
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df = df.round(4)  # 保留4位小数
    
    print(f"✅ 生成了 {len(df)} 条示例数据")
    print(f"📅 时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"💰 价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def demo_data_processing():
    """演示数据处理流程"""
    print("\n🔧 演示数据处理流程")
    print("=" * 40)
    
    # 生成示例数据
    df = generate_sample_data(days=90)  # 生成90天的数据
    
    # 计算技术指标
    print("\n📊 计算技术指标...")
    indicator_calculator = TechnicalIndicators()
    df_with_indicators = indicator_calculator.calculate_all_indicators(df)
    
    print(f"✅ 添加了 {len(df_with_indicators.columns) - len(df.columns)} 个技术指标")
    
    # 计算SMC信号
    print("\n🎯 计算SMC信号...")
    smc_calculator = SMCSignals()
    df_complete = smc_calculator.calculate_all_smc_signals(df_with_indicators)
    
    print(f"✅ 添加了 {len(df_complete.columns) - len(df_with_indicators.columns)} 个SMC信号")
    print(f"📈 总特征数: {len(df_complete.columns)}")
    
    # 显示一些关键指标
    print("\n📋 关键指标预览:")
    latest_data = df_complete.iloc[-1]
    
    indicators_to_show = [
        ('RSI', 'rsi'),
        ('布林带位置', 'bb_position'),
        ('EMA趋势', 'price_vs_ema_fast'),
        ('MACD信号', 'macd_signal'),
        ('SMC信号', 'smc_signal'),
        ('市场结构', 'market_structure')
    ]
    
    for name, col in indicators_to_show:
        if col in df_complete.columns:
            value = latest_data[col]
            print(f"  {name}: {value:.4f}")
    
    return df_complete

def demo_trading_environment():
    """演示交易环境"""
    print("\n🎮 演示交易环境")
    print("=" * 40)
    
    # 使用示例数据创建环境
    df = generate_sample_data(days=30)  # 30天数据
    
    # 创建交易环境
    print("\n🏗️ 创建交易环境...")
    env = SolUsdtTradingEnv(df=df, mode='demo')
    
    print(f"观察空间维度: {env.observation_space.shape}")
    print(f"动作空间大小: {env.action_space.n}")
    print(f"特征数量: {len(env.observation_features)}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n🔄 环境重置完成，初始观察维度: {obs.shape}")
    
    # 模拟几步交易
    print("\n🎯 模拟交易...")
    actions = [0, 1, 0, 0, 3, 2, 0, 3]  # 预定义的动作序列
    action_names = ['观望', '开多', '观望', '观望', '平仓', '开空', '观望', '平仓']
    
    for i, (action, action_name) in enumerate(zip(actions, action_names)):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Step {i+1}: {action_name}")
        print(f"  奖励: {reward:.4f}")
        print(f"  组合价值: ${info['portfolio_value']:.2f}")
        print(f"  收益率: {info['total_return']:.2%}")
        print(f"  持仓类型: {info['position_type']}")
        
        if done:
            print("  ⚠️ 交易结束")
            break
    
    # 显示交易总结
    summary = env.get_trade_summary()
    print(f"\n📊 交易总结:")
    print(f"  最终收益率: {summary['total_return']:.2%}")
    print(f"  最大回撤: {summary['max_drawdown']:.2%}")
    print(f"  交易次数: {summary['total_trades']}")
    print(f"  胜率: {summary['win_rate']:.2%}")
    print(f"  夏普比率: {summary['sharpe_ratio']:.4f}")
    
    return env, summary

def demo_simple_strategy():
    """演示简单策略"""
    print("\n🧠 演示简单策略")
    print("=" * 40)
    
    # 生成数据
    df = generate_sample_data(days=60)
    
    # 创建环境
    env = SolUsdtTradingEnv(df=df, mode='demo')
    
    def simple_strategy(observation, info):
        """
        简单的基于RSI的策略
        - RSI < 30: 开多
        - RSI > 70: 开空
        - 有持仓时观望或平仓
        """
        # 获取当前数据
        current_step = info.get('current_step', 0)
        
        if current_step + env.lookback_window >= len(env.df):
            return 0  # 观望
        
        current_data = env.df.iloc[current_step + env.lookback_window]
        
        # 获取RSI值
        rsi = current_data.get('rsi', 50)
        position_type = info.get('position_type', 0)
        
        # 策略逻辑
        if position_type == 0:  # 无持仓
            if rsi < 30:
                return 1  # 开多
            elif rsi > 70:
                return 2  # 开空
            else:
                return 0  # 观望
        else:  # 有持仓
            # 简单的平仓条件
            if position_type == 1 and rsi > 60:  # 多头平仓
                return 3
            elif position_type == -1 and rsi < 40:  # 空头平仓
                return 3
            else:
                return 0  # 持仓观望
    
    # 运行策略
    print("🚀 运行RSI策略...")
    obs, info = env.reset()
    total_steps = 0
    max_steps = min(200, env.total_steps)  # 限制步数
    
    while total_steps < max_steps:
        info = env._get_info()
        action = simple_strategy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_steps += 1
        
        if total_steps % 50 == 0:
            print(f"  进度: {total_steps}/{max_steps}, "
                  f"组合价值: ${info['portfolio_value']:.2f}, "
                  f"收益率: {info['total_return']:.2%}")
        
        if done:
            break
    
    # 显示结果
    summary = env.get_trade_summary()
    print(f"\n📈 RSI策略结果:")
    print(f"  运行步数: {total_steps}")
    print(f"  最终收益率: {summary['total_return']:.2%}")
    print(f"  最大回撤: {summary['max_drawdown']:.2%}")
    print(f"  交易次数: {summary['total_trades']}")
    print(f"  胜率: {summary['win_rate']:.2%}")
    
    return summary

def demo_performance_analysis():
    """演示性能分析"""
    print("\n📊 演示性能分析")
    print("=" * 40)
    
    # 运行多个简单策略进行对比
    strategies_results = []
    
    # 策略1: 买入并持有
    print("\n🔄 测试买入并持有策略...")
    df = generate_sample_data(days=30)
    env = SolUsdtTradingEnv(df=df, mode='demo')
    obs, info = env.reset()
    
    # 第一步买入，然后持有
    env.step(1)  # 开多
    for _ in range(min(100, env.total_steps - 1)):
        obs, reward, terminated, truncated, info = env.step(0)  # 持仓观望
        done = terminated or truncated
        if done:
            break
    
    buy_hold_summary = env.get_trade_summary()
    strategies_results.append(('买入持有', buy_hold_summary))
    
    # 策略2: 随机交易
    print("🎲 测试随机交易策略...")
    env = SolUsdtTradingEnv(df=df, mode='demo')
    obs, info = env.reset()
    
    np.random.seed(42)
    for _ in range(min(100, env.total_steps)):
        action = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])  # 偏向观望
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break
    
    random_summary = env.get_trade_summary()
    strategies_results.append(('随机交易', random_summary))
    
    # 策略3: 简单趋势跟随
    print("📈 测试趋势跟随策略...")
    env = SolUsdtTradingEnv(df=df, mode='demo')
    obs, info = env.reset()
    
    for step in range(min(100, env.total_steps)):
        if step + env.lookback_window >= len(env.df):
            break
            
        current_data = env.df.iloc[step + env.lookback_window]
        prev_data = env.df.iloc[step + env.lookback_window - 1] if step > 0 else current_data
        
        price_change = (current_data['close'] - prev_data['close']) / prev_data['close']
        info = env._get_info()
        
        if info['position_type'] == 0:  # 无持仓
            if price_change > 0.005:  # 价格上涨超过0.5%
                action = 1  # 开多
            elif price_change < -0.005:  # 价格下跌超过0.5%
                action = 2  # 开空
            else:
                action = 0  # 观望
        else:
            action = 0  # 持仓观望
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            break
    
    trend_summary = env.get_trade_summary()
    strategies_results.append(('趋势跟随', trend_summary))
    
    # 对比结果
    print(f"\n📊 策略对比结果:")
    print("-" * 60)
    print(f"{'策略名称':<12} {'收益率':<10} {'最大回撤':<10} {'交易次数':<8} {'胜率':<8}")
    print("-" * 60)
    
    for name, summary in strategies_results:
        print(f"{name:<12} {summary['total_return']:>8.2%} {summary['max_drawdown']:>9.2%} "
              f"{summary['total_trades']:>7} {summary['win_rate']:>7.2%}")
    
    print("-" * 60)

def main():
    """主演示函数"""
    print("🚀 SOL/USDT交易策略系统 - 快速演示")
    print("=" * 50)
    
    try:
        # 1. 数据处理演示
        demo_data_processing()
        
        # 2. 交易环境演示
        demo_trading_environment()
        
        # 3. 简单策略演示
        demo_simple_strategy()
        
        # 4. 性能分析演示
        demo_performance_analysis()
        
        print("\n✅ 演示完成!")
        print("\n💡 下一步操作建议:")
        print("1. 运行 'python main.py --mode collect' 收集真实数据")
        print("2. 运行 'python main.py --mode train' 训练PPO模型")
        print("3. 运行 'python main.py --mode backtest' 进行回测")
        print("4. 查看生成的日志文件和结果图表")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 