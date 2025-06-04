#!/usr/bin/env python3
"""
模型诊断脚本 - 分析模型行为和环境状态
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from environment.trading_env import SolUsdtTradingEnv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def diagnose_model_behavior(model_path: str, data_path: str = "data/test_data.pkl"):
    """诊断模型行为"""
    print("🔍 开始模型诊断...")
    
    # 加载数据和模型
    df = pd.read_pickle(data_path)
    model = PPO.load(model_path)
    env = SolUsdtTradingEnv(df, mode='test')
    
    print(f"📊 数据长度: {len(df)} 条记录")
    print(f"🎯 动作空间: {env.action_space}")
    print(f"📈 观测空间: {env.observation_space}")
    
    # 收集样本
    obs, _ = env.reset()
    action_history = []
    observation_samples = []
    action_probs_history = []
    
    print("\n🎯 收集模型预测样本...")
    for i in range(min(1000, len(df) - 100)):
        # 获取动作和动作概率
        action, _states = model.predict(obs, deterministic=False)
        
        # 🔧 确保action是整数
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # 获取动作概率分布
        if hasattr(model.policy, 'get_distribution'):
            try:
                with model.policy.eval():
                    obs_tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
                    distribution = model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]
                    action_probs_history.append(action_probs)
            except:
                action_probs_history.append([0.33, 0.33, 0.34])  # 默认均匀分布
        
        action_history.append(action)
        observation_samples.append(obs.copy())
        
        # 执行动作
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        if done:
            obs, _ = env.reset()
            
        if i % 200 == 0:
            print(f"📈 已处理 {i} 步，当前动作: {action}")
    
    # 分析结果
    print("\n" + "="*60)
    print("📊 模型行为分析结果")
    print("="*60)
    
    # 1. 动作分布分析
    action_counts = Counter(action_history)
    print(f"\n🎯 动作分布:")
    for action, count in sorted(action_counts.items()):
        percentage = count / len(action_history) * 100
        action_name = ['持仓', '开多', '开空'][action] if action < 3 else f'动作{action}'
        print(f"   {action_name} (动作{action}): {count} 次 ({percentage:.1f}%)")
    
    # 2. 观测数据分析
    obs_array = np.array(observation_samples)
    print(f"\n📈 观测数据统计:")
    print(f"   观测维度: {obs_array.shape}")
    print(f"   数据范围: [{obs_array.min():.3f}, {obs_array.max():.3f}]")
    print(f"   平均值: {obs_array.mean():.3f}")
    print(f"   标准差: {obs_array.std():.3f}")
    print(f"   是否有NaN: {np.isnan(obs_array).any()}")
    print(f"   是否有Inf: {np.isinf(obs_array).any()}")
    
    # 3. 动作概率分析
    if action_probs_history:
        probs_array = np.array(action_probs_history)
        avg_probs = probs_array.mean(axis=0)
        print(f"\n🎲 平均动作概率:")
        for i, prob in enumerate(avg_probs):
            action_name = ['持仓', '开多', '开空'][i] if i < 3 else f'动作{i}'
            print(f"   {action_name}: {prob:.3f}")
    
    # 4. 检查环境信号
    print(f"\n📊 环境信号分析:")
    smc_signals = df['enhanced_smc_signal'].value_counts() if 'enhanced_smc_signal' in df.columns else {}
    if smc_signals.empty:
        print("   ⚠️ 未找到SMC信号")
    else:
        print(f"   SMC信号分布: {dict(smc_signals)}")
        
    # 检查技术指标
    tech_indicators = ['rsi', 'bb_position', 'ema_trend']
    for indicator in tech_indicators:
        if indicator in df.columns:
            values = df[indicator].dropna()
            print(f"   {indicator}: 范围[{values.min():.3f}, {values.max():.3f}], 均值{values.mean():.3f}")
    
    # 5. 创建可视化
    print(f"\n🎨 生成诊断图表...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 动作分布饼图
    if action_counts:
        labels = [f'动作{k}' for k in sorted(action_counts.keys())]
        sizes = [action_counts[k] for k in sorted(action_counts.keys())]
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
        axes[0, 0].set_title('模型动作分布')
    
    # 观测数据分布
    if len(observation_samples) > 0:
        # 取前10个特征进行可视化
        obs_df = pd.DataFrame(observation_samples)
        obs_df.iloc[:, :min(10, obs_df.shape[1])].hist(ax=axes[0, 1], bins=30)
        axes[0, 1].set_title('观测特征分布 (前10个)')
    
    # 动作时间序列
    axes[1, 0].plot(action_history[:min(500, len(action_history))])
    axes[1, 0].set_title('动作序列 (前500步)')
    axes[1, 0].set_ylabel('动作')
    axes[1, 0].set_xlabel('时间步')
    
    # 动作概率热图
    if action_probs_history:
        probs_sample = np.array(action_probs_history)[:min(100, len(action_probs_history))]
        sns.heatmap(probs_sample.T, ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title('动作概率热图 (前100步)')
        axes[1, 1].set_ylabel('动作')
        axes[1, 1].set_xlabel('时间步')
    
    plt.tight_layout()
    plt.savefig('results/model_diagnosis.png', dpi=300, bbox_inches='tight')
    print(f"📊 诊断图表已保存: results/model_diagnosis.png")
    
    # 6. 问题诊断和建议
    print(f"\n🔧 问题诊断和建议:")
    
    if action_counts.get(0, 0) / len(action_history) > 0.95:
        print("   ❌ 问题: 模型几乎只预测持仓动作")
        print("   💡 建议: 检查奖励函数设计，可能需要增加探索奖励")
        
    if np.isnan(obs_array).any() or np.isinf(obs_array).any():
        print("   ❌ 问题: 观测数据包含异常值")
        print("   💡 建议: 检查特征工程和数据预处理")
        
    if obs_array.std() < 0.1:
        print("   ❌ 问题: 观测数据变化太小，缺乏信息")
        print("   💡 建议: 检查特征归一化和缩放")
        
    if len(set(action_history)) == 1:
        print("   ❌ 问题: 模型输出完全没有变化")
        print("   💡 建议: 模型可能需要更多训练或调整网络结构")
    
    # 7. 推荐的修复措施
    print(f"\n🛠️ 推荐的修复措施:")
    print("   1. 增加训练时间步数")
    print("   2. 调整奖励函数，增加交易激励")
    print("   3. 检查特征工程质量")
    print("   4. 尝试不同的网络架构")
    print("   5. 增加探索噪声")
    
    return {
        'action_distribution': action_counts,
        'observation_stats': {
            'shape': obs_array.shape,
            'mean': obs_array.mean(),
            'std': obs_array.std(),
            'min': obs_array.min(),
            'max': obs_array.max(),
            'has_nan': np.isnan(obs_array).any(),
            'has_inf': np.isinf(obs_array).any()
        },
        'action_probs': avg_probs if action_probs_history else None
    }

if __name__ == "__main__":
    model_path = "./models/ppo_standard_final_1748899650.zip"
    results = diagnose_model_behavior(model_path)
    print("\n✅ 诊断完成!") 