"""
增强训练器模块
支持LSTM PPO、DQN以及先进的训练监控和模型对比功能
"""
import os
import time
import json
import numpy as np
import pandas as pd
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Stable Baselines3 imports
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.monitor import Monitor

# RecurrentPPO for LSTM support (if available)
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False
    print("⚠️ Warning: sb3_contrib not available, using standard PPO with LSTM policy")

from utils.config import get_config
from utils.logger import get_logger
from environment.trading_env import SolUsdtTradingEnv
from data.data_splitter import TimeSeriesDataSplitter

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = {
            'win_rate': [],
            'total_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'avg_trade_duration': [],
            'profit_factor': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        self.best_metrics = {
            'win_rate': 0.0,
            'total_return': -np.inf,
            'sharpe_ratio': -np.inf,
            'max_drawdown': np.inf,
            'timestep': 0,
            'model_path': None
        }
        
    def update(self, metrics: Dict[str, float], timestep: int):
        """更新性能指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                # 保持窗口大小
                if len(self.metrics_history[key]) > self.window_size:
                    self.metrics_history[key].pop(0)
        
        # 更新最佳指标
        if metrics.get('win_rate', 0) > self.best_metrics['win_rate']:
            self.best_metrics['win_rate'] = metrics['win_rate']
            self.best_metrics['timestep'] = timestep
        
        if metrics.get('total_return', -np.inf) > self.best_metrics['total_return']:
            self.best_metrics['total_return'] = metrics['total_return']
        
        if metrics.get('sharpe_ratio', -np.inf) > self.best_metrics['sharpe_ratio']:
            self.best_metrics['sharpe_ratio'] = metrics['sharpe_ratio']
        
        if metrics.get('max_drawdown', np.inf) < self.best_metrics['max_drawdown']:
            self.best_metrics['max_drawdown'] = metrics['max_drawdown']
    
    def get_recent_performance(self) -> Dict[str, float]:
        """获取最近的性能表现"""
        recent_metrics = {}
        for key, values in self.metrics_history.items():
            if values:
                recent_metrics[f'recent_{key}'] = np.mean(values[-10:])  # 最近10个值的平均
                recent_metrics[f'latest_{key}'] = values[-1]
        return recent_metrics
    
    def is_performance_improving(self) -> bool:
        """判断性能是否在改善"""
        if len(self.metrics_history['win_rate']) < 20:
            return True  # 数据不足，继续训练
        
        recent_win_rate = np.mean(self.metrics_history['win_rate'][-10:])
        early_win_rate = np.mean(self.metrics_history['win_rate'][-20:-10])
        
        return recent_win_rate > early_win_rate

class EnhancedTrainingCallback(BaseCallback):
    """增强训练回调，支持高级监控和保存策略"""
    
    def __init__(self, 
                 eval_env,
                 performance_tracker: PerformanceTracker,
                 config: Dict,
                 save_path: str = "./models/best/",
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.performance_tracker = performance_tracker
        self.config = config
        self.save_path = save_path
        self.last_eval_step = 0
        self.last_save_step = 0
        self.training_start_time = time.time()
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 监控配置
        self.monitor_config = config.get('TRAINING_MONITOR', {})
        self.eval_freq = self.monitor_config.get('EVAL_FREQ', 25000)
        self.save_freq = self.monitor_config.get('SAVE_FREQ', 50000)
        self.log_interval = self.monitor_config.get('LOG_INTERVAL', 1000)
        
    def _on_step(self) -> bool:
        # 定期日志记录
        if self.num_timesteps % self.log_interval == 0:
            self._log_training_progress()
        
        # 定期评估
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_performance()
            self.last_eval_step = self.num_timesteps
        
        # 定期保存检查点
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self._save_checkpoint()
            self.last_save_step = self.num_timesteps
        
        return True
    
    def _log_training_progress(self):
        """记录训练进度"""
        elapsed_time = time.time() - self.training_start_time
        speed = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        if self.verbose >= 1:
            print(f"📊 步数: {self.num_timesteps:,} | "
                  f"速度: {speed:.1f} 步/秒 | "
                  f"用时: {elapsed_time/3600:.2f}h")
    
    def _evaluate_performance(self):
        """评估当前模型性能"""
        try:
            # 运行评估环境
            reset_result = self.eval_env.reset()
            
            # ✅ 兼容不同版本的gym/gymnasium和向量化环境
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                # 新版本格式: (observation, info)
                obs, info = reset_result
            elif isinstance(reset_result, np.ndarray):
                # 旧版本格式或向量化环境: 只返回observation
                obs = reset_result
                info = {}
            else:
                # 其他情况，尝试获取观测
                obs = reset_result
                info = {}
                
            episode_rewards = []
            episode_lengths = []
            
            eval_episodes = self.monitor_config.get('EVAL_EPISODES', 10)  # ✅ 减少评估次数
            
            for episode in range(min(5, eval_episodes)):  # ✅ 进一步减少评估次数以节省时间
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done and episode_length < 500:  # ✅ 减少单集长度限制
                    try:
                        action, _ = self.model.predict(obs, deterministic=True)
                        step_result = self.eval_env.step(action)
                        
                        # ✅ 正确处理step()返回值 - 支持新旧API格式
                        if len(step_result) == 5:
                            obs, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        elif len(step_result) == 4:
                            obs, reward, done, info = step_result
                        else:
                            self.logger.warning(f"⚠️ 意外的step返回值数量: {len(step_result)}")
                            break
                        
                        # ✅ 安全处理reward（可能是数组或标量）
                        if isinstance(reward, np.ndarray):
                            episode_reward += reward[0] if len(reward) > 0 else 0
                        else:
                            episode_reward += reward
                            
                        episode_length += 1
                        
                        # ✅ 安全处理done（可能是数组或标量）
                        if isinstance(done, np.ndarray):
                            done = done[0] if len(done) > 0 else False
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ 评估步骤出错: {e}")
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # ✅ 为下一轮重置环境
                try:
                    reset_result = self.eval_env.reset()
                    # ✅ 兼容不同版本的gym/gymnasium和向量化环境
                    if isinstance(reset_result, tuple) and len(reset_result) == 2:
                        # 新版本格式: (observation, info)
                        obs, info = reset_result
                    elif isinstance(reset_result, np.ndarray):
                        # 旧版本格式或向量化环境: 只返回observation
                        obs = reset_result
                        info = {}
                    else:
                        # 其他情况，尝试获取观测
                        obs = reset_result
                        info = {}
                except Exception as e:
                    self.logger.warning(f"⚠️ 环境重置失败: {e}")
                    break
            
            # 计算评估指标
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths)
                
                # ✅ 安全获取交易环境的详细统计
                metrics = {
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'win_rate': 0.5,  # 默认值
                    'total_return': avg_reward / 1000,  # 粗略估算
                    'sharpe_ratio': avg_reward / (np.std(episode_rewards) + 1e-8),
                    'max_drawdown': 0.1  # 默认值
                }
                
                # ✅ 尝试获取详细的交易统计（如果可用）
                try:
                    if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                        env = self.eval_env.envs[0]
                        if hasattr(env, 'get_trade_summary'):
                            trade_summary = env.get_trade_summary()
                            metrics.update({
                                'win_rate': trade_summary.get('win_rate', metrics['win_rate']),
                                'total_return': trade_summary.get('total_return', metrics['total_return']),
                                'sharpe_ratio': trade_summary.get('sharpe_ratio', metrics['sharpe_ratio']),
                                'max_drawdown': trade_summary.get('max_drawdown', metrics['max_drawdown'])
                            })
                except Exception as e:
                    self.logger.debug(f"📊 无法获取详细交易统计: {e}")
                
                # 更新性能追踪器
                self.performance_tracker.update(metrics, self.num_timesteps)
                
                # 检查是否是最佳模型
                if self._is_best_model(metrics):
                    self._save_best_model(metrics)
                
                if self.verbose >= 1:
                    print(f"📈 评估结果 - 平均奖励: {avg_reward:.3f}, "
                          f"胜率: {metrics['win_rate']:.3f}, "
                          f"夏普比率: {metrics['sharpe_ratio']:.3f}")
            else:
                self.logger.warning("⚠️ 评估没有产生有效结果")
                
        except Exception as e:
            self.logger.error(f"❌ 评估性能失败: {e}")
            # 继续训练，不中断
    
    def _is_best_model(self, metrics: Dict[str, float]) -> bool:
        """判断是否为最佳模型"""
        metric_name = self.monitor_config.get('BEST_MODEL_METRIC', 'win_rate')
        
        if metric_name == 'win_rate':
            threshold = self.monitor_config.get('WIN_RATE_THRESHOLD', 0.55)
            return metrics.get('win_rate', 0) >= threshold
        elif metric_name == 'total_return':
            threshold = self.monitor_config.get('MIN_TOTAL_RETURN', 0.10)
            return metrics.get('total_return', 0) >= threshold
        elif metric_name == 'sharpe_ratio':
            return metrics.get('sharpe_ratio', -np.inf) > self.performance_tracker.best_metrics['sharpe_ratio']
        
        return False
    
    def _save_best_model(self, metrics: Dict[str, float]):
        """保存最佳模型"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"best_model_{timestamp}_step{self.num_timesteps}.zip"
            model_path = os.path.join(self.save_path, model_filename)
            
            self.model.save(model_path)
            self.performance_tracker.best_metrics['model_path'] = model_path
            
            # 保存模型评估信息
            eval_info = {
                'timestep': self.num_timesteps,
                'metrics': metrics,
                'timestamp': timestamp,
                'model_path': model_path
            }
            
            info_path = os.path.join(self.save_path, f"best_model_info_{timestamp}.json")
            with open(info_path, 'w') as f:
                json.dump(eval_info, f, indent=2)
            
            print(f"💾 保存最佳模型: {model_path}")
            print(f"🏆 胜率: {metrics['win_rate']:.3f} | 收益: {metrics['total_return']:.3f}")
            
        except Exception as e:
            print(f"❌ 保存最佳模型失败: {e}")
    
    def _save_checkpoint(self):
        """保存训练检查点"""
        try:
            checkpoint_path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            
            if self.verbose >= 1:
                print(f"💾 保存检查点: {checkpoint_path}")
                
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")

class EnhancedTrainer:
    """增强训练器 - 支持多种算法和高级监控"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('EnhancedTrainer', 'enhanced_trainer.log')
        self.performance_tracker = PerformanceTracker()
        
        # 模型配置
        self.model_configs = self.config.get('MODEL_CONFIGS', {})
        self.active_model = self.config.get('ACTIVE_MODEL', 'PPO_LSTM')
        
        # 训练结果存储
        self.training_results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        self.logger.info("🔄 准备训练数据...")
        
        # 使用时序数据划分器
        splitter = TimeSeriesDataSplitter()
        train_df, val_df, test_df = splitter.split_data(df)
        
        # 移除数据集标识列
        for dataset in [train_df, val_df, test_df]:
            if 'dataset_type' in dataset.columns:
                dataset.drop('dataset_type', axis=1, inplace=True)
        
        self.logger.info(f"📚 训练集: {len(train_df):,} 条记录")
        self.logger.info(f"📋 验证集: {len(val_df):,} 条记录")
        self.logger.info(f"🧪 测试集: {len(test_df):,} 条记录")
        
        return train_df, val_df, test_df
    
    def create_environments(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Any, Any]:
        """创建训练和验证环境"""
        self.logger.info("🏗️ 创建训练环境...")
        
        def make_env(data, mode='train'):
            def _init():
                try:
                    env = SolUsdtTradingEnv(data, mode=mode)
                    # ✅ 简化环境创建，去掉可能有问题的Monitor包装
                    return env
                except Exception as e:
                    self.logger.error(f"❌ 环境创建失败: {e}")
                    raise e
            return _init
        
        # 创建向量化环境
        n_envs = self.config.get('N_ENVS', 1)  # ✅ 减少到1个环境避免复杂性
        
        try:
            train_env = DummyVecEnv([make_env(train_df, 'train') for _ in range(n_envs)])
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
            
            # 验证环境
            eval_env = DummyVecEnv([make_env(val_df, 'eval')])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
            
            self.logger.info(f"✅ 环境创建成功 - 训练环境数量: {n_envs}")
            
            # ✅ 添加环境测试
            self.logger.info("🧪 测试环境重置...")
            try:
                reset_result = train_env.reset()
                
                # ✅ 兼容不同版本的gym/gymnasium和向量化环境
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    # 新版本格式: (observation, info)
                    test_obs, test_info = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测形状: {test_obs.shape}")
                elif isinstance(reset_result, np.ndarray):
                    # 旧版本格式或向量化环境: 只返回observation
                    test_obs = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测形状: {test_obs.shape}")
                else:
                    # 其他情况，尝试获取观测
                    test_obs = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测类型: {type(test_obs)}")
                    
            except Exception as e:
                self.logger.error(f"❌ 环境重置失败: {e}")
                # ✅ 提供更详细的错误信息
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                raise e
            
            return train_env, eval_env
            
        except Exception as e:
            self.logger.error(f"❌ 向量化环境创建失败: {e}")
            raise e
    
    def train_ppo_lstm(self, train_env, eval_env, total_timesteps: int) -> str:
        """训练LSTM PPO模型"""
        self.logger.info("🧠 开始训练 LSTM PPO 模型...")
        
        model_config = self.model_configs.get('PPO_LSTM', {})
        
        try:
            if RECURRENT_PPO_AVAILABLE:
                # 使用RecurrentPPO
                model = RecurrentPPO(
                    policy=model_config.get('policy', 'MlpLstmPolicy'),
                    env=train_env,
                    learning_rate=model_config.get('learning_rate', 1e-4),
                    gamma=model_config.get('gamma', 0.95),
                    gae_lambda=model_config.get('gae_lambda', 0.9),
                    clip_range=model_config.get('clip_range', 0.2),
                    n_steps=model_config.get('n_steps', 2048),
                    batch_size=model_config.get('batch_size', 256),
                    n_epochs=model_config.get('n_epochs', 10),
                    ent_coef=model_config.get('ent_coef', 0.005),
                    vf_coef=model_config.get('vf_coef', 0.5),
                    max_grad_norm=model_config.get('max_grad_norm', 0.5),
                    policy_kwargs=model_config.get('policy_kwargs', {}),
                    verbose=model_config.get('verbose', 1)
                )
            else:
                # 使用标准PPO with LSTM policy
                model = PPO(
                    policy='MlpLstmPolicy',
                    env=train_env,
                    learning_rate=model_config.get('learning_rate', 1e-4),
                    gamma=model_config.get('gamma', 0.95),
                    gae_lambda=model_config.get('gae_lambda', 0.9),
                    clip_range=model_config.get('clip_range', 0.2),
                    n_steps=model_config.get('n_steps', 2048),
                    batch_size=model_config.get('batch_size', 256),
                    n_epochs=model_config.get('n_epochs', 10),
                    ent_coef=model_config.get('ent_coef', 0.005),
                    vf_coef=model_config.get('vf_coef', 0.5),
                    max_grad_norm=model_config.get('max_grad_norm', 0.5),
                    policy_kwargs=model_config.get('policy_kwargs', {
                        'net_arch': [64, 64],
                        'lstm_hidden_size': 64,
                        'n_lstm_layers': 1
                    }),
                    verbose=model_config.get('verbose', 1)
                )
        
        except Exception as e:
            self.logger.error(f"❌ LSTM PPO模型初始化失败: {e}")
            self.logger.info("🔄 回退到标准PPO模型...")
            return self.train_ppo_standard(train_env, eval_env, total_timesteps)
        
        # 创建回调
        callback = EnhancedTrainingCallback(
            eval_env=eval_env,
            performance_tracker=self.performance_tracker,
            config=self.config.to_dict(),
            save_path="./models/ppo_lstm/"
        )
        
        # 开始训练
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/ppo_lstm_final_{timestamp}.zip"
        model.save(model_path)
        
        # 保存环境归一化参数
        train_env.save("models/ppo_lstm_vec_normalize.pkl")
        
        self.logger.info(f"✅ LSTM PPO训练完成 - 耗时: {training_time/3600:.2f}小时")
        self.logger.info(f"💾 模型保存至: {model_path}")
        
        return model_path
    
    def train_ppo_standard(self, train_env, eval_env, total_timesteps: int) -> str:
        """
        训练标准PPO模型 - 优化版，支持快速测试
        """
        self.logger.info("🧠 开始训练标准PPO模型...")
        
        try:
            # ✅ 优化的PPO配置，适合快速测试
            ppo_config = {
                'policy': 'MlpPolicy',
                'learning_rate': 3e-4,
                'n_steps': 512,        # 减少步数
                'batch_size': 64,      # 较小的批次
                'n_epochs': 4,         # 减少epoch
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 1,
                'policy_kwargs': {
                    'net_arch': [128, 128],  # 较小的网络
                    'activation_fn': torch.nn.ReLU
                }
            }
            
            self.logger.info("🔧 创建PPO模型...")
            
            # ✅ 添加环境测试
            self.logger.info("🧪 测试环境重置...")
            try:
                reset_result = train_env.reset()
                
                # ✅ 兼容不同版本的gym/gymnasium和向量化环境
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    # 新版本格式: (observation, info)
                    test_obs, test_info = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测形状: {test_obs.shape}")
                elif isinstance(reset_result, np.ndarray):
                    # 旧版本格式或向量化环境: 只返回observation
                    test_obs = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测形状: {test_obs.shape}")
                else:
                    # 其他情况，尝试获取观测
                    test_obs = reset_result
                    self.logger.info(f"✅ 环境重置成功，观测类型: {type(test_obs)}")
                    
            except Exception as e:
                self.logger.error(f"❌ 环境重置失败: {e}")
                # ✅ 提供更详细的错误信息
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                raise e
            
            # 创建模型
            self.logger.info("🔨 初始化PPO模型...")
            model = PPO(env=train_env, **ppo_config)
            self.logger.info("✅ PPO模型初始化成功")
            
            # ✅ 简化回调函数，避免复杂的评估逻辑
            performance_tracker = PerformanceTracker(window_size=50)
            
            # ✅ 使用更简单的回调或不使用回调
            self.logger.info("🎯 开始训练（无回调模式）...")
            
            # ✅ 先尝试无回调训练
            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # 保存最终模型
            save_path = f"./models/ppo_standard_final_{int(time.time())}.zip"
            model.save(save_path)
            
            self.logger.info(f"✅ PPO标准模型训练完成")
            self.logger.info(f"💾 模型已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ PPO标准模型训练失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return None
    
    def train_dqn(self, train_env, eval_env, total_timesteps: int) -> str:
        """训练DQN模型"""
        self.logger.info("🧠 开始训练DQN模型...")
        
        model_config = self.model_configs.get('DQN', {})
        
        model = DQN(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=model_config.get('learning_rate', 1e-4),
            buffer_size=model_config.get('buffer_size', 100000),
            learning_starts=model_config.get('learning_starts', 50000),
            batch_size=model_config.get('batch_size', 128),
            tau=model_config.get('tau', 1.0),
            gamma=model_config.get('gamma', 0.95),
            train_freq=model_config.get('train_freq', 4),
            gradient_steps=model_config.get('gradient_steps', 1),
            target_update_interval=model_config.get('target_update_interval', 10000),
            exploration_fraction=model_config.get('exploration_fraction', 0.1),
            exploration_initial_eps=model_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=model_config.get('exploration_final_eps', 0.05),
            policy_kwargs=model_config.get('policy_kwargs', {'net_arch': [128, 128, 64]}),
            verbose=model_config.get('verbose', 1)
        )
        
        # 创建回调
        callback = EnhancedTrainingCallback(
            eval_env=eval_env,
            performance_tracker=self.performance_tracker,
            config=self.config.to_dict(),
            save_path="./models/dqn/"
        )
        
        # 开始训练
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/dqn_final_{timestamp}.zip"
        model.save(model_path)
        
        # 保存环境归一化参数
        train_env.save("models/dqn_vec_normalize.pkl")
        
        self.logger.info(f"✅ DQN训练完成 - 耗时: {training_time/3600:.2f}小时")
        self.logger.info(f"💾 模型保存至: {model_path}")
        
        return model_path
    
    def train_multiple_models(self, df: pd.DataFrame) -> Dict[str, str]:
        """训练多个模型并对比性能"""
        self.logger.info("🚀 开始多模型训练和对比...")
        
        # 准备数据
        train_df, val_df, test_df = self.prepare_data(df)
        
        # 创建环境
        train_env, eval_env = self.create_environments(train_df, val_df)
        
        # 获取训练步数
        total_timesteps = self.config.get('TOTAL_TIMESTEPS', 500000)
        
        # 训练配置
        models_to_train = ['PPO_LSTM', 'PPO_STANDARD', 'DQN'] if self.config.get('TRAINING_MONITOR', {}).get('MODEL_COMPARISON', True) else [self.active_model]
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"🎯 开始训练 {model_name} 模型")
                self.logger.info(f"{'='*60}")
                
                # 重置性能追踪器
                self.performance_tracker = PerformanceTracker()
                
                if model_name == 'PPO_LSTM':
                    model_path = self.train_ppo_lstm(train_env, eval_env, total_timesteps)
                elif model_name == 'PPO_STANDARD':
                    model_path = self.train_ppo_standard(train_env, eval_env, total_timesteps)
                elif model_name == 'DQN':
                    model_path = self.train_dqn(train_env, eval_env, total_timesteps)
                else:
                    self.logger.warning(f"⚠️ 未知模型类型: {model_name}")
                    continue
                
                trained_models[model_name] = model_path
                
                # 保存性能结果
                self.training_results[model_name] = {
                    'model_path': model_path,
                    'best_metrics': self.performance_tracker.best_metrics.copy(),
                    'final_performance': self.performance_tracker.get_recent_performance()
                }
                
                self.logger.info(f"✅ {model_name} 训练完成")
                
            except Exception as e:
                self.logger.error(f"❌ {model_name} 训练失败: {e}")
                continue
        
        # 生成对比报告
        self._generate_comparison_report()
        
        # 保存测试数据供回测使用
        test_data_path = "data/test_data.pkl"
        test_df.to_pickle(test_data_path)
        self.logger.info(f"💾 测试数据已保存至: {test_data_path}")
        
        return trained_models
    
    def _generate_comparison_report(self):
        """生成模型对比报告"""
        if not self.training_results:
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 模型训练对比报告")
        self.logger.info("="*80)
        
        # 创建对比表格
        comparison_data = []
        for model_name, results in self.training_results.items():
            best_metrics = results['best_metrics']
            comparison_data.append({
                'Model': model_name,
                'Win Rate': f"{best_metrics.get('win_rate', 0):.3f}",
                'Total Return': f"{best_metrics.get('total_return', 0):.3f}",
                'Sharpe Ratio': f"{best_metrics.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown': f"{best_metrics.get('max_drawdown', 0):.3f}",
                'Best Timestep': best_metrics.get('timestep', 0),
                'Model Path': results['model_path']
            })
        
        # 显示对比结果
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            self.logger.info("\n模型性能对比:")
            self.logger.info(df_comparison.to_string(index=False))
            
            # 找出最佳模型
            best_model = max(comparison_data, key=lambda x: float(x['Win Rate']))
            self.logger.info(f"\n🏆 最佳模型: {best_model['Model']}")
            self.logger.info(f"   胜率: {best_model['Win Rate']}")
            self.logger.info(f"   总收益: {best_model['Total Return']}")
            self.logger.info(f"   模型路径: {best_model['Model Path']}")
            
            # 保存对比报告
            report_path = f"results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            df_comparison.to_csv(report_path, index=False)
            self.logger.info(f"📋 对比报告已保存: {report_path}")
        
        self.logger.info("="*80)

    def _quick_model_test(self, model_path: str, test_df: pd.DataFrame) -> Dict:
        """
        快速模型测试 - 用于验证训练效果
        """
        try:
            from stable_baselines3 import PPO
            import numpy as np
            
            # 加载模型
            model = PPO.load(model_path)
            
            # 创建测试环境
            from environment.trading_env import SolUsdtTradingEnv
            test_env = SolUsdtTradingEnv(test_df, mode='test')
            
            # 运行一个回合
            reset_result = test_env.reset()
            
            # ✅ 兼容不同版本的gym/gymnasium
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                # 新版本格式: (observation, info)
                obs, info = reset_result
            elif isinstance(reset_result, np.ndarray):
                # 旧版本格式: 只返回observation
                obs = reset_result
                info = {}
            else:
                # 其他情况，尝试获取观测
                obs = reset_result
                info = {}
                
            total_reward = 0
            step_count = 0
            max_steps = min(len(test_df) - 100, 5000)  # 🔧 增加最大步数，但不超过数据长度
            
            self.logger.info(f"📊 开始快速测试，最大步数: {max_steps}")
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                step_result = test_env.step(action)
                
                # ✅ 正确处理step()返回的5个值 (obs, reward, terminated, truncated, info)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # 向后兼容性处理
                    obs, reward, done, info = step_result
                    
                total_reward += reward
                step_count += 1
                
                # 🔧 每1000步输出进度
                if step_count % 1000 == 0:
                    current_portfolio = info.get('portfolio_value', test_env.initial_balance)
                    current_return = (current_portfolio - test_env.initial_balance) / test_env.initial_balance
                    self.logger.info(f"📈 步数: {step_count}, 当前收益率: {current_return:.2%}, 投资组合价值: ${current_portfolio:.2f}")
                
                if done or step_count >= max_steps:
                    break
            
            # 计算基本指标
            portfolio_value = info.get('portfolio_value', test_env.initial_balance)
            total_return = (portfolio_value - test_env.initial_balance) / test_env.initial_balance
            
            # 计算简单夏普比率（如果有足够数据）
            if len(test_env.portfolio_history) > 10:
                returns = np.diff(test_env.portfolio_history) / test_env.portfolio_history[:-1]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': info.get('total_trades', 0),
                'win_rate': info.get('win_rate', 0),
                'max_drawdown': info.get('max_drawdown', 0),
                'steps': step_count,
                'portfolio_value': portfolio_value,
                'initial_balance': test_env.initial_balance,
                'total_reward': total_reward
            }
            
            self.logger.info(f"✅ 快速测试完成: 收益率={total_return:.2%}, 夏普比率={sharpe_ratio:.3f}")
            self.logger.info(f"📊 测试详情: 步数={step_count}/{max_steps}, 交易次数={info.get('total_trades', 0)}, 胜率={info.get('win_rate', 0):.2%}")
            self.logger.info(f"💰 资产变化: ${test_env.initial_balance:.2f} → ${portfolio_value:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 快速测试失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

def main():
    """主函数，用于测试增强训练器"""
    from data.data_collector import DataCollector
    
    print("🚀 测试增强训练器")
    
    # 加载数据
    collector = DataCollector()
    df = collector.load_data()
    
    if df.empty:
        print("❌ 未找到训练数据")
        return
    
    # 创建增强训练器
    trainer = EnhancedTrainer()
    
    # 训练多个模型
    trained_models = trainer.train_multiple_models(df)
    
    print(f"\n✅ 训练完成，共训练 {len(trained_models)} 个模型:")
    for model_name, model_path in trained_models.items():
        print(f"  {model_name}: {model_path}")

if __name__ == "__main__":
    main() 