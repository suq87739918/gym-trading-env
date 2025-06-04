#!/usr/bin/env python3
"""
SOL/USDT量化交易系统 - 主程序
集成强化学习、SMC分析和增强风控的完整交易解决方案
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import time
import hashlib
import pandas as pd
import numpy as np
import pickle

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from data.data_collector import DataCollector
from data.technical_indicators import TechnicalIndicators
from data.smc_signals import SMCSignals
from environment.trading_env import SolUsdtTradingEnv, make_env
from analysis.trading_visualizer import TradingVisualizer
from utils.config import get_config
from utils.logger import get_logger
from data.data_splitter import TimeSeriesDataSplitter
from training.enhanced_trainer import EnhancedTrainer

# 强化学习相关导入
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    import tensorboard
    
    # ✅ 新增：导入sb3_contrib用于RecurrentPPO
    try:
        from sb3_contrib import RecurrentPPO
        RECURRENT_PPO_AVAILABLE = True
        print("✅ sb3_contrib可用，支持RecurrentPPO")
    except ImportError:
        RECURRENT_PPO_AVAILABLE = False
        print("⚠️ sb3_contrib不可用，将使用标准PPO代替")
    
    REINFORCEMENT_LEARNING_AVAILABLE = True
    print("✅ 强化学习库加载成功")
    
except ImportError as e:
    print(f"⚠️ 强化学习库不可用: {e}")
    print("💡 请安装: pip install stable-baselines3[extra] sb3_contrib")
    REINFORCEMENT_LEARNING_AVAILABLE = False
    RECURRENT_PPO_AVAILABLE = False

config = get_config()
logger = get_logger('MainSystem', 'main.log')

warnings.filterwarnings('ignore')

class DataCache:
    """✅ 数据缓存管理器 - 避免重复计算"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_data_hash(self, df) -> str:
        """计算数据hash值"""
        return hashlib.md5(str(df.shape).encode() + str(df.index[0]).encode() + str(df.index[-1]).encode()).hexdigest()
    
    def get_processed_data(self, df, force_refresh: bool = False):
        """获取处理后的数据（带缓存）"""
        data_hash = self._get_data_hash(df)
        cache_file = self.cache_dir / f"processed_data_{data_hash}.pkl"
        
        # 检查缓存
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"✅ 从缓存加载处理后的数据: {cache_file}")
                    return cached_data
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}")
        
        # 计算技术指标和SMC信号
        print("🔄 计算技术指标...")
        indicator_calculator = TechnicalIndicators()
        
        # 检查是否已有指标
        if 'rsi' not in df.columns:
            df = indicator_calculator.calculate_all_indicators(df)
        else:
            print("✅ 技术指标已存在，跳过计算")
        
        print("🎯 计算SMC信号...")
        smc_calculator = SMCSignals()
        
        # 检查是否已有SMC信号
        if 'smc_signal' not in df.columns:
            df = smc_calculator.calculate_all_smc_signals(df)
        else:
            print("✅ SMC信号已存在，跳过计算")
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
                print(f"💾 数据已缓存至: {cache_file}")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")
        
        return df

def setup_directories():
    """设置必要的目录结构 - 增强版"""
    config = get_config()
    directories = [
        config.get('DATA_DIR'),
        config.get('MODEL_DIR'),
        config.get('LOG_DIR'),
        config.get('RESULTS_DIR'),
        'cache',  # 新增缓存目录
        'logs/tensorboard',  # TensorBoard日志目录
        'logs/eval',  # 评估日志目录
        'models/best',  # 最佳模型目录
        'models/checkpoints'  # 检查点目录
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 目录已创建/确认: {directory}")
        except Exception as e:
            print(f"⚠️ 创建目录失败 {directory}: {e}")
    
    # 清理可能存在的TensorBoard日志冲突
    tensorboard_dir = "logs/tensorboard"
    if os.path.exists(tensorboard_dir):
        # 检查是否有文件冲突
        for item in os.listdir(tensorboard_dir):
            item_path = os.path.join(tensorboard_dir, item)
            if os.path.isfile(item_path) and item.startswith("PPO"):
                # 删除可能的冲突文件
                try:
                    os.remove(item_path)
                    print(f"🗑️ 清理TensorBoard冲突文件: {item}")
                except Exception as e:
                    print(f"⚠️ 清理文件失败 {item}: {e}")
    
    print("✅ 目录结构设置完成")

def collect_data():
    """数据收集模式"""
    print("🔄 开始数据收集...")
    
    collector = DataCollector()
    
    # 获取历史数据
    df = collector.get_historical_data()
    
    if not df.empty:
        # 保存原始数据
        filepath = collector.save_data(df)
        
        # 使用缓存管理器处理数据
        cache_manager = DataCache()
        df_processed = cache_manager.get_processed_data(df)
        
        # 保存完整数据
        complete_filepath = filepath.replace('.csv', '_complete.csv')
        df_processed.to_csv(complete_filepath)
        
        print(f"✅ 数据收集完成!")
        print(f"📈 原始数据: {len(df)} 条记录")
        print(f"🔢 技术指标数量: {len([col for col in df_processed.columns if col not in df.columns])}")
        print(f"💾 完整数据保存至: {complete_filepath}")
        
        # 显示数据摘要
        print(f"\n📊 数据摘要:")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"平均成交量: {df['volume'].mean():.0f}")
        
        return df_processed
    else:
        print("❌ 数据收集失败")
        return None

def train_model(df=None):
    """
    ✅ 优化版训练函数 - 支持多种模型训练
    """
    print("🚀 开始智能训练流程...")
    start_time = time.time()
    
    try:
        # 数据准备
        if df is None:
            cache = DataCache()
            df = cache.get_processed_data(None, force_refresh=False)
            if df is None or df.empty:
                print("❌ 无法获取训练数据")
                return None
        
        print(f"📊 数据概览: {len(df)} 条记录, 时间跨度: {df.index[0]} 到 {df.index[-1]}")
        
        # ✅ 修改：使用较少的训练步数以便快速测试
        training_config = {
            'total_timesteps': 50000,  # 减少到5万步，约10-15分钟训练
            'eval_freq': 5000,        # 每5千步评估一次
            'n_eval_episodes': 3,     # 减少评估回合数
            'save_freq': 10000,       # 每1万步保存一次
            'log_interval': 1000      # 每1千步记录一次
        }
        
        print(f"⚡ 快速测试模式: {training_config['total_timesteps']} 步训练")
        
        trainer = EnhancedTrainer()
        
        # 数据分割（使用更小的验证集）
        train_df, val_df, test_df = trainer.prepare_data(df)
        print(f"📊 数据分割: 训练 {len(train_df)}, 验证 {len(val_df)}, 测试 {len(test_df)}")
        
        # 创建环境
        train_env, eval_env = trainer.create_environments(train_df, val_df)
        print(f"🏢 环境创建完成: 观测空间 {train_env.observation_space}, 动作空间 {train_env.action_space}")
        
        # ✅ 使用单一模型进行快速测试
        model_type = 'PPO_STANDARD'  # 使用最稳定的PPO
        print(f"🎯 开始训练 {model_type} 模型（快速测试模式）")
        
        # 训练模型
        if model_type == 'PPO_STANDARD':
            model_path = trainer.train_ppo_standard(train_env, eval_env, training_config['total_timesteps'])
        else:
            print(f"❌ 暂不支持 {model_type} 模型")
            return None
        
        if model_path:
            print(f"✅ 模型训练成功: {model_path}")
            
            # 快速验证测试
            print("🧪 开始快速验证测试...")
            test_results = trainer._quick_model_test(model_path, test_df)
            
            if test_results:
                print(f"📊 测试结果: 收益率 {test_results.get('total_return', 0):.2%}, "
                      f"夏普比率 {test_results.get('sharpe_ratio', 0):.3f}")
            
            training_time = time.time() - start_time
            print(f"⏱️ 训练完成，耗时: {training_time:.1f}秒")
            
            return model_path
        else:
            print("❌ 模型训练失败")
            return None
            
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model_legacy(df=None):
    """
    ✅ 传统训练模式 - 作为后备选项
    """
    if df is None:
        collector = DataCollector()
        df = collector.load_data()
        
        if df.empty:
            print("❌ 未找到训练数据，开始获取历史数据...")
            df = collector.get_historical_data()
            if not df.empty:
                collector.save_data(df)
            else:
                print("❌ 数据获取失败")
                return None

    print(f"📈 使用数据: {len(df)} 条记录 (时间范围: {df.index.min()} 到 {df.index.max()})")
    
    # ✅ 新增：使用数据划分器进行训练测试分离
    splitter = TimeSeriesDataSplitter()
    train_df, val_df, test_df = splitter.split_data(df)
    
    print("🔄 数据划分完成:")
    print(f"  📚 训练集: {len(train_df):,} 条记录 ({train_df.index.min()} 到 {train_df.index.max()})")
    print(f"  📋 验证集: {len(val_df):,} 条记录 ({val_df.index.min()} 到 {val_df.index.max()})")
    print(f"  🧪 测试集: {len(test_df):,} 条记录 ({test_df.index.min()} 到 {test_df.index.max()})")
    
    # 移除数据集标识列（避免模型学习到这个信息）
    for dataset in [train_df, val_df, test_df]:
        if 'dataset_type' in dataset.columns:
            dataset.drop('dataset_type', axis=1, inplace=True)
    
    # 使用缓存管理器处理训练数据
    cache_manager = DataCache()
    train_df_processed = cache_manager.get_processed_data(train_df, force_refresh=True)
    
    # 创建训练环境 - 只使用训练数据
    print("🏗️ 创建训练环境（仅使用训练数据）...")
    def make_env_func(data, mode='train'):
        def _init():
            return SolUsdtTradingEnv(data, mode=mode)
        return _init
    
    # 创建向量化环境
    n_envs = config.get('N_ENVS', 4)
    train_env = DummyVecEnv([make_env_func(train_df_processed, 'train') for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # 验证环境 - 使用验证数据
    val_df_processed = cache_manager.get_processed_data(val_df, force_refresh=False)
    eval_env = DummyVecEnv([make_env_func(val_df_processed, 'eval')])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    
    # ✅ 保存测试数据供后续回测使用
    test_data_path = "data/test_data.pkl"
    test_df.to_pickle(test_data_path)
    print(f"💾 测试数据已保存至: {test_data_path}")
    
    # ✅ 使用优化的模型配置
    active_model = config.get('ACTIVE_MODEL', 'PPO_LSTM')
    model_configs = config.get('MODEL_CONFIGS', {})
    
    if active_model in model_configs:
        model_config = model_configs[active_model]
        print(f"🎯 使用模型配置: {active_model}")
    else:
        # 使用默认配置
        model_config = {
            'policy': 'MlpPolicy',
            'learning_rate': 2.5e-4,
            'gamma': 0.99,
            'clip_range': 0.2,
            'n_steps': 2048,
            'batch_size': 512,
            'n_epochs': 10,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'gae_lambda': 0.95,
            'verbose': 1
        }
        print("🎯 使用默认PPO配置")
    
    print("🧠 初始化PPO模型...")
    
    try:
        # ✅ 根据配置选择算法
        algorithm_type = model_config.get('algorithm', 'PPO')
        
        # 从配置中移除algorithm字段，避免传递给模型构造函数
        model_params = {k: v for k, v in model_config.items() if k != 'algorithm'}
        
        if algorithm_type == 'RecurrentPPO' and RECURRENT_PPO_AVAILABLE:
            # 使用RecurrentPPO for LSTM
            print("🧠 初始化RecurrentPPO模型...")
            
            # ✅ 确保使用正确的策略类型
            if model_params.get('policy') == 'MlpPolicy':
                model_params['policy'] = 'MlpLstmPolicy'
                print("📝 策略类型已调整为MlpLstmPolicy（RecurrentPPO专用）")
            
            # ✅ 处理policy_kwargs中的RecurrentPPO特有参数
            policy_kwargs = model_params.get('policy_kwargs', {})
            # 移除可能冲突的参数，RecurrentPPO会自动处理这些
            cleaned_policy_kwargs = {k: v for k, v in policy_kwargs.items() 
                                   if k not in ['enable_critic_lstm', 'lstm_hidden_size']}
            model_params['policy_kwargs'] = cleaned_policy_kwargs
            
            # ✅ 添加诊断信息
            print(f"🔍 RecurrentPPO 配置诊断:")
            print(f"   策略类型: {model_params.get('policy')}")
            print(f"   学习率: {model_params.get('learning_rate')}")
            print(f"   网络架构: {model_params.get('policy_kwargs', {})}")
            print(f"   环境类型: {type(train_env)}")
            print(f"   观测空间: {train_env.observation_space}")
            print(f"   动作空间: {train_env.action_space}")
            
            model = RecurrentPPO(
                env=train_env,
                **model_params
            )
            print("✅ RecurrentPPO模型初始化成功 (支持LSTM)")
        else:
            # 使用标准PPO（兼容性回退）
            if algorithm_type == 'RecurrentPPO':
                print("⚠️ RecurrentPPO不可用，回退到标准PPO")
                # 如果是RecurrentPPO配置，需要调整policy参数
                if model_params.get('policy') == 'MlpLstmPolicy':
                    model_params['policy'] = 'MlpPolicy'
                    print("📝 策略类型已调整为MlpPolicy")
            
            print("🧠 初始化标准PPO模型...")
            # 移除RecurrentPPO特有的参数
            ppo_params = {k: v for k, v in model_params.items() 
                         if k not in ['enable_critic_lstm', 'lstm_hidden_size', 'target_kl']}
            
            model = PPO(
                env=train_env,
                **ppo_params
            )
            print("✅ 标准PPO模型初始化成功")
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("🔍 诊断信息:")
        print(f"   训练环境类型: {type(train_env)}")
        print(f"   观测空间: {train_env.observation_space}")
        print(f"   动作空间: {train_env.action_space}")
        print(f"   配置的算法类型: {model_config.get('algorithm', 'PPO')}")
        print(f"   RecurrentPPO可用性: {RECURRENT_PPO_AVAILABLE}")
        return None
    
    # ✅ 保留有用的回调函数
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_sol_trading'
    )
    
    # ✅ 训练进度监控回调
    class TrainingProgressCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TrainingProgressCallback, self).__init__(verbose)
            self.start_time = time.time()
            self.last_checkpoint = 0
            
        def _on_step(self) -> bool:
            # 每10万步显示详细进度
            if self.num_timesteps - self.last_checkpoint >= 100000:
                elapsed_time = time.time() - self.start_time
                progress = self.num_timesteps / self.model.num_timesteps
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"\n" + "="*60)
                print(f"📊 训练进度检查点 - 步数: {self.num_timesteps:,}")
                print(f"📈 完成进度: {progress*100:.2f}%")
                print(f"⏱️  已用时间: {elapsed_time/3600:.2f}小时")
                print(f"⏳ 剩余时间: {remaining_time/3600:.2f}小时")
                print(f"🚀 平均速度: {self.num_timesteps/elapsed_time:.1f} 步/秒")
                print("="*60)
                
                self.last_checkpoint = self.num_timesteps
            
            return True
    
    progress_callback = TrainingProgressCallback()
    
    # 组合所有回调
    callbacks = [eval_callback, checkpoint_callback, progress_callback]
    
    # 开始训练
    total_timesteps = config.get('TOTAL_TIMESTEPS', 2000000)
    print(f"\n🚀 开始训练 - 总步数: {total_timesteps:,}")
    print(f"💾 模型保存路径: ./models/")
    print(f"🔧 环境数量: {n_envs}")
    print(f"⚙️  训练参数: 学习率={model_config.get('learning_rate')}, 批次={model_config.get('batch_size')}")
    
    # 添加环境诊断信息
    print(f"\n🔍 环境诊断信息:")
    print(f"   数据集大小: {len(df)} 行")
    print(f"   特征维度: {len(df.columns)} 列")
    print(f"   观测空间维度: {train_env.observation_space.shape}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 测试环境重置
    try:
        reset_result = train_env.reset()
        
        # ✅ 兼容不同版本的gym/gymnasium和向量化环境
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            # 新版本格式: (observation, info)
            test_obs, test_info = reset_result
        elif isinstance(reset_result, np.ndarray):
            # 旧版本格式或向量化环境: 只返回observation
            test_obs = reset_result
            test_info = {}
        else:
            # 其他情况，尝试获取观测
            test_obs = reset_result
            test_info = {}
            
        print(f"   测试观测向量形状: {test_obs.shape}")
        print(f"   观测向量样本: {test_obs[0][:5]}...")  # 显示前5个值
        print("✅ 环境测试通过")
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return None
    
    try:
        training_start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        # 保存最终模型
        model_save_path = f"models/ppo_sol_trading_final_{total_timesteps}.zip"
        model.save(model_save_path)
        
        # 保存环境归一化参数
        train_env.save("models/vec_normalize.pkl")
        
        print(f"\n✅ 训练完成！")
        print(f"⏱️  训练耗时: {training_duration/3600:.2f}小时")
        print(f"🚀 平均训练速度: {total_timesteps/training_duration:.1f} 步/秒")
        print(f"💾 模型已保存至: {model_save_path}")
        print(f"📊 启动TensorBoard: tensorboard --logdir=logs/tensorboard/")
        
        return model_save_path
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        return None

def backtest_model(df=None, model_path=None):
    """
    ✅ 优化的回测模式 - 使用独立测试数据防止泄露
    """
    if model_path is None:
        model_path = "models/ppo_sol_trading_final.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
        return None
    
    print(f"📊 开始模型回测，模型路径: {model_path}")
    
    # ✅ 优先使用独立的测试数据
    test_data_path = "data/test_data.pkl"
    
    if os.path.exists(test_data_path) and df is None:
        print("📋 使用独立测试数据进行回测...")
        try:
            df = pd.read_pickle(test_data_path)
            print(f"✅ 测试数据加载成功: {len(df)} 条记录")
            print(f"📅 测试数据时间范围: {df.index.min()} 到 {df.index.max()}")
            print("🔒 这些数据在训练过程中未被使用，确保回测结果的真实性")
        except Exception as e:
            print(f"❌ 测试数据加载失败: {e}")
            df = None
    
    # 如果没有测试数据，回退到常规数据加载
    if df is None:
        print("⚠️  未找到独立测试数据，使用完整数据集进行回测")
        print("⚠️  注意：这可能包含训练期间使用的数据")
        
        collector = DataCollector()
        df = collector.load_data()
        
        if df.empty:
            print("❌ 未找到回测数据，请先运行数据收集")
            return None
    
    print(f"📈 回测数据: {len(df)} 条记录")
    
    # 确保数据已处理
    try:
        # 检查是否有必要的特征列
        required_features = ['rsi', 'bb_position', 'ema_fast', 'smc_signal']
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            print(f"🔧 检测到缺失特征: {missing_features}")
            print("🔄 重新处理回测数据...")
            
            # 重新计算指标
            from data.technical_indicators import TechnicalIndicators
            from data.smc_signals import SMCSignals
            
            indicator_calculator = TechnicalIndicators()
            df = indicator_calculator.calculate_all_indicators(df)
            
            smc_calculator = SMCSignals()
            df = smc_calculator.calculate_all_smc_signals(df)
            
            # 填充缺失值
            df = df.fillna(method='ffill').fillna(0)
            print("✅ 回测数据处理完成")
        
    except Exception as e:
        print(f"❌ 回测数据处理失败: {e}")
        return None
    
    try:
        # 加载模型
        print("🧠 加载PPO模型...")
        model = PPO.load(model_path)
        
        # 加载环境归一化参数
        vec_normalize_path = "models/vec_normalize.pkl"
        if os.path.exists(vec_normalize_path):
            print("📐 加载环境归一化参数...")
        
        # 创建回测环境
        print("🏗️ 创建回测环境...")
        env = SolUsdtTradingEnv(df, mode='test')
        
        # 确保使用与训练时相同的归一化
        if os.path.exists(vec_normalize_path):
            try:
                # 由于VecNormalize需要向量化环境，我们手动处理归一化
                print("ℹ️  注意：回测环境将使用原始数据，未应用训练时的归一化")
            except Exception as e:
                print(f"⚠️  归一化参数加载失败: {e}")
        
        # 运行回测
        print("🚀 开始回测...")
        reset_result = env.reset()
        
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
            
        done = False
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # ✅ 正确处理step()返回值 - 支持新旧API格式
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            step_count += 1
            
            if step_count % 1000 == 0:
                current_balance = info.get('portfolio_value', 0)
                total_trades = info.get('total_trades', 0)
                print(f"📊 步数: {step_count:,}, 当前资产: ${current_balance:.2f}, 总交易次数: {total_trades}")
        
        # 获取回测结果
        trade_summary = env.get_trade_summary()
        
        print("\n" + "="*60)
        print("📋 回测结果摘要:")
        print("="*60)
        
        # 基础统计
        print(f"📈 总收益率: {trade_summary.get('total_return', 0)*100:.2f}%")
        print(f"💰 最终资产: ${trade_summary.get('final_balance', 0):.2f}")
        print(f"📊 总交易次数: {trade_summary.get('total_trades', 0)}")
        print(f"🎯 胜率: {trade_summary.get('win_rate', 0)*100:.2f}%")
        print(f"📉 最大回撤: {trade_summary.get('max_drawdown', 0)*100:.2f}%")
        print(f"📏 夏普比率: {trade_summary.get('sharpe_ratio', 0):.3f}")
        print(f"⚖️  盈亏比: {trade_summary.get('profit_factor', 0):.2f}")
        
        # 数据来源说明
        print(f"\n🔍 回测数据信息:")
        if os.path.exists(test_data_path):
            print("✅ 使用独立测试数据集 - 无数据泄露风险")
        else:
            print("⚠️  使用完整数据集 - 可能包含训练数据")
        
        print(f"📅 回测时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"📊 回测数据量: {len(df):,} 条记录")
        
        # 创建可视化分析
        print(f"\n🎨 生成可视化分析报告...")
        try:
            from analysis.trading_visualizer import TradingVisualizer
            
            visualizer = TradingVisualizer()
            
            # 获取交易历史和组合价值历史
            trade_history = env.trade_history
            portfolio_history = env.portfolio_history
            
            # 生成回测分析图表
            chart_path = visualizer.create_backtest_analysis_chart(
                df=df,
                trade_history=trade_history,
                portfolio_history=portfolio_history
            )
            
            if chart_path:
                print(f"📊 可视化报告已生成: {chart_path}")
                print("🌐 请用浏览器打开HTML文件查看详细分析")
            
            # 生成性能分析图表
            performance_chart = visualizer.create_performance_analysis_chart(trade_summary)
            if performance_chart:
                print(f"📈 性能分析图表: {performance_chart}")
            
        except Exception as e:
            print(f"⚠️  可视化生成失败: {e}")
        
        print("="*60)
        return trade_summary
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def live_trading():
    """实时交易模式"""
    print("🔴 实时交易模式")
    print("💡 这个功能需要连接实际的交易所API")
    print("📝 建议先在模拟环境中充分测试策略")
    
    # 可以在这里集成 deployment/live_trader.py
    try:
        from deployment.live_trader import LiveTrader
        print("✅ 实时交易模块可用")
        print("📋 使用方法:")
        print("   1. 配置API密钥")
        print("   2. 设置风险参数")
        print("   3. 启动实时交易")
    except ImportError:
        print("⚠️ 实时交易模块未安装")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SOL/USDT中频合约交易策略 - 优化版')
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'backtest', 'live'], 
                       default='collect', help='运行模式')
    parser.add_argument('--model-path', type=str, help='模型路径（用于回测）')
    parser.add_argument('--force-refresh', action='store_true', help='强制刷新缓存')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_directories()
    
    # 初始化日志
    logger = get_logger('Main', 'main.log')
    logger.info(f"启动 SOL/USDT 交易策略 - 模式: {args.mode}")
    
    print("🚀 SOL/USDT 中频合约交易策略 - 优化版")
    print("=" * 60)
    print(f"🎯 运行模式: {args.mode}")
    if args.force_refresh:
        print("🔄 强制刷新缓存")
    
    try:
        start_time = time.time()
        
        if args.mode == 'collect':
            collect_data()
            
        elif args.mode == 'train':
            # 先收集数据
            df = collect_data()
            if df is not None:
                model_path = train_model(df)
                if model_path:
                    print(f"\n🎉 训练成功完成！")
                    print(f"💾 模型保存在: {model_path}")
                    print(f"📊 启动TensorBoard: tensorboard --logdir=logs/tensorboard/")
            
        elif args.mode == 'backtest':
            results = backtest_model(model_path=args.model_path)
            if results:
                print(f"\n🎉 回测成功完成！")
                print(f"📊 查看结果图表了解详细分析")
            
        elif args.mode == 'live':
            live_trading()
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️  总耗时: {duration:.2f}秒")
            
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        logger.exception(f"程序执行出错: {e}")
        print(f"❌ 程序执行出错: {e}")
    
    print("\n👋 程序结束")

if __name__ == "__main__":
    main() 