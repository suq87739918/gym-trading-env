"""
数据划分模块
负责将时序数据按照时间顺序正确划分为训练集、验证集和测试集
确保不发生数据泄露问题
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.config import get_config
from utils.logger import get_logger

class TimeSeriesDataSplitter:
    """时序数据划分器 - 防止数据泄露"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('DataSplitter', 'data_splitter.log')
        
        # 划分配置
        self.train_ratio = self.config.get('TRAIN_TEST_SPLIT_RATIO', 0.8)
        self.validation_ratio = self.config.get('VALIDATION_SPLIT_RATIO', 0.1)
        self.use_time_split = self.config.get('USE_TIME_SPLIT', True)
        self.prevent_leakage = self.config.get('PREVENT_DATA_LEAKAGE', True)
        
        # 计算测试集比例
        self.test_ratio = 1.0 - self.train_ratio - self.validation_ratio
        
        self.logger.info(f"数据划分配置: 训练集={self.train_ratio*100:.1f}%, "
                        f"验证集={self.validation_ratio*100:.1f}%, "
                        f"测试集={self.test_ratio*100:.1f}%")
    
    def split_data(self, df: pd.DataFrame, shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        将数据划分为训练集、验证集和测试集
        
        Args:
            df: 输入数据DataFrame，必须包含时间索引
            shuffle: 是否打乱数据（对时序数据不推荐）
            
        Returns:
            (train_df, val_df, test_df): 训练集、验证集、测试集
        """
        if df.empty:
            self.logger.error("输入数据为空")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # 确保数据按时间排序
        if not df.index.is_monotonic_increasing:
            self.logger.warning("数据未按时间排序，正在重新排序...")
            df = df.sort_index()
        
        total_len = len(df)
        
        if self.use_time_split:
            # 按时间顺序划分（推荐方式）
            train_df, val_df, test_df = self._time_based_split(df)
        else:
            # 随机划分（不推荐用于时序数据）
            if not shuffle:
                self.logger.warning("非时间划分模式建议启用shuffle参数")
            train_df, val_df, test_df = self._random_split(df, shuffle)
        
        # 验证划分结果
        self._validate_split(train_df, val_df, test_df, df)
        
        # 记录划分信息
        self._log_split_info(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间顺序划分数据"""
        total_len = len(df)
        
        # 计算分割点
        train_end_idx = int(total_len * self.train_ratio)
        val_end_idx = int(total_len * (self.train_ratio + self.validation_ratio))
        
        # 时间顺序划分：训练集 -> 验证集 -> 测试集
        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()
        test_df = df.iloc[val_end_idx:].copy()
        
        # 添加数据集标识
        train_df.loc[:, 'dataset_type'] = 'train'
        val_df.loc[:, 'dataset_type'] = 'validation'
        test_df.loc[:, 'dataset_type'] = 'test'
        
        return train_df, val_df, test_df
    
    def _random_split(self, df: pd.DataFrame, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """随机划分数据（不推荐用于时序数据）"""
        self.logger.warning("使用随机划分可能导致数据泄露，强烈建议使用时间划分")
        
        total_len = len(df)
        indices = np.arange(total_len)
        
        if shuffle:
            np.random.seed(42)  # 固定随机种子保证可重复性
            np.random.shuffle(indices)
        
        # 计算分割点
        train_end_idx = int(total_len * self.train_ratio)
        val_end_idx = int(total_len * (self.train_ratio + self.validation_ratio))
        
        # 获取索引
        train_indices = indices[:train_end_idx]
        val_indices = indices[train_end_idx:val_end_idx]
        test_indices = indices[val_end_idx:]
        
        # 划分数据
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        # 重新按时间排序
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()
        test_df = test_df.sort_index()
        
        # 添加数据集标识
        train_df.loc[:, 'dataset_type'] = 'train'
        val_df.loc[:, 'dataset_type'] = 'validation'
        test_df.loc[:, 'dataset_type'] = 'test'
        
        return train_df, val_df, test_df
    
    def _validate_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, original_df: pd.DataFrame):
        """验证数据划分的正确性"""
        total_original = len(original_df)
        total_split = len(train_df) + len(val_df) + len(test_df)
        
        # 检查数据总量
        if total_split != total_original:
            self.logger.error(f"数据划分错误：原始数据{total_original}行，划分后{total_split}行")
            raise ValueError("数据划分过程中丢失了数据")
        
        # 检查时间序列连续性（仅对时间划分）
        if self.use_time_split and self.prevent_leakage:
            self._check_time_leakage(train_df, val_df, test_df)
        
        # 检查数据分布
        self._check_data_distribution(train_df, val_df, test_df, original_df)
    
    def _check_time_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """检查时间泄露问题"""
        try:
            # 获取各数据集的时间范围
            train_max_time = train_df.index.max()
            val_min_time = val_df.index.min()
            val_max_time = val_df.index.max()
            test_min_time = test_df.index.min()
            
            # 检查训练集和验证集之间的时间顺序
            if train_max_time >= val_min_time:
                self.logger.warning(f"潜在时间泄露：训练集最晚时间({train_max_time}) >= 验证集最早时间({val_min_time})")
            
            # 检查验证集和测试集之间的时间顺序
            if val_max_time >= test_min_time:
                self.logger.warning(f"潜在时间泄露：验证集最晚时间({val_max_time}) >= 测试集最早时间({test_min_time})")
            
            # 如果没有问题，记录成功
            if train_max_time < val_min_time < val_max_time < test_min_time:
                self.logger.info("✅ 时间序列划分正确，无数据泄露风险")
            
        except Exception as e:
            self.logger.error(f"时间泄露检查失败: {e}")
    
    def _check_data_distribution(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame, original_df: pd.DataFrame):
        """检查数据分布"""
        try:
            # 检查价格分布
            if 'close' in original_df.columns:
                original_price_stats = original_df['close'].describe()
                train_price_stats = train_df['close'].describe()
                val_price_stats = val_df['close'].describe()
                test_price_stats = test_df['close'].describe()
                
                self.logger.info("价格分布统计:")
                self.logger.info(f"  原始数据: 均值={original_price_stats['mean']:.2f}, 标准差={original_price_stats['std']:.2f}")
                self.logger.info(f"  训练集: 均值={train_price_stats['mean']:.2f}, 标准差={train_price_stats['std']:.2f}")
                self.logger.info(f"  验证集: 均值={val_price_stats['mean']:.2f}, 标准差={val_price_stats['std']:.2f}")
                self.logger.info(f"  测试集: 均值={test_price_stats['mean']:.2f}, 标准差={test_price_stats['std']:.2f}")
                
        except Exception as e:
            self.logger.warning(f"数据分布检查失败: {e}")
    
    def _log_split_info(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """记录划分信息"""
        self.logger.info("=" * 60)
        self.logger.info("数据划分完成")
        self.logger.info(f"训练集: {len(train_df):,} 行 ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        self.logger.info(f"验证集: {len(val_df):,} 行 ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        self.logger.info(f"测试集: {len(test_df):,} 行 ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        
        if not train_df.empty and not test_df.empty:
            self.logger.info(f"训练集时间范围: {train_df.index.min()} 到 {train_df.index.max()}")
            if not val_df.empty:
                self.logger.info(f"验证集时间范围: {val_df.index.min()} 到 {val_df.index.max()}")
            self.logger.info(f"测试集时间范围: {test_df.index.min()} 到 {test_df.index.max()}")
        
        self.logger.info("=" * 60)
    
    def create_train_test_only(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        仅创建训练集和测试集（不要验证集）
        
        Args:
            df: 输入数据
            
        Returns:
            (train_df, test_df): 训练集和测试集
        """
        # 临时调整比例
        original_train_ratio = self.train_ratio
        original_val_ratio = self.validation_ratio
        
        # 将验证集比例加到训练集
        self.train_ratio = original_train_ratio + original_val_ratio
        self.validation_ratio = 0.0
        
        try:
            train_df, _, test_df = self.split_data(df)
            return train_df, test_df
        finally:
            # 恢复原始比例
            self.train_ratio = original_train_ratio
            self.validation_ratio = original_val_ratio
    
    def get_split_summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                         test_df: pd.DataFrame) -> Dict:
        """获取数据划分摘要"""
        total_samples = len(train_df) + len(val_df) + len(test_df)
        
        summary = {
            'total_samples': total_samples,
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'train_percentage': len(train_df) / total_samples * 100,
            'validation_percentage': len(val_df) / total_samples * 100,
            'test_percentage': len(test_df) / total_samples * 100,
            'split_method': 'time_based' if self.use_time_split else 'random',
            'leakage_prevention': self.prevent_leakage
        }
        
        if not train_df.empty and not test_df.empty:
            summary.update({
                'train_time_range': (train_df.index.min(), train_df.index.max()),
                'test_time_range': (test_df.index.min(), test_df.index.max())
            })
            
            if not val_df.empty:
                summary['validation_time_range'] = (val_df.index.min(), val_df.index.max())
        
        return summary

def main():
    """测试数据划分功能"""
    from data_collector import DataCollector
    
    print("🔄 测试数据划分功能")
    
    # 加载数据
    collector = DataCollector()
    df = collector.load_data()
    
    if df.empty:
        print("❌ 没有可用数据，请先运行数据收集")
        return
    
    # 创建数据划分器
    splitter = TimeSeriesDataSplitter()
    
    # 划分数据
    train_df, val_df, test_df = splitter.split_data(df)
    
    # 获取摘要
    summary = splitter.get_split_summary(train_df, val_df, test_df)
    
    print("📊 数据划分摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 