"""
决策日志记录器
记录强化学习模型的决策过程，提供可视化和调试功能
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from utils.config import get_config
from utils.logger import get_logger

class DecisionLogger:
    """决策日志记录器"""
    
    def __init__(self, save_dir: str = "logs/decisions"):
        self.config = get_config()
        self.logger = get_logger('DecisionLogger', 'decision_logger.log')
        
        self.save_dir = save_dir
        self.decision_records = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def log_decision(self, 
                    timestamp: Optional[str] = None,
                    step: int = 0,
                    state_features: Dict = None,
                    action_taken: int = 0,
                    action_probabilities: List[float] = None,
                    state_value: float = 0.0,
                    reward: float = 0.0,
                    market_data: Dict = None,
                    model_info: Dict = None,
                    explanation: Dict = None) -> Dict:
        """
        记录单个决策
        
        Args:
            timestamp: 时间戳
            step: 步骤编号
            state_features: 状态特征字典
            action_taken: 采取的动作
            action_probabilities: 动作概率分布
            state_value: 状态价值
            reward: 获得的奖励
            market_data: 市场数据
            model_info: 模型信息
            explanation: 解释信息（SHAP值等）
        
        Returns:
            记录ID
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            # 构建决策记录
            decision_record = {
                'record_id': len(self.decision_records),
                'session_id': self.session_id,
                'timestamp': timestamp,
                'step': step,
                'action_taken': action_taken,
                'action_probabilities': action_probabilities or [],
                'state_value': state_value,
                'reward': reward,
                'state_features': state_features or {},
                'market_data': market_data or {},
                'model_info': model_info or {},
                'explanation': explanation or {}
            }
            
            # 添加计算字段
            decision_record.update(self._calculate_derived_fields(decision_record))
            
            # 存储记录
            self.decision_records.append(decision_record)
            
            # 实时日志输出
            self._log_decision_summary(decision_record)
            
            return decision_record
            
        except Exception as e:
            self.logger.error(f"❌ 决策记录失败: {e}")
            return {}
    
    def _calculate_derived_fields(self, record: Dict) -> Dict:
        """计算衍生字段"""
        try:
            derived = {}
            
            # 计算决策置信度
            if record.get('action_probabilities'):
                probs = record['action_probabilities']
                max_prob = max(probs) if probs else 0
                entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
                derived['decision_confidence'] = max_prob
                derived['decision_entropy'] = entropy
            
            # 计算市场状态
            market_data = record.get('market_data', {})
            if market_data:
                price = market_data.get('price', 0)
                volume = market_data.get('volume', 0)
                derived['price'] = price
                derived['volume'] = volume
                
                # 价格变化率
                if hasattr(self, '_last_price') and self._last_price > 0:
                    derived['price_change_pct'] = (price - self._last_price) / self._last_price
                else:
                    derived['price_change_pct'] = 0.0
                self._last_price = price
            
            # 动作类型标记
            action = record.get('action_taken', 0)
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            derived['action_name'] = action_names.get(action, f'ACTION_{action}')
            
            # 特征统计
            features = record.get('state_features', {})
            if features:
                feature_values = [v for v in features.values() if isinstance(v, (int, float))]
                if feature_values:
                    derived['feature_mean'] = np.mean(feature_values)
                    derived['feature_std'] = np.std(feature_values)
                    derived['feature_max'] = np.max(feature_values)
                    derived['feature_min'] = np.min(feature_values)
            
            return derived
            
        except Exception as e:
            self.logger.error(f"计算衍生字段失败: {e}")
            return {}
    
    def _log_decision_summary(self, record: Dict):
        """输出决策摘要日志"""
        try:
            step = record.get('step', 0)
            action_name = record.get('action_name', 'UNKNOWN')
            confidence = record.get('decision_confidence', 0)
            reward = record.get('reward', 0)
            state_value = record.get('state_value', 0)
            
            # 简化的特征信息
            key_features = {}
            features = record.get('state_features', {})
            
            # 选择几个关键特征显示
            important_feature_keys = ['rsi', 'price', 'volume', 'smc_signal', 'trend']
            for key in important_feature_keys:
                for feature_name, value in features.items():
                    if key.lower() in feature_name.lower():
                        key_features[feature_name] = value
                        break
            
            log_msg = (
                f"Step {step:4d}: {action_name:4s} "
                f"(conf={confidence:.3f}, r={reward:+.3f}, v={state_value:.3f})"
            )
            
            if key_features:
                feature_str = ", ".join([f"{k}={v:.3f}" for k, v in list(key_features.items())[:3]])
                log_msg += f" | {feature_str}"
            
            self.logger.info(log_msg)
            
        except Exception as e:
            self.logger.debug(f"决策摘要日志失败: {e}")
    
    def save_session_log(self, filename: str = None) -> str:
        """保存会话日志到文件"""
        try:
            if filename is None:
                filename = f"decision_log_{self.session_id}.json"
            
            filepath = f"{self.save_dir}/{filename}"
            
            # 构建会话元数据
            session_metadata = {
                'session_id': self.session_id,
                'total_decisions': len(self.decision_records),
                'session_start': self.decision_records[0]['timestamp'] if self.decision_records else None,
                'session_end': self.decision_records[-1]['timestamp'] if self.decision_records else None,
                'summary_stats': self._calculate_session_stats()
            }
            
            # 保存数据
            log_data = {
                'metadata': session_metadata,
                'decisions': self.decision_records
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ 会话日志已保存: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 保存会话日志失败: {e}")
            return None
    
    def _calculate_session_stats(self) -> Dict:
        """计算会话统计信息"""
        try:
            if not self.decision_records:
                return {}
            
            # 提取数据
            actions = [r.get('action_taken', 0) for r in self.decision_records]
            rewards = [r.get('reward', 0) for r in self.decision_records]
            confidences = [r.get('decision_confidence', 0) for r in self.decision_records]
            state_values = [r.get('state_value', 0) for r in self.decision_records]
            
            # 计算统计
            stats = {
                'total_steps': len(self.decision_records),
                'action_distribution': dict(zip(*np.unique(actions, return_counts=True))),
                'total_reward': sum(rewards),
                'avg_reward': np.mean(rewards) if rewards else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'avg_state_value': np.mean(state_values) if state_values else 0,
                'reward_volatility': np.std(rewards) if rewards else 0
            }
            
            # 动作序列分析
            action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
            stats['action_changes'] = action_changes
            stats['action_stability'] = 1 - (action_changes / max(len(actions) - 1, 1))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算会话统计失败: {e}")
            return {}
    
    def load_session_log(self, filepath: str) -> bool:
        """加载会话日志"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            self.decision_records = log_data.get('decisions', [])
            metadata = log_data.get('metadata', {})
            self.session_id = metadata.get('session_id', self.session_id)
            
            self.logger.info(f"✅ 会话日志已加载: {len(self.decision_records)}条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载会话日志失败: {e}")
            return False
    
    def create_decision_analysis_dashboard(self, save_path: str = None) -> str:
        """创建决策分析看板"""
        try:
            if not self.decision_records:
                self.logger.warning("没有决策记录可分析")
                return None
            
            self.logger.info("📊 创建决策分析看板...")
            
            # 转换为DataFrame方便分析
            df = pd.DataFrame(self.decision_records)
            
            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    '决策序列时间线', '动作分布统计',
                    '奖励累积曲线', '决策置信度分析',
                    '特征重要性趋势', '模型状态价值预测'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "pie"}],
                    [{"secondary_y": True}, {"type": "box"}],
                    [{"secondary_y": True}, {"secondary_y": True}]
                ]
            )
            
            # 1. 决策序列时间线
            self._add_decision_timeline(fig, df, row=1, col=1)
            
            # 2. 动作分布饼图
            self._add_action_distribution_pie(fig, df, row=1, col=2)
            
            # 3. 奖励累积曲线
            self._add_reward_analysis(fig, df, row=2, col=1)
            
            # 4. 决策置信度箱图
            self._add_confidence_analysis(fig, df, row=2, col=2)
            
            # 5. 特征重要性趋势
            self._add_feature_trends(fig, df, row=3, col=1)
            
            # 6. 状态价值预测
            self._add_value_prediction_analysis(fig, df, row=3, col=2)
            
            # 更新布局
            fig.update_layout(
                title=f'决策分析看板 - Session {self.session_id}',
                height=1200,
                showlegend=True
            )
            
            # 保存
            if save_path is None:
                save_path = f"{self.save_dir}/decision_dashboard_{self.session_id}.html"
            
            fig.write_html(save_path)
            self.logger.info(f"✅ 决策分析看板已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 创建决策分析看板失败: {e}")
            return None
    
    def _add_decision_timeline(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加决策时间线"""
        try:
            steps = df['step'].values
            actions = df['action_taken'].values
            rewards = df['reward'].values
            
            # 动作序列
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=actions,
                    mode='markers+lines',
                    name='Action Sequence',
                    marker=dict(
                        size=8,
                        color=actions,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Action")
                    ),
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # 奖励序列（辅助y轴）
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=rewards,
                    mode='lines',
                    name='Rewards',
                    line=dict(color='red', dash='dash'),
                    yaxis='y2'
                ),
                row=row, col=col, secondary_y=True
            )
            
        except Exception as e:
            self.logger.error(f"添加决策时间线失败: {e}")
    
    def _add_action_distribution_pie(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加动作分布饼图"""
        try:
            action_counts = df['action_taken'].value_counts()
            action_names = [f'Action {i}' for i in action_counts.index]
            
            fig.add_trace(
                go.Pie(
                    labels=action_names,
                    values=action_counts.values,
                    name="Action Distribution"
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加动作分布饼图失败: {e}")
    
    def _add_reward_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加奖励分析"""
        try:
            steps = df['step'].values
            rewards = df['reward'].values
            cumulative_rewards = np.cumsum(rewards)
            
            # 单步奖励
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=rewards,
                    mode='lines',
                    name='Step Rewards',
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                row=row, col=col
            )
            
            # 累积奖励
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=cumulative_rewards,
                    mode='lines',
                    name='Cumulative Rewards',
                    line=dict(color='red', width=3),
                    yaxis='y2'
                ),
                row=row, col=col, secondary_y=True
            )
            
        except Exception as e:
            self.logger.error(f"添加奖励分析失败: {e}")
    
    def _add_confidence_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加置信度分析"""
        try:
            if 'decision_confidence' in df.columns:
                # 按动作分组的置信度分布
                for action in df['action_taken'].unique():
                    action_df = df[df['action_taken'] == action]
                    confidence_values = action_df['decision_confidence'].values
                    
                    fig.add_trace(
                        go.Box(
                            y=confidence_values,
                            name=f'Action {action}',
                            boxpoints='outliers'
                        ),
                        row=row, col=col
                    )
            
        except Exception as e:
            self.logger.error(f"添加置信度分析失败: {e}")
    
    def _add_feature_trends(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加特征趋势"""
        try:
            steps = df['step'].values
            
            # 尝试提取一些关键特征
            key_features = ['feature_mean', 'feature_std', 'price_change_pct']
            
            for feature in key_features:
                if feature in df.columns:
                    values = df[feature].values
                    
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=values,
                            mode='lines',
                            name=feature,
                            line=dict(width=2),
                            opacity=0.8
                        ),
                        row=row, col=col
                    )
            
        except Exception as e:
            self.logger.error(f"添加特征趋势失败: {e}")
    
    def _add_value_prediction_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加价值预测分析"""
        try:
            steps = df['step'].values
            state_values = df['state_value'].values
            rewards = df['reward'].values
            
            # 状态价值预测
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=state_values,
                    mode='lines',
                    name='Predicted State Value',
                    line=dict(color='green', width=2)
                ),
                row=row, col=col
            )
            
            # 实际奖励（用于对比）
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=rewards,
                    mode='markers',
                    name='Actual Rewards',
                    marker=dict(color='red', size=4),
                    yaxis='y2'
                ),
                row=row, col=col, secondary_y=True
            )
            
        except Exception as e:
            self.logger.error(f"添加价值预测分析失败: {e}")
    
    def analyze_decision_patterns(self) -> Dict:
        """分析决策模式"""
        try:
            if not self.decision_records:
                return {"error": "没有决策记录可分析"}
            
            df = pd.DataFrame(self.decision_records)
            
            analysis = {
                'session_summary': self._calculate_session_stats(),
                'decision_patterns': {},
                'performance_metrics': {},
                'anomaly_detection': {}
            }
            
            # 决策模式分析
            actions = df['action_taken'].values
            
            # 动作转移矩阵
            n_actions = len(np.unique(actions))
            transition_matrix = np.zeros((n_actions, n_actions))
            
            for i in range(len(actions) - 1):
                current_action = actions[i]
                next_action = actions[i + 1]
                transition_matrix[current_action, next_action] += 1
            
            # 归一化
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                        out=np.zeros_like(transition_matrix), where=row_sums[:, np.newaxis]!=0)
            
            analysis['decision_patterns']['transition_matrix'] = transition_matrix.tolist()
            
            # 决策持续性分析
            action_runs = []
            current_run = 1
            for i in range(1, len(actions)):
                if actions[i] == actions[i-1]:
                    current_run += 1
                else:
                    action_runs.append(current_run)
                    current_run = 1
            action_runs.append(current_run)
            
            analysis['decision_patterns']['avg_action_duration'] = np.mean(action_runs)
            analysis['decision_patterns']['max_action_duration'] = np.max(action_runs)
            
            # 性能指标
            if 'reward' in df.columns:
                rewards = df['reward'].values
                analysis['performance_metrics'] = {
                    'total_reward': float(np.sum(rewards)),
                    'avg_reward_per_step': float(np.mean(rewards)),
                    'reward_volatility': float(np.std(rewards)),
                    'max_reward': float(np.max(rewards)),
                    'min_reward': float(np.min(rewards)),
                    'positive_reward_ratio': float(np.mean(rewards > 0))
                }
            
            # 异常检测
            if 'decision_confidence' in df.columns:
                confidences = df['decision_confidence'].values
                confidence_threshold = np.percentile(confidences, 10)  # 底部10%
                low_confidence_decisions = np.sum(confidences < confidence_threshold)
                
                analysis['anomaly_detection'] = {
                    'low_confidence_decisions': int(low_confidence_decisions),
                    'low_confidence_ratio': float(low_confidence_decisions / len(confidences)),
                    'avg_confidence': float(np.mean(confidences)),
                    'confidence_trend': self._calculate_trend(confidences)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"决策模式分析失败: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """计算趋势"""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # 简单线性回归计算趋势
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except:
            return "unknown"
    
    def export_decision_summary(self, format: str = 'csv') -> str:
        """导出决策摘要"""
        try:
            if not self.decision_records:
                self.logger.warning("没有决策记录可导出")
                return None
            
            df = pd.DataFrame(self.decision_records)
            
            # 选择关键列
            summary_columns = [
                'step', 'action_taken', 'action_name', 'decision_confidence',
                'reward', 'state_value', 'price', 'volume'
            ]
            
            available_columns = [col for col in summary_columns if col in df.columns]
            summary_df = df[available_columns]
            
            # 导出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'csv':
                filepath = f"{self.save_dir}/decision_summary_{timestamp}.csv"
                summary_df.to_csv(filepath, index=False, encoding='utf-8')
            elif format.lower() == 'excel':
                filepath = f"{self.save_dir}/decision_summary_{timestamp}.xlsx"
                summary_df.to_excel(filepath, index=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            self.logger.info(f"✅ 决策摘要已导出: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ 导出决策摘要失败: {e}")
            return None
    
    def get_recent_decisions(self, n: int = 10) -> List[Dict]:
        """获取最近的决策记录"""
        return self.decision_records[-n:] if len(self.decision_records) >= n else self.decision_records
    
    def clear_session(self):
        """清空当前会话"""
        self.decision_records.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info("🔄 会话已重置")

def main():
    """测试决策日志记录器"""
    print("📝 测试决策日志记录器")
    
    # 创建日志记录器
    logger = DecisionLogger()
    
    # 模拟一些决策记录
    print("📊 模拟决策记录...")
    np.random.seed(42)
    
    for step in range(50):
        # 模拟决策数据
        action = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
        action_probs = np.random.dirichlet([1, 1, 1, 1])
        reward = np.random.normal(0, 0.1)
        state_value = np.random.normal(0, 1)
        
        # 模拟市场数据
        price = 100 + step * 0.1 + np.random.normal(0, 0.5)
        volume = np.random.lognormal(8, 0.3)
        
        # 模拟特征
        features = {
            'rsi': np.random.uniform(20, 80),
            'sma_20': price * np.random.uniform(0.98, 1.02),
            'volume_sma': volume * np.random.uniform(0.8, 1.2),
            'volatility': np.random.uniform(0.01, 0.05)
        }
        
        # 记录决策
        logger.log_decision(
            step=step,
            state_features=features,
            action_taken=action,
            action_probabilities=action_probs.tolist(),
            state_value=state_value,
            reward=reward,
            market_data={'price': price, 'volume': volume}
        )
    
    print(f"✅ 已记录 {len(logger.decision_records)} 个决策")
    
    # 保存会话日志
    print("💾 保存会话日志...")
    log_file = logger.save_session_log()
    
    # 创建分析看板
    print("📊 创建分析看板...")
    dashboard_file = logger.create_decision_analysis_dashboard()
    if dashboard_file:
        print(f"📈 分析看板已保存: {dashboard_file}")
    
    # 分析决策模式
    print("🔍 分析决策模式...")
    patterns = logger.analyze_decision_patterns()
    print("📋 决策模式分析结果:")
    for key, value in patterns.items():
        if key != 'decision_patterns':  # 跳过复杂的嵌套数据
            print(f"  {key}: {value}")
    
    # 导出摘要
    print("📤 导出决策摘要...")
    summary_file = logger.export_decision_summary()
    if summary_file:
        print(f"📄 摘要已导出: {summary_file}")
    
    print("✅ 决策日志记录器测试完成！")

if __name__ == "__main__":
    main() 