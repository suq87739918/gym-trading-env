"""
可解释性分析器 - 基于SHAP的强化学习模型解释
提供特征贡献度分析、决策解释、可视化等功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from utils.config import get_config
from utils.logger import get_logger

class RLModelExplainer:
    """强化学习模型可解释性分析器"""
    
    def __init__(self, model=None, feature_names: List[str] = None):
        self.config = get_config()
        self.logger = get_logger('ModelExplainer', 'model_explainer.log')
        
        self.model = model
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.shap_values = None
        self.decision_records = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 配色方案
        self.colors = {
            'positive': '#2E8B57',  # 正贡献
            'negative': '#DC143C',  # 负贡献
            'neutral': '#808080',   # 中性
            'action_buy': '#00FF00',
            'action_sell': '#FF0000',
            'action_hold': '#0000FF'
        }
    
    def setup_shap_explainer(self, background_data: np.ndarray, 
                           explainer_type: str = 'permutation') -> bool:
        """
        设置SHAP解释器
        
        Args:
            background_data: 背景数据样本
            explainer_type: 解释器类型 ('permutation', 'kernel', 'deep')
        """
        try:
            self.logger.info(f"🔧 设置SHAP解释器，类型: {explainer_type}")
            
            if explainer_type == 'permutation':
                # 对于策略网络，使用Permutation解释器
                self.shap_explainer = shap.Explainer(
                    self._model_predict_wrapper, 
                    background_data,
                    feature_names=self.feature_names
                )
            elif explainer_type == 'kernel':
                # Kernel解释器适用于任何模型
                self.shap_explainer = shap.KernelExplainer(
                    self._model_predict_wrapper,
                    background_data
                )
            elif explainer_type == 'deep':
                # 深度学习模型专用（需要模型支持）
                if hasattr(self.model, 'get_layer'):
                    self.shap_explainer = shap.DeepExplainer(
                        self.model, background_data
                    )
                else:
                    self.logger.warning("模型不支持DeepExplainer，回退到Permutation")
                    self.shap_explainer = shap.Explainer(
                        self._model_predict_wrapper, background_data
                    )
            
            self.logger.info("✅ SHAP解释器设置完成")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SHAP解释器设置失败: {e}")
            return False
    
    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """模型预测包装器，适配SHAP输入格式"""
        try:
            if hasattr(self.model, 'predict'):
                # 标准模型预测接口
                predictions = self.model.predict(X)
            elif hasattr(self.model, 'forward'):
                # PyTorch模型
                import torch
                with torch.no_grad():
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    predictions = self.model.forward(X).cpu().numpy()
            elif hasattr(self.model, '__call__'):
                # 可调用对象
                predictions = self.model(X)
            else:
                raise ValueError("模型不支持预测接口")
            
            # 确保输出格式正确
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"模型预测包装器错误: {e}")
            return np.zeros((X.shape[0], 1))
    
    def analyze_single_decision(self, state: np.ndarray, 
                              action_taken: int = None,
                              context_info: Dict = None) -> Dict:
        """
        分析单个决策的SHAP值
        
        Args:
            state: 决策状态特征
            action_taken: 实际采取的动作
            context_info: 上下文信息（价格、时间等）
        
        Returns:
            包含SHAP分析结果的字典
        """
        try:
            if self.shap_explainer is None:
                self.logger.error("SHAP解释器未初始化")
                return {}
            
            # 确保输入格式正确
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # 计算SHAP值
            shap_values = self.shap_explainer(state)
            
            # 提取特征贡献
            feature_contributions = {}
            if hasattr(shap_values, 'values'):
                contributions = shap_values.values[0]  # 第一个样本
                if len(contributions.shape) > 1:
                    # 多输出情况，选择对应动作的输出
                    if action_taken is not None and action_taken < contributions.shape[1]:
                        contributions = contributions[:, action_taken]
                    else:
                        contributions = contributions[:, 0]  # 默认第一个输出
                
                for i, contrib in enumerate(contributions):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                    feature_contributions[feature_name] = float(contrib)
            
            # 计算基准值
            base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0
            
            # 预测值
            predicted_value = base_value + sum(feature_contributions.values())
            
            # 构建分析结果
            analysis_result = {
                'timestamp': context_info.get('timestamp') if context_info else None,
                'state_features': state.flatten().tolist(),
                'action_taken': action_taken,
                'predicted_value': predicted_value,
                'base_value': base_value,
                'feature_contributions': feature_contributions,
                'top_positive_features': self._get_top_features(feature_contributions, positive=True),
                'top_negative_features': self._get_top_features(feature_contributions, positive=False),
                'context_info': context_info or {}
            }
            
            # 记录决策
            self.decision_records.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"单决策SHAP分析失败: {e}")
            return {}
    
    def analyze_episode_decisions(self, states: np.ndarray, 
                                actions: List[int],
                                context_infos: List[Dict] = None) -> List[Dict]:
        """
        分析整个回合的决策序列
        
        Args:
            states: 状态序列
            actions: 动作序列
            context_infos: 上下文信息序列
        
        Returns:
            决策分析结果列表
        """
        try:
            self.logger.info(f"📊 分析回合决策序列，包含{len(states)}个决策点")
            
            results = []
            context_infos = context_infos or [{}] * len(states)
            
            for i, (state, action) in enumerate(zip(states, actions)):
                context = context_infos[i] if i < len(context_infos) else {}
                context['step'] = i
                
                result = self.analyze_single_decision(state, action, context)
                if result:
                    results.append(result)
            
            self.logger.info(f"✅ 完成{len(results)}个决策点的分析")
            return results
            
        except Exception as e:
            self.logger.error(f"回合决策分析失败: {e}")
            return []
    
    def _get_top_features(self, feature_contributions: Dict, 
                         positive: bool = True, top_k: int = 5) -> List[Tuple[str, float]]:
        """获取贡献度最高的特征"""
        filtered_contribs = {
            k: v for k, v in feature_contributions.items() 
            if (v > 0) == positive
        }
        
        sorted_features = sorted(
            filtered_contribs.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return sorted_features[:top_k]
    
    def create_decision_explanation_dashboard(self, decision_analysis: Dict,
                                            save_path: str = None) -> str:
        """
        创建单个决策的解释看板
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    '特征贡献度瀑布图', '特征重要性排序',
                    '正负贡献分布', '决策上下文信息'
                ],
                specs=[
                    [{"type": "waterfall"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "table"}]
                ]
            )
            
            # 1. 瀑布图显示特征贡献路径
            self._add_waterfall_plot(fig, decision_analysis, row=1, col=1)
            
            # 2. 特征重要性条形图
            self._add_feature_importance_bar(fig, decision_analysis, row=1, col=2)
            
            # 3. 正负贡献分布
            self._add_contribution_distribution(fig, decision_analysis, row=2, col=1)
            
            # 4. 上下文信息表格
            self._add_context_table(fig, decision_analysis, row=2, col=2)
            
            # 更新布局
            action_name = self._get_action_name(decision_analysis.get('action_taken', 0))
            fig.update_layout(
                title=f'决策解释分析 - 动作: {action_name}',
                height=800,
                showlegend=True
            )
            
            # 保存
            if save_path is None:
                save_path = f"results/decision_explanation.html"
            
            fig.write_html(save_path)
            self.logger.info(f"💾 决策解释看板已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建决策解释看板失败: {e}")
            return None
    
    def _add_waterfall_plot(self, fig, decision_analysis: Dict, row: int, col: int):
        """添加瀑布图"""
        try:
            contributions = decision_analysis.get('feature_contributions', {})
            base_value = decision_analysis.get('base_value', 0)
            
            # 准备瀑布图数据
            features = list(contributions.keys())
            values = list(contributions.values())
            
            # 只显示贡献度最大的前10个特征
            if len(features) > 10:
                sorted_indices = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:10]
                features = [features[i] for i in sorted_indices]
                values = [values[i] for i in sorted_indices]
            
            # 创建瀑布图
            fig.add_trace(
                go.Waterfall(
                    name="特征贡献",
                    orientation="v",
                    measure=["relative"] * len(features) + ["total"],
                    x=features + ["预测值"],
                    textposition="outside",
                    text=[f"{v:.3f}" for v in values] + [f"{sum(values) + base_value:.3f}"],
                    y=values + [base_value],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": self.colors['positive']}},
                    decreasing={"marker": {"color": self.colors['negative']}},
                    totals={"marker": {"color": self.colors['neutral']}}
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加瀑布图失败: {e}")
    
    def _add_feature_importance_bar(self, fig, decision_analysis: Dict, row: int, col: int):
        """添加特征重要性条形图"""
        try:
            contributions = decision_analysis.get('feature_contributions', {})
            
            # 按绝对值排序
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            features, values = zip(*sorted_features) if sorted_features else ([], [])
            
            # 设置颜色
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            
            fig.add_trace(
                go.Bar(
                    x=list(values),
                    y=list(features),
                    orientation='h',
                    name='特征贡献',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加特征重要性条形图失败: {e}")
    
    def _add_contribution_distribution(self, fig, decision_analysis: Dict, row: int, col: int):
        """添加贡献分布图"""
        try:
            contributions = decision_analysis.get('feature_contributions', {})
            values = list(contributions.values())
            
            if not values:
                return
            
            positive_values = [v for v in values if v > 0]
            negative_values = [v for v in values if v < 0]
            
            fig.add_trace(
                go.Bar(
                    x=['正贡献', '负贡献'],
                    y=[sum(positive_values), abs(sum(negative_values))],
                    name='贡献汇总',
                    marker_color=[self.colors['positive'], self.colors['negative']],
                    text=[f"{sum(positive_values):.3f}", f"{sum(negative_values):.3f}"],
                    textposition='auto'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加贡献分布图失败: {e}")
    
    def _add_context_table(self, fig, decision_analysis: Dict, row: int, col: int):
        """添加上下文信息表格"""
        try:
            context = decision_analysis.get('context_info', {})
            
            # 准备表格数据
            headers = ['信息类型', '值']
            cells = []
            
            # 基本信息
            cells.append(['动作', self._get_action_name(decision_analysis.get('action_taken', 0))])
            cells.append(['预测值', f"{decision_analysis.get('predicted_value', 0):.4f}"])
            cells.append(['基准值', f"{decision_analysis.get('base_value', 0):.4f}"])
            
            # 上下文信息
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    cells.append([str(key), f"{value:.4f}"])
                else:
                    cells.append([str(key), str(value)])
            
            # 转置数据
            if cells:
                cell_values = list(zip(*cells))
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=headers, fill_color='lightgray'),
                        cells=dict(values=cell_values, align='left')
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.error(f"添加上下文表格失败: {e}")
    
    def _get_action_name(self, action: int) -> str:
        """获取动作名称"""
        action_names = {0: '持有', 1: '开多', 2: '开空', 3: '平仓'}
        return action_names.get(action, f'动作{action}')
    
    def create_episode_summary_report(self, episode_results: List[Dict],
                                    save_path: str = None) -> str:
        """
        创建回合汇总分析报告
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    '决策时序分析', '特征贡献度趋势',
                    '动作分布统计', 'Top特征重要性',
                    '决策置信度分析', '关键决策点标注'
                ],
                specs=[
                    [{"secondary_y": True}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "box"}, {"secondary_y": True}]
                ]
            )
            
            # 1. 决策时序分析
            self._add_decision_timeline(fig, episode_results, row=1, col=1)
            
            # 2. 特征贡献度趋势
            self._add_feature_contribution_trends(fig, episode_results, row=1, col=2)
            
            # 3. 动作分布统计
            self._add_action_distribution(fig, episode_results, row=2, col=1)
            
            # 4. Top特征重要性
            self._add_top_features_summary(fig, episode_results, row=2, col=2)
            
            # 5. 决策置信度分析
            self._add_confidence_analysis(fig, episode_results, row=3, col=1)
            
            # 6. 关键决策点
            self._add_key_decisions(fig, episode_results, row=3, col=2)
            
            fig.update_layout(
                title='强化学习策略决策解释 - 回合汇总报告',
                height=1200,
                showlegend=True
            )
            
            if save_path is None:
                save_path = f"results/episode_explanation_report.html"
            
            fig.write_html(save_path)
            self.logger.info(f"💾 回合分析报告已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建回合汇总报告失败: {e}")
            return None
    
    def _add_decision_timeline(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加决策时序分析"""
        try:
            steps = [r.get('context_info', {}).get('step', i) for i, r in enumerate(episode_results)]
            actions = [r.get('action_taken', 0) for r in episode_results]
            predicted_values = [r.get('predicted_value', 0) for r in episode_results]
            
            # 动作序列
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=actions,
                    mode='markers+lines',
                    name='动作序列',
                    marker=dict(size=8),
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # 预测值序列（辅助y轴）
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=predicted_values,
                    mode='lines',
                    name='预测值',
                    line=dict(dash='dash'),
                    yaxis='y2'
                ),
                row=row, col=col, secondary_y=True
            )
            
        except Exception as e:
            self.logger.error(f"添加决策时序分析失败: {e}")
    
    def _add_feature_contribution_trends(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加特征贡献度趋势"""
        try:
            # 提取主要特征的贡献度变化
            all_features = set()
            for result in episode_results:
                all_features.update(result.get('feature_contributions', {}).keys())
            
            # 选择贡献度最大的前5个特征
            feature_importance = {}
            for feature in all_features:
                total_contrib = sum(abs(r.get('feature_contributions', {}).get(feature, 0)) 
                                  for r in episode_results)
                feature_importance[feature] = total_contrib
            
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            steps = list(range(len(episode_results)))
            
            for feature_name, _ in top_features:
                contributions = [r.get('feature_contributions', {}).get(feature_name, 0) 
                               for r in episode_results]
                
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=contributions,
                        mode='lines',
                        name=f'{feature_name}',
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.error(f"添加特征贡献度趋势失败: {e}")
    
    def _add_action_distribution(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加动作分布饼图"""
        try:
            actions = [r.get('action_taken', 0) for r in episode_results]
            action_counts = {}
            
            for action in actions:
                action_name = self._get_action_name(action)
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(action_counts.keys()),
                    values=list(action_counts.values()),
                    name='动作分布'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加动作分布失败: {e}")
    
    def _add_top_features_summary(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加Top特征汇总"""
        try:
            # 计算所有特征的平均绝对贡献度
            feature_contributions = {}
            
            for result in episode_results:
                for feature, contrib in result.get('feature_contributions', {}).items():
                    if feature not in feature_contributions:
                        feature_contributions[feature] = []
                    feature_contributions[feature].append(abs(contrib))
            
            # 计算平均值并排序
            avg_contributions = {
                feature: np.mean(contribs) 
                for feature, contribs in feature_contributions.items()
            }
            
            top_features = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)[:10]
            features, values = zip(*top_features) if top_features else ([], [])
            
            fig.add_trace(
                go.Bar(
                    x=list(values),
                    y=list(features),
                    orientation='h',
                    name='平均贡献度',
                    marker_color=self.colors['positive']
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加Top特征汇总失败: {e}")
    
    def _add_confidence_analysis(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加决策置信度分析"""
        try:
            # 基于预测值的分布分析决策置信度
            predicted_values = [r.get('predicted_value', 0) for r in episode_results]
            actions = [r.get('action_taken', 0) for r in episode_results]
            
            action_groups = {}
            for action, pred_val in zip(actions, predicted_values):
                action_name = self._get_action_name(action)
                if action_name not in action_groups:
                    action_groups[action_name] = []
                action_groups[action_name].append(pred_val)
            
            for action_name, values in action_groups.items():
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=action_name,
                        boxpoints='outliers'
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.error(f"添加置信度分析失败: {e}")
    
    def _add_key_decisions(self, fig, episode_results: List[Dict], row: int, col: int):
        """添加关键决策点分析"""
        try:
            # 找出贡献度最大的决策点
            steps = []
            total_contributions = []
            
            for i, result in enumerate(episode_results):
                contributions = result.get('feature_contributions', {})
                total_contrib = sum(abs(v) for v in contributions.values())
                
                steps.append(i)
                total_contributions.append(total_contrib)
            
            # 标记关键决策点（贡献度最大的前20%）
            threshold = np.percentile(total_contributions, 80) if total_contributions else 0
            key_steps = [s for s, c in zip(steps, total_contributions) if c >= threshold]
            key_contributions = [c for c in total_contributions if c >= threshold]
            
            # 总贡献度曲线
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=total_contributions,
                    mode='lines',
                    name='总贡献度',
                    line=dict(color='blue', width=2)
                ),
                row=row, col=col
            )
            
            # 关键决策点
            fig.add_trace(
                go.Scatter(
                    x=key_steps,
                    y=key_contributions,
                    mode='markers',
                    name='关键决策点',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='star'
                    )
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加关键决策点分析失败: {e}")
    
    def get_explainability_summary(self) -> Dict:
        """获取可解释性分析摘要"""
        try:
            if not self.decision_records:
                return {"error": "无决策记录可分析"}
            
            # 计算总体统计
            all_contributions = {}
            action_distribution = {}
            
            for record in self.decision_records:
                # 特征贡献统计
                for feature, contrib in record.get('feature_contributions', {}).items():
                    if feature not in all_contributions:
                        all_contributions[feature] = []
                    all_contributions[feature].append(abs(contrib))
                
                # 动作分布统计
                action = record.get('action_taken', 0)
                action_name = self._get_action_name(action)
                action_distribution[action_name] = action_distribution.get(action_name, 0) + 1
            
            # 计算特征重要性排序
            feature_importance = {
                feature: {
                    'avg_contribution': np.mean(contribs),
                    'max_contribution': np.max(contribs),
                    'std_contribution': np.std(contribs),
                    'frequency': len(contribs)
                }
                for feature, contribs in all_contributions.items()
            }
            
            # 排序
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1]['avg_contribution'],
                reverse=True
            )
            
            summary = {
                'total_decisions_analyzed': len(self.decision_records),
                'action_distribution': action_distribution,
                'top_10_features': {
                    feature: stats for feature, stats in sorted_features[:10]
                },
                'feature_importance_ranking': [feature for feature, _ in sorted_features],
                'analysis_coverage': {
                    'features_analyzed': len(all_contributions),
                    'avg_features_per_decision': np.mean([
                        len(r.get('feature_contributions', {})) for r in self.decision_records
                    ]),
                    'decisions_with_explanations': len([
                        r for r in self.decision_records if r.get('feature_contributions')
                    ])
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成可解释性摘要失败: {e}")
            return {"error": str(e)}

def main():
    """测试可解释性分析器功能"""
    print("🔍 测试强化学习模型可解释性分析器")
    
    explainer = RLModelExplainer()
    print("✅ 可解释性分析器初始化完成")
    print("📋 主要功能:")
    print("  - SHAP值特征贡献分析")
    print("  - 单决策深度解释")
    print("  - 回合决策序列分析")
    print("  - 可视化解释看板")
    print("  - 决策规律发现")

if __name__ == "__main__":
    main() 