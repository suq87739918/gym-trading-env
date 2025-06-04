"""
特征分析器 - 特征监控和可视化工具
提供特征重要性分析、相关性热图、特征分布分析等功能
帮助优化31个特征，提升模型性能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from utils.config import get_config
from utils.logger import get_logger

class FeatureAnalyzer:
    """特征分析器 - 监控和优化特征质量"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('FeatureAnalyzer', 'feature_analyzer.log')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 配色方案
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#17becf'
        }
    
    def create_comprehensive_feature_report(self, df: pd.DataFrame, 
                                          feature_importance: Dict = None,
                                          save_path: str = None) -> str:
        """
        ✅ 创建全面的特征分析报告
        """
        try:
            self.logger.info("📊 开始创建特征分析报告...")
            
            # 创建子图
            fig = make_subplots(
                rows=4, cols=3,
                subplot_titles=[
                    '特征重要性排序', '特征相关性热图', '特征分布概览',
                    '特征缺失值分析', 'PCA方差解释', '特征聚类分析',
                    '特征稳定性分析', '特征偏度分析', '特征异常值检测',
                    '时间序列特征变化', '特征互信息矩阵', '特征选择建议'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "heatmap"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "line"}, {"type": "bar"}, {"type": "box"}],
                    [{"type": "line"}, {"type": "heatmap"}, {"type": "table"}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # 1. 特征重要性分析
            self._add_feature_importance_plot(fig, feature_importance, row=1, col=1)
            
            # 2. 相关性热图
            self._add_correlation_heatmap(fig, df, row=1, col=2)
            
            # 3. 特征分布
            self._add_feature_distribution_plot(fig, df, row=1, col=3)
            
            # 4. 缺失值分析
            self._add_missing_values_analysis(fig, df, row=2, col=1)
            
            # 5. PCA分析
            self._add_pca_analysis(fig, df, row=2, col=2)
            
            # 6. 特征聚类
            self._add_feature_clustering(fig, df, row=2, col=3)
            
            # 7. 特征稳定性
            self._add_stability_analysis(fig, df, row=3, col=1)
            
            # 8. 偏度分析
            self._add_skewness_analysis(fig, df, row=3, col=2)
            
            # 9. 异常值检测
            self._add_outlier_detection(fig, df, row=3, col=3)
            
            # 10. 时间序列变化
            self._add_temporal_analysis(fig, df, row=4, col=1)
            
            # 11. 互信息矩阵
            self._add_mutual_information_matrix(fig, df, row=4, col=2)
            
            # 12. 特征选择建议
            self._add_feature_selection_recommendations(fig, df, feature_importance, row=4, col=3)
            
            # 更新布局
            fig.update_layout(
                title={
                    'text': 'SOL/USDT 特征工程全面分析报告',
                    'x': 0.5,
                    'font': {'size': 20}
                },
                height=1600,
                showlegend=False,
                template='plotly_white'
            )
            
            # 保存报告
            if save_path is None:
                save_path = f"results/feature_analysis_report.html"
            
            fig.write_html(save_path)
            self.logger.info(f"✅ 特征分析报告已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.exception(f"❌ 创建特征分析报告失败: {e}")
            return None
    
    def _add_feature_importance_plot(self, fig, feature_importance: Dict, row: int, col: int):
        """添加特征重要性图"""
        try:
            if not feature_importance or 'combined_score' not in feature_importance:
                return
            
            # 获取top20特征
            scores = feature_importance['combined_score']
            top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
            
            features, importance = zip(*top_features)
            
            fig.add_trace(
                go.Bar(
                    x=list(importance),
                    y=list(features),
                    orientation='h',
                    name='特征重要性',
                    marker_color=self.colors['primary']
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="重要性评分", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加特征重要性图失败: {e}")
    
    def _add_correlation_heatmap(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加相关性热图"""
        try:
            # 获取数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # 限制显示前20个
            corr_matrix = df[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name='相关性'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加相关性热图失败: {e}")
    
    def _add_feature_distribution_plot(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加特征分布图"""
        try:
            # 随机选择几个特征展示分布
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            sample_cols = np.random.choice(numeric_cols, min(5, len(numeric_cols)), replace=False)
            
            for i, col in enumerate(sample_cols):
                fig.add_trace(
                    go.Histogram(
                        x=df[col].dropna(),
                        name=f'{col}',
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="特征值", row=row, col=col)
            fig.update_yaxes(title_text="频次", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加特征分布图失败: {e}")
    
    def _add_missing_values_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加缺失值分析"""
        try:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        name='缺失值数量',
                        marker_color=self.colors['warning']
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="特征", row=row, col=col)
            fig.update_yaxes(title_text="缺失值数量", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加缺失值分析失败: {e}")
    
    def _add_pca_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加PCA分析"""
        try:
            # 获取数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].fillna(0)
            
            # 进行PCA
            pca = PCA(n_components=min(10, len(numeric_cols)))
            pca.fit(X)
            
            # 累积方差解释率
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumulative_variance) + 1)),
                    y=cumulative_variance,
                    mode='lines+markers',
                    name='累积方差解释率',
                    line=dict(color=self.colors['success'])
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="主成分数量", row=row, col=col)
            fig.update_yaxes(title_text="累积方差解释率", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加PCA分析失败: {e}")
    
    def _add_feature_clustering(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加特征聚类分析"""
        try:
            # 计算特征间的相关性
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr().abs()
            
            # 使用相关性进行聚类
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # 转换为距离矩阵
            distance_matrix = 1 - corr_matrix
            condensed_distances = squareform(distance_matrix)
            
            # 进行层次聚类
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # 这里简化显示，实际可以用dendrogram
            # 改为显示聚类质量评估
            silhouette_scores = []
            K_range = range(2, min(10, len(numeric_cols)))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(corr_matrix)
                silhouette_avg = silhouette_score(corr_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            fig.add_trace(
                go.Scatter(
                    x=list(K_range),
                    y=silhouette_scores,
                    mode='lines+markers',
                    name='轮廓系数',
                    line=dict(color=self.colors['info'])
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="聚类数量", row=row, col=col)
            fig.update_yaxes(title_text="轮廓系数", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加特征聚类分析失败: {e}")
    
    def _add_stability_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加特征稳定性分析"""
        try:
            # 计算滚动标准差来评估稳定性
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # 选择前5个特征
            
            for col in numeric_cols:
                rolling_std = df[col].rolling(100).std()
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rolling_std,
                        mode='lines',
                        name=f'{col}_stability',
                        opacity=0.7
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="时间", row=row, col=col)
            fig.update_yaxes(title_text="滚动标准差", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加稳定性分析失败: {e}")
    
    def _add_skewness_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加偏度分析"""
        try:
            from scipy.stats import skew
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            skewness_values = []
            feature_names = []
            
            for col in numeric_cols:
                skew_val = skew(df[col].dropna())
                skewness_values.append(skew_val)
                feature_names.append(col)
            
            # 只显示前20个
            if len(skewness_values) > 20:
                indices = np.argsort(np.abs(skewness_values))[-20:]
                skewness_values = [skewness_values[i] for i in indices]
                feature_names = [feature_names[i] for i in indices]
            
            colors = [self.colors['warning'] if abs(s) > 1 else self.colors['primary'] for s in skewness_values]
            
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=skewness_values,
                    name='偏度',
                    marker_color=colors
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="特征", row=row, col=col)
            fig.update_yaxes(title_text="偏度值", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加偏度分析失败: {e}")
    
    def _add_outlier_detection(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加异常值检测"""
        try:
            # 选择几个特征进行异常值检测
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
            
            outlier_counts = []
            feature_names = []
            
            for col in numeric_cols:
                data = df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_counts.append(len(outliers))
                feature_names.append(col)
            
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=outlier_counts,
                    name='异常值数量',
                    marker_color=self.colors['warning']
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="特征", row=row, col=col)
            fig.update_yaxes(title_text="异常值数量", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加异常值检测失败: {e}")
    
    def _add_temporal_analysis(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加时间序列特征变化分析"""
        try:
            # 计算特征的时间稳定性
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # 选择前3个特征
            
            for col in numeric_cols:
                # 计算滚动均值
                rolling_mean = df[col].rolling(100).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rolling_mean,
                        mode='lines',
                        name=f'{col}_趋势',
                        opacity=0.8
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="时间", row=row, col=col)
            fig.update_yaxes(title_text="滚动均值", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加时间序列分析失败: {e}")
    
    def _add_mutual_information_matrix(self, fig, df: pd.DataFrame, row: int, col: int):
        """添加互信息矩阵"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # 选择前10个数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
            
            # 计算互信息矩阵
            mi_matrix = np.zeros((len(numeric_cols), len(numeric_cols)))
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i == j:
                        mi_matrix[i, j] = 1.0  # 自己与自己的互信息设为1
                    elif i < j:
                        try:
                            X = df[col1].fillna(0).values.reshape(-1, 1)
                            y = df[col2].fillna(0).values
                            mi_score = mutual_info_regression(X, y, random_state=42)[0]
                            mi_matrix[i, j] = mi_score
                            mi_matrix[j, i] = mi_score  # 对称矩阵
                        except:
                            mi_matrix[i, j] = 0
                            mi_matrix[j, i] = 0
            
            fig.add_trace(
                go.Heatmap(
                    z=mi_matrix,
                    x=numeric_cols,
                    y=numeric_cols,
                    colorscale='Viridis',
                    name='互信息'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"添加互信息矩阵失败: {e}")
    
    def _add_feature_selection_recommendations(self, fig, df: pd.DataFrame, 
                                             feature_importance: Dict, row: int, col: int):
        """添加特征选择建议"""
        try:
            # 创建建议表格
            recommendations = []
            
            # 基于重要性的建议
            if feature_importance and 'combined_score' in feature_importance:
                scores = feature_importance['combined_score']
                low_importance = [k for k, v in scores.items() if v < 0.01]
                recommendations.extend([f"移除低重要性特征: {', '.join(low_importance[:5])}" if low_importance else "无需移除低重要性特征"])
            
            # 基于相关性的建议
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append(f"发现{len(high_corr_pairs)}对高相关特征，建议移除")
            else:
                recommendations.append("特征相关性良好，无需移除")
            
            # 基于缺失值的建议
            missing_features = df.isnull().sum()
            high_missing = missing_features[missing_features > len(df) * 0.3]
            if len(high_missing) > 0:
                recommendations.append(f"移除高缺失率特征: {', '.join(high_missing.index[:3])}")
            else:
                recommendations.append("缺失值控制良好")
            
            # 总体建议
            total_features = len(df.select_dtypes(include=[np.number]).columns)
            if total_features > 50:
                recommendations.append("特征数量过多，建议降维至30-40个")
            elif total_features < 20:
                recommendations.append("特征数量偏少，可考虑增加技术指标")
            else:
                recommendations.append("特征数量适中")
            
            # 简化显示为文本
            recommendation_text = "<br>".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])
            
            fig.add_annotation(
                text=recommendation_text,
                xref=f"x{col}" if row > 1 or col > 1 else "x",
                yref=f"y{((row-1)*3 + col)}" if row > 1 or col > 1 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12),
                align="left"
            )
            
        except Exception as e:
            self.logger.error(f"添加特征选择建议失败: {e}")
    
    def create_feature_importance_dashboard(self, feature_importance: Dict, save_path: str = None) -> str:
        """创建特征重要性看板"""
        try:
            if not feature_importance or 'combined_score' not in feature_importance:
                self.logger.warning("缺少特征重要性数据")
                return None
            
            # 获取数据
            combined_scores = feature_importance['combined_score']
            rf_scores = feature_importance.get('random_forest', {})
            mi_scores = feature_importance.get('mutual_info', {})
            correlations = feature_importance.get('correlations', {})
            
            # 创建看板
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    '综合重要性排序', '随机森林重要性',
                    '互信息重要性', '与目标的相关性'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # 综合重要性
            top_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            features, scores = zip(*top_combined)
            
            fig.add_trace(
                go.Bar(x=list(scores), y=list(features), orientation='h',
                      name='综合重要性', marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            # 随机森林重要性
            if rf_scores:
                top_rf = sorted(rf_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                rf_features, rf_vals = zip(*top_rf)
                
                fig.add_trace(
                    go.Bar(x=list(rf_vals), y=list(rf_features), orientation='h',
                          name='随机森林', marker_color=self.colors['success']),
                    row=1, col=2
                )
            
            # 互信息重要性
            if mi_scores:
                top_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                mi_features, mi_vals = zip(*top_mi)
                
                fig.add_trace(
                    go.Bar(x=list(mi_vals), y=list(mi_features), orientation='h',
                          name='互信息', marker_color=self.colors['info']),
                    row=2, col=1
                )
            
            # 相关性
            if correlations:
                corr_data = [(k, abs(v['correlation'])) for k, v in correlations.items() 
                           if 'correlation' in v]
                if corr_data:
                    top_corr = sorted(corr_data, key=lambda x: x[1], reverse=True)[:15]
                    corr_features, corr_vals = zip(*top_corr)
                    
                    fig.add_trace(
                        go.Bar(x=list(corr_vals), y=list(corr_features), orientation='h',
                              name='相关性', marker_color=self.colors['warning']),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title='特征重要性综合分析看板',
                height=800,
                showlegend=False
            )
            
            if save_path is None:
                save_path = "results/feature_importance_dashboard.html"
            
            fig.write_html(save_path)
            self.logger.info(f"✅ 特征重要性看板已保存: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.exception(f"❌ 创建特征重要性看板失败: {e}")
            return None
    
    def analyze_feature_quality(self, df: pd.DataFrame) -> Dict:
        """分析特征质量"""
        try:
            quality_report = {
                'total_features': 0,
                'high_quality_features': 0,
                'medium_quality_features': 0,
                'low_quality_features': 0,
                'problematic_features': [],
                'recommendations': []
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            quality_report['total_features'] = len(numeric_cols)
            
            for col in numeric_cols:
                feature_quality = self._assess_single_feature_quality(df[col], col)
                
                if feature_quality['overall_score'] >= 0.7:
                    quality_report['high_quality_features'] += 1
                elif feature_quality['overall_score'] >= 0.4:
                    quality_report['medium_quality_features'] += 1
                else:
                    quality_report['low_quality_features'] += 1
                    quality_report['problematic_features'].append({
                        'feature': col,
                        'issues': feature_quality['issues'],
                        'score': feature_quality['overall_score']
                    })
            
            # 生成建议
            if quality_report['low_quality_features'] > quality_report['total_features'] * 0.3:
                quality_report['recommendations'].append("低质量特征过多，建议重新设计特征工程")
            
            if len(quality_report['problematic_features']) > 0:
                quality_report['recommendations'].append(f"移除或修复{len(quality_report['problematic_features'])}个问题特征")
            
            return quality_report
            
        except Exception as e:
            self.logger.exception(f"❌ 特征质量分析失败: {e}")
            return {}
    
    def _assess_single_feature_quality(self, feature_series: pd.Series, feature_name: str) -> Dict:
        """评估单个特征的质量"""
        try:
            quality_assessment = {
                'feature_name': feature_name,
                'overall_score': 0.0,
                'issues': []
            }
            
            scores = []
            
            # 1. 缺失值检查
            missing_rate = feature_series.isnull().sum() / len(feature_series)
            if missing_rate < 0.05:
                scores.append(1.0)
            elif missing_rate < 0.2:
                scores.append(0.5)
                quality_assessment['issues'].append(f"中等缺失率: {missing_rate:.2%}")
            else:
                scores.append(0.0)
                quality_assessment['issues'].append(f"高缺失率: {missing_rate:.2%}")
            
            # 2. 方差检查
            if feature_series.var() == 0:
                scores.append(0.0)
                quality_assessment['issues'].append("零方差特征")
            elif feature_series.var() < 1e-8:
                scores.append(0.2)
                quality_assessment['issues'].append("极低方差")
            else:
                scores.append(1.0)
            
            # 3. 异常值检查
            Q1 = feature_series.quantile(0.25)
            Q3 = feature_series.quantile(0.75)
            IQR = Q3 - Q1
            outlier_rate = len(feature_series[(feature_series < Q1 - 1.5*IQR) | 
                                           (feature_series > Q3 + 1.5*IQR)]) / len(feature_series)
            
            if outlier_rate < 0.05:
                scores.append(1.0)
            elif outlier_rate < 0.1:
                scores.append(0.7)
            else:
                scores.append(0.3)
                quality_assessment['issues'].append(f"高异常值率: {outlier_rate:.2%}")
            
            # 4. 分布检查
            from scipy.stats import skew, kurtosis
            feature_skew = abs(skew(feature_series.dropna()))
            feature_kurt = abs(kurtosis(feature_series.dropna()))
            
            if feature_skew < 2 and feature_kurt < 7:
                scores.append(1.0)
            elif feature_skew < 5 and feature_kurt < 15:
                scores.append(0.6)
            else:
                scores.append(0.2)
                quality_assessment['issues'].append("极端分布偏度/峰度")
            
            # 计算综合评分
            quality_assessment['overall_score'] = np.mean(scores)
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"单特征质量评估失败: {e}")
            return {'feature_name': feature_name, 'overall_score': 0.0, 'issues': ['评估失败']}

def main():
    """测试特征分析器功能"""
    print("📊 测试特征分析器")
    
    analyzer = FeatureAnalyzer()
    print("✅ 特征分析器初始化完成")
    print("📋 主要功能:")
    print("  - 特征重要性可视化")
    print("  - 相关性热图分析")
    print("  - 特征质量评估")
    print("  - 特征选择建议")
    print("  - 综合分析报告")

if __name__ == "__main__":
    main() 