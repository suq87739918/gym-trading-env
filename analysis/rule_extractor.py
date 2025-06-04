"""
策略规则提取器
从训练好的强化学习模型中提取可解释的交易规则
支持决策树、规则集合、逻辑回归等方法
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz

from utils.config import get_config
from utils.logger import get_logger

class TradingRuleExtractor:
    """交易规则提取器"""
    
    def __init__(self, feature_names: List[str] = None):
        self.config = get_config()
        self.logger = get_logger('RuleExtractor', 'rule_extractor.log')
        
        self.feature_names = feature_names or []
        self.extracted_rules = {}
        self.surrogate_models = {}
        self.training_data = None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def collect_policy_data(self, model, states: np.ndarray, 
                          context_info: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集策略决策数据
        
        Args:
            model: 训练好的强化学习模型
            states: 状态数组
            context_info: 上下文信息
        
        Returns:
            (states, actions): 状态和对应的动作
        """
        try:
            self.logger.info(f"📊 收集策略数据，状态数量: {len(states)}")
            
            actions = []
            
            for i, state in enumerate(states):
                try:
                    # 使用模型预测动作
                    if hasattr(model, 'predict'):
                        action, _ = model.predict(state.reshape(1, -1), deterministic=True)
                        actions.append(action[0])
                    elif hasattr(model, 'forward'):
                        # PyTorch模型
                        import torch
                        with torch.no_grad():
                            if isinstance(state, np.ndarray):
                                state_tensor = torch.FloatTensor(state.reshape(1, -1))
                            else:
                                state_tensor = state
                            action_probs = model.forward(state_tensor)
                            action = torch.argmax(action_probs, dim=-1).item()
                            actions.append(action)
                    else:
                        # 直接调用
                        action = model(state.reshape(1, -1))
                        actions.append(int(action))
                        
                except Exception as e:
                    self.logger.warning(f"状态{i}预测失败: {e}")
                    actions.append(0)  # 默认动作
            
            actions = np.array(actions)
            
            # 存储训练数据
            self.training_data = {
                'states': states,
                'actions': actions,
                'context_info': context_info or [{}] * len(states)
            }
            
            self.logger.info(f"✅ 收集完成，获得{len(actions)}个决策样本")
            self.logger.info(f"📈 动作分布: {np.bincount(actions)}")
            
            return states, actions
            
        except Exception as e:
            self.logger.error(f"❌ 策略数据收集失败: {e}")
            return np.array([]), np.array([])
    
    def extract_decision_tree_rules(self, states: np.ndarray, actions: np.ndarray,
                                  max_depth: int = 10, min_samples_leaf: int = 20) -> Dict:
        """
        使用决策树提取规则
        
        Args:
            states: 状态特征
            actions: 对应动作
            max_depth: 最大深度
            min_samples_leaf: 叶子节点最小样本数
        
        Returns:
            规则提取结果
        """
        try:
            self.logger.info("🌲 使用决策树提取交易规则...")
            
            # 训练决策树
            dt_model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                class_weight='balanced'
            )
            
            dt_model.fit(states, actions)
            
            # 预测准确率
            predictions = dt_model.predict(states)
            accuracy = accuracy_score(actions, predictions)
            
            # 提取文本规则
            tree_rules = export_text(
                dt_model, 
                feature_names=self.feature_names[:states.shape[1]] if self.feature_names else None,
                class_names=[f'Action_{i}' for i in range(len(np.unique(actions)))]
            )
            
            # 特征重要性
            feature_importance = dict(zip(
                self.feature_names[:states.shape[1]] if self.feature_names else [f'feature_{i}' for i in range(states.shape[1])],
                dt_model.feature_importances_
            ))
            
            # 解析规则为可读格式
            readable_rules = self._parse_decision_tree_rules(dt_model, states, actions)
            
            result = {
                'model': dt_model,
                'accuracy': accuracy,
                'tree_rules_text': tree_rules,
                'feature_importance': feature_importance,
                'readable_rules': readable_rules,
                'classification_report': classification_report(actions, predictions, output_dict=True)
            }
            
            self.extracted_rules['decision_tree'] = result
            self.surrogate_models['decision_tree'] = dt_model
            
            self.logger.info(f"✅ 决策树规则提取完成，准确率: {accuracy:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 决策树规则提取失败: {e}")
            return {}
    
    def extract_random_forest_rules(self, states: np.ndarray, actions: np.ndarray,
                                   n_trees: int = 50, max_depth: int = 8) -> Dict:
        """
        使用随机森林提取规则
        
        Args:
            states: 状态特征
            actions: 对应动作
            n_trees: 树的数量
            max_depth: 最大深度
        
        Returns:
            规则提取结果
        """
        try:
            self.logger.info("🌳 使用随机森林提取交易规则...")
            
            # 训练随机森林
            rf_model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            )
            
            rf_model.fit(states, actions)
            
            # 预测准确率
            predictions = rf_model.predict(states)
            accuracy = accuracy_score(actions, predictions)
            
            # 特征重要性
            feature_importance = dict(zip(
                self.feature_names[:states.shape[1]] if self.feature_names else [f'feature_{i}' for i in range(states.shape[1])],
                rf_model.feature_importances_
            ))
            
            # 提取最重要的几棵树的规则
            important_trees_rules = []
            tree_importances = []
            
            for i, tree in enumerate(rf_model.estimators_[:5]):  # 只分析前5棵树
                tree_rules = export_text(
                    tree,
                    feature_names=self.feature_names[:states.shape[1]] if self.feature_names else None
                )
                
                # 计算树的重要性（基于准确率）
                tree_pred = tree.predict(states)
                tree_acc = accuracy_score(actions, tree_pred)
                
                important_trees_rules.append({
                    'tree_index': i,
                    'accuracy': tree_acc,
                    'rules': tree_rules
                })
                tree_importances.append(tree_acc)
            
            # 排序选择最好的树
            important_trees_rules.sort(key=lambda x: x['accuracy'], reverse=True)
            
            result = {
                'model': rf_model,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'important_trees': important_trees_rules[:3],  # 前3棵最好的树
                'classification_report': classification_report(actions, predictions, output_dict=True)
            }
            
            self.extracted_rules['random_forest'] = result
            self.surrogate_models['random_forest'] = rf_model
            
            self.logger.info(f"✅ 随机森林规则提取完成，准确率: {accuracy:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 随机森林规则提取失败: {e}")
            return {}
    
    def extract_logistic_regression_rules(self, states: np.ndarray, actions: np.ndarray) -> Dict:
        """
        使用逻辑回归提取线性规则
        
        Args:
            states: 状态特征
            actions: 对应动作
        
        Returns:
            规则提取结果
        """
        try:
            self.logger.info("📈 使用逻辑回归提取线性规则...")
            
            # 训练逻辑回归
            lr_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                multi_class='ovr'
            )
            
            lr_model.fit(states, actions)
            
            # 预测准确率
            predictions = lr_model.predict(states)
            accuracy = accuracy_score(actions, predictions)
            
            # 获取系数
            feature_names_used = self.feature_names[:states.shape[1]] if self.feature_names else [f'feature_{i}' for i in range(states.shape[1])]
            
            # 每个类别的线性规则
            linear_rules = {}
            
            if len(lr_model.classes_) == 2:
                # 二分类
                coefficients = lr_model.coef_[0]
                intercept = lr_model.intercept_[0]
                
                linear_rules[f'Action_{lr_model.classes_[1]}_vs_{lr_model.classes_[0]}'] = {
                    'intercept': intercept,
                    'coefficients': dict(zip(feature_names_used, coefficients)),
                    'equation': self._create_linear_equation(feature_names_used, coefficients, intercept)
                }
            else:
                # 多分类
                for i, class_label in enumerate(lr_model.classes_):
                    coefficients = lr_model.coef_[i]
                    intercept = lr_model.intercept_[i]
                    
                    linear_rules[f'Action_{class_label}'] = {
                        'intercept': intercept,
                        'coefficients': dict(zip(feature_names_used, coefficients)),
                        'equation': self._create_linear_equation(feature_names_used, coefficients, intercept)
                    }
            
            # 特征重要性（基于系数绝对值）
            if len(lr_model.classes_) == 2:
                feature_importance = dict(zip(feature_names_used, np.abs(lr_model.coef_[0])))
            else:
                # 多分类：计算所有类别系数的平均绝对值
                avg_abs_coef = np.mean(np.abs(lr_model.coef_), axis=0)
                feature_importance = dict(zip(feature_names_used, avg_abs_coef))
            
            result = {
                'model': lr_model,
                'accuracy': accuracy,
                'linear_rules': linear_rules,
                'feature_importance': feature_importance,
                'classification_report': classification_report(actions, predictions, output_dict=True)
            }
            
            self.extracted_rules['logistic_regression'] = result
            self.surrogate_models['logistic_regression'] = lr_model
            
            self.logger.info(f"✅ 逻辑回归规则提取完成，准确率: {accuracy:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 逻辑回归规则提取失败: {e}")
            return {}
    
    def _parse_decision_tree_rules(self, dt_model, states: np.ndarray, actions: np.ndarray) -> List[Dict]:
        """解析决策树规则为可读格式"""
        try:
            tree = dt_model.tree_
            feature_names_used = self.feature_names[:states.shape[1]] if self.feature_names else [f'feature_{i}' for i in range(states.shape[1])]
            
            def extract_rules_recursive(node_id, depth=0, parent_rule=""):
                rules = []
                
                # 如果是叶子节点
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    class_counts = tree.value[node_id][0]
                    predicted_class = np.argmax(class_counts)
                    confidence = class_counts[predicted_class] / np.sum(class_counts)
                    
                    rule = {
                        'conditions': parent_rule,
                        'action': f'Action_{predicted_class}',
                        'confidence': confidence,
                        'samples': int(np.sum(class_counts)),
                        'depth': depth
                    }
                    rules.append(rule)
                else:
                    # 内部节点
                    feature_idx = tree.feature[node_id]
                    threshold = tree.threshold[node_id]
                    feature_name = feature_names_used[feature_idx] if feature_idx < len(feature_names_used) else f'feature_{feature_idx}'
                    
                    # 左子树 (<=)
                    left_condition = f"{feature_name} <= {threshold:.3f}"
                    new_parent_rule = f"{parent_rule} AND {left_condition}" if parent_rule else left_condition
                    rules.extend(extract_rules_recursive(tree.children_left[node_id], depth+1, new_parent_rule))
                    
                    # 右子树 (>)
                    right_condition = f"{feature_name} > {threshold:.3f}"
                    new_parent_rule = f"{parent_rule} AND {right_condition}" if parent_rule else right_condition
                    rules.extend(extract_rules_recursive(tree.children_right[node_id], depth+1, new_parent_rule))
                
                return rules
            
            return extract_rules_recursive(0)
            
        except Exception as e:
            self.logger.error(f"解析决策树规则失败: {e}")
            return []
    
    def _create_linear_equation(self, feature_names: List[str], coefficients: np.ndarray, intercept: float) -> str:
        """创建线性方程字符串"""
        try:
            terms = []
            
            for name, coef in zip(feature_names, coefficients):
                if abs(coef) > 1e-6:  # 忽略极小的系数
                    sign = "+" if coef > 0 else "-"
                    if not terms:  # 第一项不需要符号
                        terms.append(f"{coef:.3f} * {name}")
                    else:
                        terms.append(f" {sign} {abs(coef):.3f} * {name}")
            
            if intercept != 0:
                sign = "+" if intercept > 0 else "-"
                terms.append(f" {sign} {abs(intercept):.3f}")
            
            equation = "".join(terms)
            return f"score = {equation}"
            
        except Exception as e:
            return f"方程生成错误: {e}"
    
    def create_rule_summary_report(self, save_path: str = None) -> str:
        """
        创建规则摘要报告
        """
        try:
            self.logger.info("📋 创建规则摘要报告...")
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('交易策略规则提取摘要报告', fontsize=16, fontweight='bold')
            
            # 1. 模型准确率对比
            if self.extracted_rules:
                model_names = list(self.extracted_rules.keys())
                accuracies = [result['accuracy'] for result in self.extracted_rules.values()]
                
                axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
                axes[0, 0].set_title('代理模型准确率对比')
                axes[0, 0].set_ylabel('准确率')
                axes[0, 0].set_ylim([0, 1])
                
                # 添加数值标签
                for i, acc in enumerate(accuracies):
                    axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
            
            # 2. 特征重要性对比（如果有决策树结果）
            if 'decision_tree' in self.extracted_rules:
                feature_imp = self.extracted_rules['decision_tree']['feature_importance']
                # 选择top10特征
                top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                features, importances = zip(*top_features)
                
                axes[0, 1].barh(features, importances, color='orange')
                axes[0, 1].set_title('Top10 特征重要性 (决策树)')
                axes[0, 1].set_xlabel('重要性')
            
            # 3. 动作分布
            if self.training_data:
                actions = self.training_data['actions']
                action_counts = np.bincount(actions)
                action_labels = [f'Action_{i}' for i in range(len(action_counts))]
                
                axes[1, 0].pie(action_counts, labels=action_labels, autopct='%1.1f%%')
                axes[1, 0].set_title('策略动作分布')
            
            # 4. 规则复杂度分析
            complexity_data = {}
            if 'decision_tree' in self.extracted_rules:
                dt_model = self.extracted_rules['decision_tree']['model']
                complexity_data['决策树深度'] = dt_model.get_depth()
                complexity_data['叶子节点数'] = dt_model.get_n_leaves()
            
            if 'random_forest' in self.extracted_rules:
                rf_model = self.extracted_rules['random_forest']['model']
                complexity_data['随机森林树数'] = rf_model.n_estimators
                complexity_data['平均树深度'] = np.mean([tree.get_depth() for tree in rf_model.estimators_])
            
            if complexity_data:
                metrics = list(complexity_data.keys())
                values = list(complexity_data.values())
                
                axes[1, 1].bar(metrics, values, color='purple', alpha=0.7)
                axes[1, 1].set_title('规则复杂度指标')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = "results/rule_extraction_summary.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成文本报告
            text_report = self._generate_text_report()
            text_report_path = save_path.replace('.png', '_text.txt')
            
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            self.logger.info(f"✅ 规则摘要报告已保存: {save_path}")
            self.logger.info(f"📄 文本报告已保存: {text_report_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 创建规则摘要报告失败: {e}")
            return None
    
    def _generate_text_report(self) -> str:
        """生成文本格式的详细报告"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("交易策略规则提取详细报告")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # 总体统计
            if self.training_data:
                total_samples = len(self.training_data['actions'])
                unique_actions = len(np.unique(self.training_data['actions']))
                report_lines.append(f"数据统计:")
                report_lines.append(f"  总样本数: {total_samples:,}")
                report_lines.append(f"  动作类别数: {unique_actions}")
                report_lines.append(f"  特征维度: {self.training_data['states'].shape[1]}")
                report_lines.append("")
            
            # 各模型结果
            for model_name, result in self.extracted_rules.items():
                report_lines.append(f"{model_name.upper()} 规则提取结果:")
                report_lines.append("-" * 40)
                report_lines.append(f"  准确率: {result['accuracy']:.4f}")
                
                if 'feature_importance' in result:
                    report_lines.append("  Top5 重要特征:")
                    top_features = sorted(result['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
                    for feature, importance in top_features:
                        report_lines.append(f"    {feature}: {importance:.4f}")
                
                if model_name == 'decision_tree' and 'readable_rules' in result:
                    report_lines.append("  主要决策规则（前5条）:")
                    rules = result['readable_rules'][:5]
                    for i, rule in enumerate(rules, 1):
                        confidence = rule.get('confidence', 0)
                        samples = rule.get('samples', 0)
                        conditions = rule.get('conditions', '')
                        action = rule.get('action', '')
                        
                        report_lines.append(f"    规则{i}: IF {conditions}")
                        report_lines.append(f"            THEN {action} (置信度: {confidence:.3f}, 样本: {samples})")
                
                if model_name == 'logistic_regression' and 'linear_rules' in result:
                    report_lines.append("  线性决策方程:")
                    for action, rule_info in result['linear_rules'].items():
                        equation = rule_info.get('equation', '')
                        report_lines.append(f"    {action}: {equation}")
                
                report_lines.append("")
            
            # 建议和总结
            report_lines.append("规则分析建议:")
            report_lines.append("-" * 40)
            
            if self.extracted_rules:
                # 找出最准确的模型
                best_model = max(self.extracted_rules.items(), key=lambda x: x[1]['accuracy'])
                report_lines.append(f"1. 最佳代理模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")
                
                # 通用重要特征
                all_features = {}
                for result in self.extracted_rules.values():
                    if 'feature_importance' in result:
                        for feature, importance in result['feature_importance'].items():
                            if feature not in all_features:
                                all_features[feature] = []
                            all_features[feature].append(importance)
                
                if all_features:
                    avg_importance = {f: np.mean(imps) for f, imps in all_features.items()}
                    top_common_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    report_lines.append("2. 最重要的通用特征:")
                    for feature, avg_imp in top_common_features:
                        report_lines.append(f"   - {feature}: 平均重要性 {avg_imp:.4f}")
                
                report_lines.append("3. 策略解释建议:")
                report_lines.append("   - 可以根据提取的规则验证策略逻辑的合理性")
                report_lines.append("   - 关注高重要性特征的阈值设定")
                report_lines.append("   - 考虑将复杂规则简化为人工可执行的策略")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"报告生成错误: {e}"
    
    def get_trading_strategy_summary(self) -> Dict:
        """获取交易策略摘要"""
        try:
            if not self.extracted_rules:
                return {"error": "没有提取的规则可分析"}
            
            summary = {
                'models_trained': list(self.extracted_rules.keys()),
                'best_model': None,
                'model_accuracies': {},
                'common_important_features': [],
                'strategy_insights': []
            }
            
            # 模型准确率
            for model_name, result in self.extracted_rules.items():
                summary['model_accuracies'][model_name] = result['accuracy']
            
            # 最佳模型
            if summary['model_accuracies']:
                best_model_name = max(summary['model_accuracies'].items(), key=lambda x: x[1])[0]
                summary['best_model'] = {
                    'name': best_model_name,
                    'accuracy': summary['model_accuracies'][best_model_name],
                    'details': self.extracted_rules[best_model_name]
                }
            
            # 通用重要特征
            all_features = {}
            for result in self.extracted_rules.values():
                if 'feature_importance' in result:
                    for feature, importance in result['feature_importance'].items():
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append(importance)
            
            if all_features:
                avg_importance = {f: np.mean(imps) for f, imps in all_features.items()}
                summary['common_important_features'] = sorted(
                    avg_importance.items(), key=lambda x: x[1], reverse=True
                )[:10]
            
            # 策略洞察
            if 'decision_tree' in self.extracted_rules:
                dt_rules = self.extracted_rules['decision_tree'].get('readable_rules', [])
                high_confidence_rules = [r for r in dt_rules if r.get('confidence', 0) > 0.8]
                summary['strategy_insights'].append(f"发现{len(high_confidence_rules)}条高置信度规则")
            
            if self.training_data:
                action_dist = np.bincount(self.training_data['actions'])
                dominant_action = np.argmax(action_dist)
                dominant_ratio = action_dist[dominant_action] / len(self.training_data['actions'])
                summary['strategy_insights'].append(f"策略倾向于Action_{dominant_action} (占比{dominant_ratio:.1%})")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取策略摘要失败: {e}")
            return {"error": str(e)}

def main():
    """测试规则提取器功能"""
    print("📏 测试交易规则提取器")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # 模拟状态特征
    states = np.random.randn(n_samples, n_features)
    
    # 模拟策略决策（简单规则）
    actions = []
    for state in states:
        if state[0] > 0.5 and state[1] < -0.3:  # 买入条件
            action = 1
        elif state[0] < -0.5 and state[1] > 0.3:  # 卖出条件
            action = 2
        elif abs(state[0]) < 0.2:  # 平仓条件
            action = 3
        else:  # 持有
            action = 0
        actions.append(action)
    
    actions = np.array(actions)
    
    # 特征名称
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # 创建提取器
    extractor = TradingRuleExtractor(feature_names=feature_names)
    
    # 提取规则
    print("🌲 提取决策树规则...")
    dt_result = extractor.extract_decision_tree_rules(states, actions, max_depth=5)
    print(f"决策树准确率: {dt_result.get('accuracy', 0):.3f}")
    
    print("🌳 提取随机森林规则...")
    rf_result = extractor.extract_random_forest_rules(states, actions, n_trees=20)
    print(f"随机森林准确率: {rf_result.get('accuracy', 0):.3f}")
    
    print("📈 提取逻辑回归规则...")
    lr_result = extractor.extract_logistic_regression_rules(states, actions)
    print(f"逻辑回归准确率: {lr_result.get('accuracy', 0):.3f}")
    
    # 生成报告
    print("📋 生成规则摘要报告...")
    report_path = extractor.create_rule_summary_report()
    if report_path:
        print(f"报告已保存: {report_path}")
    
    # 获取策略摘要
    summary = extractor.get_trading_strategy_summary()
    print("📊 策略摘要:")
    for key, value in summary.items():
        if key != 'details':
            print(f"  {key}: {value}")
    
    print("✅ 规则提取器测试完成！")

if __name__ == "__main__":
    main() 