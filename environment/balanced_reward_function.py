"""
平衡奖励函数模块 - 实现高胜率与高年化收益的多目标优化
基于数学公式: r_t = R_t + α·1{R_t>0} - β·1{R_t<0} - γ·DD_t

核心设计原理：
1. R_t: 当步相对收益率（基础收益项）
2. α·1{R_t>0}: 胜率奖励项（盈利时的固定奖励）
3. β·1{R_t<0}: 亏损惩罚项（亏损时的固定惩罚）
4. γ·DD_t: 回撤软约束项（当前回撤的惩罚）

多目标平衡：
- 年化收益最大化：通过R_t项直接激励收益
- 胜率提升：通过α, β参数调节胜负奖励强度
- 风险控制：通过γ参数控制回撤容忍度
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class RewardObjective(Enum):
    """奖励目标类型"""
    BALANCED = "balanced"           # 平衡收益、胜率、风险
    HIGH_WINRATE = "high_winrate"   # 高胜率导向
    HIGH_RETURN = "high_return"     # 高收益导向
    LOW_RISK = "low_risk"          # 低风险导向
    ADAPTIVE = "adaptive"          # 自适应调整

@dataclass
class BalancedRewardConfig:
    """平衡奖励函数配置参数"""
    
    # ================= 核心数学公式参数 =================
    # r_t = R_t + α·1{R_t>0} - β·1{R_t<0} - γ·DD_t
    
    # 胜率奖励参数 (α)
    alpha_small_win: float = 0.5    # 小盈利奖励 (0-2%)
    alpha_medium_win: float = 1.0   # 中盈利奖励 (2-5%)
    alpha_large_win: float = 2.0    # 大盈利奖励 (>5%)
    
    # 亏损惩罚参数 (β) - 通常 β > α 以提升胜率倾向
    beta_small_loss: float = 0.8    # 小亏损惩罚 (0-2%)
    beta_medium_loss: float = 1.8   # 中亏损惩罚 (2-5%)
    beta_large_loss: float = 3.5    # 大亏损惩罚 (>5%)
    
    # 回撤软约束参数 (γ)
    gamma_drawdown: float = 2.0     # 回撤惩罚系数
    
    # ================= 收益计算参数 =================
    return_scale_factor: float = 100.0    # R_t 缩放因子
    use_log_returns: bool = False          # 是否使用对数收益
    min_return_threshold: float = 0.001    # 最小收益阈值（避免噪音）
    
    # ================= 胜率分级阈值 =================
    small_profit_threshold: float = 0.02   # 2%
    medium_profit_threshold: float = 0.05  # 5%
    small_loss_threshold: float = 0.02     # 2%
    medium_loss_threshold: float = 0.05    # 5%
    
    # ================= 回撤计算参数 =================
    drawdown_window: int = 100             # 回撤计算窗口
    max_drawdown_penalty: float = 5.0      # 最大回撤惩罚
    drawdown_recovery_bonus: float = 0.5   # 回撤恢复奖励
    
    # ================= 年化收益增强参数 =================
    annual_return_bonus_scale: float = 1.5  # 年化收益奖励倍数
    sharpe_ratio_weight: float = 0.3        # 夏普比率权重
    profit_factor_weight: float = 0.2       # 盈亏比权重
    
    # ================= 连胜/连败调整 =================
    consecutive_win_bonus: float = 0.1      # 连胜奖励递增
    max_consecutive_bonus: float = 1.0      # 连胜奖励上限
    consecutive_loss_penalty: float = 0.2   # 连败惩罚递增
    max_consecutive_penalty: float = 2.0    # 连败惩罚上限
    
    # ================= 自适应调整参数 =================
    enable_adaptive: bool = True            # 启用自适应调整
    performance_window: int = 50            # 表现评估窗口
    alpha_adjustment_rate: float = 0.1      # α参数调整速率
    beta_adjustment_rate: float = 0.1       # β参数调整速率
    gamma_adjustment_rate: float = 0.05     # γ参数调整速率
    
    # ================= 目标权重配置 =================
    winrate_target: float = 0.65            # 目标胜率
    annual_return_target: float = 0.30      # 目标年化收益率
    max_drawdown_target: float = 0.15       # 目标最大回撤

class BalancedRewardFunction:
    """平衡奖励函数计算器"""
    
    def __init__(self, config: BalancedRewardConfig = None, logger: logging.Logger = None):
        self.config = config or BalancedRewardConfig()
        self.logger = logger or self._setup_logger()
        
        # 历史数据存储
        self.portfolio_history: List[float] = []
        self.return_history: List[float] = []
        self.reward_history: List[float] = []
        self.peak_portfolio_value: float = 0.0
        
        # 交易统计
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        
        # 动态参数（自适应调整）
        self.current_alpha: float = self.config.alpha_medium_win
        self.current_beta: float = self.config.beta_medium_loss
        self.current_gamma: float = self.config.gamma_drawdown
        
        self.logger.info(f"🎯 平衡奖励函数初始化: α={self.current_alpha:.2f}, "
                        f"β={self.current_beta:.2f}, γ={self.current_gamma:.2f}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def calculate_reward(self, current_portfolio_value: float, previous_portfolio_value: float,
                        action: int, trade_completed: bool = False, 
                        trade_pnl_pct: float = None) -> Tuple[float, Dict]:
        """
        ✅ 核心奖励计算函数
        实现数学公式: r_t = R_t + α·1{R_t>0} - β·1{R_t<0} - γ·DD_t
        
        Args:
            current_portfolio_value: 当前组合价值
            previous_portfolio_value: 前一步组合价值
            action: 执行的动作 (0=持仓, 1=开多, 2=开空, 3=平仓)
            trade_completed: 是否完成了一笔交易
            trade_pnl_pct: 交易盈亏百分比（仅在交易完成时提供）
            
        Returns:
            (总奖励, 奖励分解字典)
        """
        
        # 更新历史数据
        self.portfolio_history.append(current_portfolio_value)
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)
        
        # 奖励分解字典
        reward_breakdown = {
            'R_t': 0.0,              # 基础收益项
            'alpha_term': 0.0,       # 胜率奖励项 α·1{R_t>0}
            'beta_term': 0.0,        # 亏损惩罚项 β·1{R_t<0}
            'gamma_term': 0.0,       # 回撤软约束项 γ·DD_t
            'annual_return_bonus': 0.0,  # 年化收益奖励
            'consecutive_bonus': 0.0,    # 连胜/连败调整
            'total_reward': 0.0      # 总奖励
        }
        
        # ==================== 1. 计算基础收益项 R_t ====================
        if previous_portfolio_value > 0:
            if self.config.use_log_returns:
                # 对数收益：r_t = ln(V_t / V_{t-1})
                R_t = np.log(current_portfolio_value / previous_portfolio_value)
            else:
                # 相对收益：R_t = (V_t - V_{t-1}) / V_{t-1}
                R_t = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
            
            # 应用缩放因子和阈值过滤
            if abs(R_t) >= self.config.min_return_threshold:
                R_t_scaled = R_t * self.config.return_scale_factor
            else:
                R_t_scaled = 0.0  # 过滤小幅波动噪音
        else:
            R_t = 0.0
            R_t_scaled = 0.0
        
        reward_breakdown['R_t'] = R_t_scaled
        self.return_history.append(R_t)
        
        # ==================== 2. 计算胜率奖励/惩罚项 ====================
        alpha_term = 0.0
        beta_term = 0.0
        
        if trade_completed and trade_pnl_pct is not None:
            # 交易完成时，基于实际交易结果计算奖励/惩罚
            self.total_trades += 1
            
            if trade_pnl_pct > 0:  # 盈利交易
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                
                # 根据盈利幅度选择α值
                if trade_pnl_pct > self.config.medium_profit_threshold:  # >5%
                    alpha_term = self.config.alpha_large_win
                elif trade_pnl_pct > self.config.small_profit_threshold:  # 2%-5%
                    alpha_term = self.config.alpha_medium_win
                else:  # 0%-2%
                    alpha_term = self.config.alpha_small_win
                
                self.logger.debug(f"💰 盈利交易: {trade_pnl_pct*100:.2f}%, α奖励={alpha_term:.2f}")
                
            else:  # 亏损交易
                self.losing_trades += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
                # 根据亏损幅度选择β值
                abs_loss = abs(trade_pnl_pct)
                if abs_loss > self.config.medium_loss_threshold:  # >5%
                    beta_term = -self.config.beta_large_loss
                elif abs_loss > self.config.small_loss_threshold:  # 2%-5%
                    beta_term = -self.config.beta_medium_loss
                else:  # 0%-2%
                    beta_term = -self.config.beta_small_loss
                
                self.logger.debug(f"📉 亏损交易: {trade_pnl_pct*100:.2f}%, β惩罚={beta_term:.2f}")
        
        elif R_t != 0 and not trade_completed:
            # 非交易完成时，基于单步收益计算（较小的奖励/惩罚）
            if R_t > 0:
                alpha_term = self.current_alpha * 0.1  # 降低非交易时的奖励
            else:
                beta_term = -self.current_beta * 0.1   # 降低非交易时的惩罚
        
        reward_breakdown['alpha_term'] = alpha_term
        reward_breakdown['beta_term'] = beta_term
        
        # ==================== 3. 计算回撤软约束项 γ·DD_t ====================
        current_drawdown = self._calculate_current_drawdown()
        gamma_term = -self.current_gamma * current_drawdown
        
        # 限制回撤惩罚的最大值
        gamma_term = max(gamma_term, -self.config.max_drawdown_penalty)
        
        reward_breakdown['gamma_term'] = gamma_term
        
        # ==================== 4. 计算连胜/连败调整 ====================
        consecutive_bonus = 0.0
        
        if self.consecutive_wins >= 3:
            # 连胜奖励：递增但有上限
            consecutive_bonus = min(
                self.consecutive_wins * self.config.consecutive_win_bonus,
                self.config.max_consecutive_bonus
            )
        elif self.consecutive_losses >= 3:
            # 连败惩罚：递增但有上限
            consecutive_bonus = -min(
                self.consecutive_losses * self.config.consecutive_loss_penalty,
                self.config.max_consecutive_penalty
            )
        
        reward_breakdown['consecutive_bonus'] = consecutive_bonus
        
        # ==================== 5. 计算年化收益奖励 ====================
        annual_return_bonus = 0.0
        if len(self.portfolio_history) >= 20:  # 足够的历史数据
            annual_return_bonus = self._calculate_annual_return_bonus()
        
        reward_breakdown['annual_return_bonus'] = annual_return_bonus
        
        # ==================== 6. 组合最终奖励 ====================
        total_reward = (R_t_scaled + alpha_term + beta_term + gamma_term + 
                       consecutive_bonus + annual_return_bonus)
        
        # 限制奖励范围，避免极端值
        total_reward = np.clip(total_reward, -20.0, 20.0)
        reward_breakdown['total_reward'] = total_reward
        
        # 更新奖励历史
        self.reward_history.append(total_reward)
        
        # ==================== 7. 自适应参数调整 ====================
        if self.config.enable_adaptive and self.total_trades % 10 == 0:
            self._adaptive_parameter_adjustment()
        
        # ==================== 8. 详细日志记录 ====================
        if trade_completed or abs(total_reward) > 1.0:
            self.logger.info(f"🎁 奖励计算: R_t={R_t_scaled:.3f}, α={alpha_term:.3f}, "
                           f"β={beta_term:.3f}, γ={gamma_term:.3f}, "
                           f"连胜={consecutive_bonus:.3f}, 年化={annual_return_bonus:.3f}, "
                           f"总奖励={total_reward:.3f}")
        
        return total_reward, reward_breakdown
    
    def _calculate_current_drawdown(self) -> float:
        """
        计算当前回撤 DD_t
        DD_t = (max(V_0...V_t) - V_t) / max(V_0...V_t)
        """
        if len(self.portfolio_history) == 0 or self.peak_portfolio_value <= 0:
            return 0.0
        
        current_value = self.portfolio_history[-1]
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        return max(0.0, drawdown)  # 确保非负
    
    def _calculate_annual_return_bonus(self) -> float:
        """计算年化收益奖励"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            # 计算最近一段时间的收益率
            recent_window = min(252, len(self.portfolio_history))  # 最多取一年数据
            start_value = self.portfolio_history[-recent_window]
            end_value = self.portfolio_history[-1]
            
            if start_value <= 0:
                return 0.0
            
            # 计算年化收益率
            total_return = (end_value - start_value) / start_value
            periods_per_year = 252 * 24 * 4  # 假设15分钟级别数据
            annual_return = total_return * (periods_per_year / recent_window)
            
            # 计算相对于目标的奖励
            target_return = self.config.annual_return_target
            if annual_return > target_return:
                bonus = (annual_return - target_return) * self.config.annual_return_bonus_scale
                return min(bonus, 2.0)  # 限制最大奖励
            else:
                # 低于目标时轻微惩罚
                penalty = (annual_return - target_return) * 0.5
                return max(penalty, -1.0)  # 限制最大惩罚
        
        except Exception as e:
            self.logger.error(f"年化收益奖励计算失败: {e}")
            return 0.0
    
    def _adaptive_parameter_adjustment(self):
        """自适应参数调整"""
        try:
            if self.total_trades < 20:  # 需要足够的交易样本
                return
            
            # 计算当前表现指标
            current_winrate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
            current_drawdown = self._calculate_current_drawdown()
            
            # 根据表现调整α (胜率奖励)
            winrate_gap = current_winrate - self.config.winrate_target
            if winrate_gap < -0.1:  # 胜率太低，增加胜率奖励
                self.current_alpha = min(
                    self.current_alpha * (1 + self.config.alpha_adjustment_rate),
                    self.config.alpha_large_win * 1.5
                )
                self.logger.info(f"📈 胜率过低({current_winrate:.2%})，增加α奖励: {self.current_alpha:.2f}")
            elif winrate_gap > 0.1:  # 胜率太高，可以降低奖励
                self.current_alpha = max(
                    self.current_alpha * (1 - self.config.alpha_adjustment_rate),
                    self.config.alpha_small_win * 0.5
                )
            
            # 根据表现调整β (亏损惩罚)
            if current_winrate < self.config.winrate_target:
                # 胜率低时，增加亏损惩罚
                self.current_beta = min(
                    self.current_beta * (1 + self.config.beta_adjustment_rate),
                    self.config.beta_large_loss * 1.5
                )
            else:
                # 胜率高时，可以减少亏损惩罚
                self.current_beta = max(
                    self.current_beta * (1 - self.config.beta_adjustment_rate),
                    self.config.beta_small_loss * 0.5
                )
            
            # 根据回撤调整γ (回撤惩罚)
            if current_drawdown > self.config.max_drawdown_target:
                # 回撤过大，增加回撤惩罚
                self.current_gamma = min(
                    self.current_gamma * (1 + self.config.gamma_adjustment_rate),
                    self.config.gamma_drawdown * 2.0
                )
                self.logger.info(f"📉 回撤过大({current_drawdown:.2%})，增加γ惩罚: {self.current_gamma:.2f}")
            elif current_drawdown < self.config.max_drawdown_target * 0.5:
                # 回撤很小，可以适度降低回撤惩罚
                self.current_gamma = max(
                    self.current_gamma * (1 - self.config.gamma_adjustment_rate),
                    self.config.gamma_drawdown * 0.5
                )
        
        except Exception as e:
            self.logger.error(f"自适应参数调整失败: {e}")
    
    def get_performance_summary(self) -> Dict:
        """获取表现摘要"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'current_drawdown': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # 计算表现指标
        win_rate = self.winning_trades / self.total_trades
        current_drawdown = self._calculate_current_drawdown()
        
        # 计算年化收益
        if len(self.portfolio_history) >= 2:
            total_return = (self.portfolio_history[-1] - self.portfolio_history[0]) / self.portfolio_history[0]
            annual_return = total_return * (252 * 24 * 4 / len(self.portfolio_history))
        else:
            annual_return = 0.0
        
        # 计算夏普比率
        if len(self.return_history) > 1:
            returns = np.array(self.return_history)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 4)
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'current_drawdown': current_drawdown,
            'max_drawdown': max(self._calculate_historical_max_drawdown(), current_drawdown),
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'current_parameters': {
                'alpha': self.current_alpha,
                'beta': self.current_beta,
                'gamma': self.current_gamma
            }
        }
    
    def _calculate_historical_max_drawdown(self) -> float:
        """计算历史最大回撤"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        values = np.array(self.portfolio_history)
        cumulative_max = np.maximum.accumulate(values)
        drawdowns = (cumulative_max - values) / cumulative_max
        
        return np.max(drawdowns)
    
    def reset_for_new_episode(self):
        """重置为新的训练回合"""
        self.portfolio_history.clear()
        self.return_history.clear()
        self.reward_history.clear()
        self.peak_portfolio_value = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # 重置动态参数
        self.current_alpha = self.config.alpha_medium_win
        self.current_beta = self.config.beta_medium_loss
        self.current_gamma = self.config.gamma_drawdown
        
        self.logger.debug("🔄 奖励函数重置完成")

def create_reward_config(objective: RewardObjective = RewardObjective.BALANCED) -> BalancedRewardConfig:
    """
    根据目标类型创建奖励配置
    
    Args:
        objective: 奖励目标类型
        
    Returns:
        配置好的奖励函数配置
    """
    base_config = BalancedRewardConfig()
    
    if objective == RewardObjective.HIGH_WINRATE:
        # 高胜率导向：增强胜率奖励，加重亏损惩罚
        base_config.alpha_small_win = 0.8
        base_config.alpha_medium_win = 1.5
        base_config.alpha_large_win = 2.5
        base_config.beta_small_loss = 1.2
        base_config.beta_medium_loss = 2.5
        base_config.beta_large_loss = 4.0
        base_config.winrate_target = 0.70
        
    elif objective == RewardObjective.HIGH_RETURN:
        # 高收益导向：增强收益奖励，降低亏损惩罚
        base_config.return_scale_factor = 150.0
        base_config.alpha_large_win = 3.0
        base_config.beta_small_loss = 0.5
        base_config.beta_medium_loss = 1.2
        base_config.beta_large_loss = 2.5
        base_config.annual_return_target = 0.50
        base_config.annual_return_bonus_scale = 2.0
        
    elif objective == RewardObjective.LOW_RISK:
        # 低风险导向：强化回撤控制
        base_config.gamma_drawdown = 3.0
        base_config.max_drawdown_target = 0.10
        base_config.max_drawdown_penalty = 8.0
        base_config.return_scale_factor = 80.0
        base_config.alpha_large_win = 1.5
        
    elif objective == RewardObjective.ADAPTIVE:
        # 自适应：启用所有自适应功能
        base_config.enable_adaptive = True
        base_config.alpha_adjustment_rate = 0.15
        base_config.beta_adjustment_rate = 0.15
        base_config.gamma_adjustment_rate = 0.10
        
    return base_config

def main():
    """测试平衡奖励函数"""
    # 创建不同目标的配置
    configs = {
        'balanced': create_reward_config(RewardObjective.BALANCED),
        'high_winrate': create_reward_config(RewardObjective.HIGH_WINRATE),
        'high_return': create_reward_config(RewardObjective.HIGH_RETURN),
        'low_risk': create_reward_config(RewardObjective.LOW_RISK),
    }
    
    print("="*80)
    print("平衡奖励函数配置对比")
    print("="*80)
    
    for name, config in configs.items():
        print(f"\n{name.upper()} 配置:")
        print(f"  α (胜率奖励): 小={config.alpha_small_win}, 中={config.alpha_medium_win}, 大={config.alpha_large_win}")
        print(f"  β (亏损惩罚): 小={config.beta_small_loss}, 中={config.beta_medium_loss}, 大={config.beta_large_loss}")
        print(f"  γ (回撤惩罚): {config.gamma_drawdown}")
        print(f"  收益缩放: {config.return_scale_factor}")
        print(f"  目标胜率: {config.winrate_target:.1%}")
        print(f"  目标年化: {config.annual_return_target:.1%}")
        print(f"  目标回撤: {config.max_drawdown_target:.1%}")
    
    # 模拟测试
    print("\n" + "="*80)
    print("模拟测试")
    print("="*80)
    
    reward_func = BalancedRewardFunction(configs['balanced'])
    
    # 模拟一系列交易
    portfolio_values = [10000, 10200, 10150, 10300, 10250, 10400, 10350, 10500]
    trade_results = [None, 0.02, -0.005, 0.015, -0.01, 0.018, -0.008, 0.025]
    
    total_rewards = []
    
    for i in range(1, len(portfolio_values)):
        trade_completed = trade_results[i] is not None
        reward, breakdown = reward_func.calculate_reward(
            current_portfolio_value=portfolio_values[i],
            previous_portfolio_value=portfolio_values[i-1],
            action=3 if trade_completed else 0,
            trade_completed=trade_completed,
            trade_pnl_pct=trade_results[i]
        )
        total_rewards.append(reward)
        
        if trade_completed:
            print(f"交易 {i}: PnL={trade_results[i]*100:.1f}%, 奖励={reward:.3f}")
            print(f"  分解: R_t={breakdown['R_t']:.3f}, α={breakdown['alpha_term']:.3f}, "
                  f"β={breakdown['beta_term']:.3f}, γ={breakdown['gamma_term']:.3f}")
    
    # 显示最终统计
    summary = reward_func.get_performance_summary()
    print(f"\n最终统计:")
    print(f"  总交易: {summary['total_trades']}")
    print(f"  胜率: {summary['win_rate']:.1%}")
    print(f"  当前回撤: {summary['current_drawdown']:.1%}")
    print(f"  总奖励: {sum(total_rewards):.3f}")

if __name__ == "__main__":
    main() 