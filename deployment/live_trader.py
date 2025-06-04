"""
实时交易部署模块
支持Binance WebSocket实时行情监听和智能下单
"""
import asyncio
import websockets
import json
import threading
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import ccxt
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from utils.config import get_config, RISK_CONTROL_CONFIG
from utils.logger import get_trading_logger
from data.technical_indicators import TechnicalIndicators
from data.smc_signals import SMCSignals
from environment.trading_env import SolUsdtTradingEnv

class LiveTrader:
    """实时交易执行器"""
    
    def __init__(self, model_path: str, api_key: str = None, api_secret: str = None, 
                 testnet: bool = True, dry_run: bool = True):
        """
        初始化实时交易器
        
        Args:
            model_path: 训练好的PPO模型路径
            api_key: Binance API密钥
            api_secret: Binance API密钥
            testnet: 是否使用测试网络
            dry_run: 是否为模拟模式（不执行真实交易）
        """
        self.config = get_config()
        self.logger = get_trading_logger()
        
        # 交易模式设置
        self.dry_run = dry_run
        self.testnet = testnet
        
        # ✅ 导入风控配置
        self.risk_config = RISK_CONTROL_CONFIG
        
        # 加载PPO模型
        try:
            self.model = PPO.load(model_path)
            self.logger.info(f"✅ 成功加载PPO模型: {model_path}")
        except Exception as e:
            self.logger.error(f"❌ 加载PPO模型失败: {e}")
            raise
        
        # 初始化技术分析器
        self.tech_indicators = TechnicalIndicators()
        self.smc_signals = SMCSignals()
        
        # 初始化交易所连接
        self._init_exchange(api_key, api_secret)
        
        # 实时数据缓冲区
        self.kline_buffer = []
        self.max_buffer_size = 200  # 保留最近200根K线
        
        # 交易状态
        self.current_position = {
            'symbol': 'SOLUSDT',
            'side': None,  # 'long', 'short', None
            'size': 0.0,
            'entry_price': 0.0,
            'leverage': 1.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'entry_time': None
        }
        
        # ✅ 增强风控状态
        self.risk_control_state = {
            'stop_loss_price': 0.0,
            'take_profit_price': 0.0,
            'trailing_stop_price': 0.0,
            'trailing_stop_active': False,
            'highest_profit': 0.0,
            'position_entry_time': 0,
            'consecutive_losses': 0,
            'daily_pnl': 0.0,
            'stop_loss_triggered_count': 0,
            'take_profit_triggered_count': 0,
            'trailing_stop_triggered_count': 0,
            'time_stop_triggered_count': 0
        }
        
        # 风险控制参数
        self.max_daily_trades = self.config.get('MAX_DAILY_TRADES', 20)
        self.max_daily_loss = self.risk_config['DAILY_LOSS_LIMIT']
        self.min_trade_interval = self.config.get('MIN_TRADE_INTERVAL', 300)  # 5分钟
        
        # ✅ 风控参数
        self.base_stop_loss = self.risk_config['STOP_LOSS']
        self.base_take_profit = self.risk_config['TAKE_PROFIT']
        self.enable_trailing_stop = self.risk_config['ENABLE_TRAILING_STOP']
        self.trailing_stop_distance = self.risk_config['TRAILING_STOP_DISTANCE']
        self.trailing_stop_activation = self.risk_config['TRAILING_STOP_ACTIVATION']
        self.max_single_loss = self.risk_config['MAX_SINGLE_LOSS']
        
        # 统计数据
        self.daily_stats = {
            'trades_count': 0,
            'total_pnl': 0.0,
            'start_balance': 0.0,
            'current_balance': 0.0,
            'last_trade_time': 0
        }
        
        # WebSocket连接状态
        self.ws_connected = False
        self.ws_task = None
        
        self.logger.info(f"🚀 LiveTrader初始化完成 - 模式: {'模拟' if dry_run else '实盘'}")
        self.logger.info(f"🛡️ 风控配置: 止损={self.base_stop_loss*100:.1f}%, 止盈={self.base_take_profit*100:.1f}%")
    
    def _init_exchange(self, api_key: str, api_secret: str):
        """初始化交易所连接"""
        try:
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'timeout': 30000,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # 使用合约交易
                }
            }
            
            if self.testnet:
                exchange_config['sandbox'] = True
                exchange_config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com',
                        'private': 'https://testnet.binancefuture.com'
                    }
                }
            
            self.exchange = ccxt.binance(exchange_config)
            
            # 测试连接
            if not self.dry_run:
                balance = self.exchange.fetch_balance()
                self.daily_stats['start_balance'] = balance['USDT']['total']
                self.daily_stats['current_balance'] = balance['USDT']['total']
                self.logger.info(f"✅ 交易所连接成功, 余额: {balance['USDT']['total']:.2f} USDT")
            else:
                self.daily_stats['start_balance'] = 10000.0  # 模拟余额
                self.daily_stats['current_balance'] = 10000.0
                self.logger.info("📊 模拟模式 - 初始余额: 10000 USDT")
                
        except Exception as e:
            self.logger.error(f"❌ 交易所连接失败: {e}")
            if not self.dry_run:
                raise
    
    async def start_live_trading(self):
        """启动实时交易"""
        self.logger.info("🎯 启动实时交易系统...")
        
        try:
            # 启动WebSocket数据流
            self.ws_task = asyncio.create_task(self._websocket_handler())
            
            # 启动交易逻辑循环
            await self._trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ 实时交易系统启动失败: {e}")
            await self.stop_live_trading()
    
    async def stop_live_trading(self):
        """停止实时交易"""
        self.logger.info("🛑 停止实时交易系统...")
        
        if self.ws_task:
            self.ws_task.cancel()
        
        # 平仓所有持仓
        if self.current_position['side'] is not None and not self.dry_run:
            await self._close_position("系统停止，强制平仓")
        
        self.logger.info("✅ 实时交易系统已停止")
    
    async def _websocket_handler(self):
        """WebSocket数据处理"""
        ws_url = "wss://fstream.binance.com/ws/solusdt@kline_15m"  # 15分钟K线
        
        while True:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.ws_connected = True
                    self.logger.info("📡 WebSocket连接成功")
                    
                    async for message in websocket:
                        data = json.loads(message)
                        await self._process_kline_data(data)
                        
            except Exception as e:
                self.logger.error(f"❌ WebSocket连接错误: {e}")
                self.ws_connected = False
                await asyncio.sleep(5)  # 等待5秒后重连
    
    async def _process_kline_data(self, data: Dict):
        """处理K线数据"""
        try:
            kline = data.get('k', {})
            if not kline.get('x'):  # 只处理完成的K线
                return
            
            # 提取K线数据
            kline_data = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            
            # 添加到缓冲区
            self.kline_buffer.append(kline_data)
            
            # 保持缓冲区大小
            if len(self.kline_buffer) > self.max_buffer_size:
                self.kline_buffer.pop(0)
            
            # 当有足够数据时，执行交易逻辑
            if len(self.kline_buffer) >= 50:  # 至少需要50根K线进行技术分析
                await self._execute_trading_logic()
                
        except Exception as e:
            self.logger.error(f"❌ K线数据处理错误: {e}")
    
    async def _execute_trading_logic(self):
        """执行交易逻辑"""
        try:
            # 将缓冲区数据转换为DataFrame
            df = pd.DataFrame(self.kline_buffer)
            df.set_index('timestamp', inplace=True)
            
            # 计算技术指标
            df = self.tech_indicators.calculate_all_indicators(df)
            df = self.smc_signals.calculate_all_signals(df)
            
            # 准备环境观测数据
            observation = self._prepare_observation(df)
            
            # 使用PPO模型预测动作
            action, _states = self.model.predict(observation, deterministic=True)
            
            # 执行动作
            await self._execute_action(action, df.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"❌ 交易逻辑执行错误: {e}")
    
    def _prepare_observation(self, df: pd.DataFrame) -> np.ndarray:
        """准备环境观测数据"""
        try:
            # 使用最近的数据准备观测
            lookback = 20
            recent_data = df.tail(lookback)
            
            # 技术指标特征
            features = []
            
            # 价格特征 (归一化)
            price_features = ['close', 'high', 'low', 'volume']
            for feature in price_features:
                if feature in recent_data.columns:
                    values = recent_data[feature].values
                    normalized = (values - values.mean()) / (values.std() + 1e-8)
                    features.extend(normalized[-5:])  # 最近5个值
            
            # 技术指标特征
            tech_features = ['rsi_normalized', 'bb_position', 'price_vs_ema_fast', 
                           'macd_normalized', 'atr_normalized']
            for feature in tech_features:
                if feature in recent_data.columns:
                    values = recent_data[feature].fillna(0).values
                    features.extend(values[-5:])  # 最近5个值
            
            # SMC信号特征
            smc_features = ['smc_signal', 'bos_bullish', 'bos_bearish', 'po3_phase', 
                          'signal_confluence', 'market_structure']
            for feature in smc_features:
                if feature in recent_data.columns:
                    values = recent_data[feature].fillna(0).values
                    features.extend(values[-3:])  # 最近3个值
            
            # 确保特征长度一致
            expected_length = 98  # 根据环境定义调整
            if len(features) < expected_length:
                features.extend([0.0] * (expected_length - len(features)))
            elif len(features) > expected_length:
                features = features[:expected_length]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ 观测数据准备错误: {e}")
            return np.zeros(98, dtype=np.float32)  # 返回零向量作为后备
    
    async def _execute_action(self, action: int, current_data: pd.Series):
        """✅ 增强的交易动作执行 - 整合风控检查"""
        try:
            current_price = current_data['close']
            current_time = time.time()
            
            # ✅ 1. 首先进行增强风控检查
            should_force_exit, exit_reason, exit_info = await self._check_enhanced_risk_control_live(current_price)
            
            if should_force_exit:
                # 强制平仓
                await self._close_position(f"风控强制平仓: {exit_reason}")
                self.logger.warning(f"🚨 实盘风控强制平仓: {exit_reason} - {exit_info}")
                return
            
            # ✅ 2. 检查基础风险控制限制
            if not self._check_risk_limits(current_time):
                return
            
            # ✅ 3. 执行交易动作
            # 动作映射: 0=持仓/观望, 1=开多, 2=开空, 3=平仓
            if action == 1 and self.current_position['side'] is None:
                await self._open_long_position(current_price, current_data)
            elif action == 2 and self.current_position['side'] is None:
                await self._open_short_position(current_price, current_data)
            elif action == 3 and self.current_position['side'] is not None:
                await self._close_position("PPO模型建议平仓")
            
            # ✅ 4. 检查传统止盈止损（保持兼容性）
            if not should_force_exit and self.current_position['side'] is not None:
                await self._check_stop_loss_take_profit(current_price)
            
            # ✅ 5. 更新移动止损（如果启用）
            if self.current_position['side'] is not None and self.enable_trailing_stop:
                current_pnl_pct = self._calculate_current_pnl_pct_live(current_price)
                if current_pnl_pct >= self.trailing_stop_activation and not self.risk_control_state['trailing_stop_active']:
                    self.risk_control_state['trailing_stop_active'] = True
                    self.logger.info(f"🔄 实盘移动止损激活: 当前盈利={current_pnl_pct*100:.2f}%")
                
                if self.risk_control_state['trailing_stop_active']:
                    self._update_trailing_stop_live(current_price)
                
        except Exception as e:
            self.logger.error(f"❌ 动作执行错误: {e}")
    
    def _check_risk_limits(self, current_time: float) -> bool:
        """检查风险控制限制"""
        # 检查每日交易次数
        if self.daily_stats['trades_count'] >= self.max_daily_trades:
            self.logger.warning("⚠️ 达到每日最大交易次数限制")
            return False
        
        # 检查每日最大亏损
        daily_pnl_pct = (self.daily_stats['current_balance'] - self.daily_stats['start_balance']) / self.daily_stats['start_balance']
        if daily_pnl_pct <= -self.max_daily_loss:
            self.logger.warning(f"⚠️ 达到每日最大亏损限制: {daily_pnl_pct*100:.2f}%")
            return False
        
        # 检查最小交易间隔
        if current_time - self.daily_stats['last_trade_time'] < self.min_trade_interval:
            return False
        
        return True
    
    async def _open_long_position(self, price: float, data: pd.Series):
        """开多头仓位"""
        try:
            # 计算动态杠杆
            leverage = self._calculate_dynamic_leverage(data, 'long')
            
            # 计算仓位大小
            balance = self.daily_stats['current_balance']
            position_value = balance * self.config.get('MAX_POSITION_SIZE', 0.1)
            leveraged_value = position_value * leverage
            size = leveraged_value / price
            
            # 计算止损止盈
            stop_loss = self._calculate_dynamic_stop_loss(price, 'long', data)
            take_profit = self._calculate_dynamic_take_profit(price, 'long', data)
            
            if self.dry_run:
                # 模拟交易
                self.current_position.update({
                    'side': 'long',
                    'size': size,
                    'entry_price': price,
                    'leverage': leverage,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now()
                })
                
                # ✅ 设置增强风控状态
                self._set_initial_stop_loss_take_profit_live(price, 'long')
                
                self.logger.info(f"📈 模拟开多: 价格={price:.4f}, 数量={size:.4f}, 杠杆={leverage:.2f}x")
            else:
                # 实际交易
                order = await self._place_market_order('SOLUSDT', 'buy', size, leverage)
                if order:
                    self.current_position.update({
                        'side': 'long',
                        'size': size,
                        'entry_price': order['price'],
                        'leverage': leverage,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now()
                    })
                    
                    # ✅ 设置增强风控状态
                    self._set_initial_stop_loss_take_profit_live(order['price'], 'long')
                    
                    self.logger.info(f"📈 实盘开多: 订单ID={order['id']}, 价格={order['price']:.4f}")
            
            # 更新统计
            self.daily_stats['trades_count'] += 1
            self.daily_stats['last_trade_time'] = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ 开多仓位失败: {e}")
    
    async def _open_short_position(self, price: float, data: pd.Series):
        """开空头仓位"""
        try:
            # 计算动态杠杆
            leverage = self._calculate_dynamic_leverage(data, 'short')
            
            # 计算仓位大小
            balance = self.daily_stats['current_balance']
            position_value = balance * self.config.get('MAX_POSITION_SIZE', 0.1)
            leveraged_value = position_value * leverage
            size = leveraged_value / price
            
            # 计算止损止盈
            stop_loss = self._calculate_dynamic_stop_loss(price, 'short', data)
            take_profit = self._calculate_dynamic_take_profit(price, 'short', data)
            
            if self.dry_run:
                # 模拟交易
                self.current_position.update({
                    'side': 'short',
                    'size': size,
                    'entry_price': price,
                    'leverage': leverage,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now()
                })
                
                # ✅ 设置增强风控状态
                self._set_initial_stop_loss_take_profit_live(price, 'short')
                
                self.logger.info(f"📉 模拟开空: 价格={price:.4f}, 数量={size:.4f}, 杠杆={leverage:.2f}x")
            else:
                # 实际交易
                order = await self._place_market_order('SOLUSDT', 'sell', size, leverage)
                if order:
                    self.current_position.update({
                        'side': 'short',
                        'size': size,
                        'entry_price': order['price'],
                        'leverage': leverage,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now()
                    })
                    
                    # ✅ 设置增强风控状态
                    self._set_initial_stop_loss_take_profit_live(order['price'], 'short')
                    
                    self.logger.info(f"📉 实盘开空: 订单ID={order['id']}, 价格={order['price']:.4f}")
            
            # 更新统计
            self.daily_stats['trades_count'] += 1
            self.daily_stats['last_trade_time'] = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ 开空仓位失败: {e}")
    
    async def _close_position(self, reason: str):
        """平仓"""
        try:
            if self.current_position['side'] is None:
                return
            
            current_price = self.kline_buffer[-1]['close'] if self.kline_buffer else 0
            
            # 计算盈亏
            if self.current_position['side'] == 'long':
                pnl = self.current_position['size'] * (current_price - self.current_position['entry_price'])
            else:
                pnl = self.current_position['size'] * (self.current_position['entry_price'] - current_price)
            
            if self.dry_run:
                # 模拟交易
                self.daily_stats['current_balance'] += pnl
                self.daily_stats['total_pnl'] += pnl
                
                self.logger.info(f"🔄 模拟平仓: 原因={reason}, 盈亏={pnl:.2f} USDT")
            else:
                # 实际交易
                side = 'sell' if self.current_position['side'] == 'long' else 'buy'
                order = await self._place_market_order('SOLUSDT', side, self.current_position['size'])
                if order:
                    self.logger.info(f"🔄 实盘平仓: 订单ID={order['id']}, 原因={reason}")
            
            # 重置仓位
            self.current_position.update({
                'side': None,
                'size': 0.0,
                'entry_price': 0.0,
                'leverage': 1.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'entry_time': None
            })
            
            # ✅ 重置风控状态
            self.risk_control_state.update({
                'stop_loss_price': 0.0,
                'take_profit_price': 0.0,
                'trailing_stop_price': 0.0,
                'trailing_stop_active': False,
                'highest_profit': 0.0,
                'position_entry_time': 0
            })
            
            # ✅ 更新风控统计
            self.risk_control_state['daily_pnl'] += pnl
            if pnl < 0:
                self.risk_control_state['consecutive_losses'] += 1
            else:
                self.risk_control_state['consecutive_losses'] = 0
            
        except Exception as e:
            self.logger.error(f"❌ 平仓失败: {e}")
    
    async def _check_stop_loss_take_profit(self, current_price: float):
        """检查止盈止损"""
        if self.current_position['side'] is None:
            return
        
        should_close = False
        reason = ""
        
        if self.current_position['side'] == 'long':
            if current_price <= self.current_position['stop_loss']:
                should_close = True
                reason = f"多头止损: {current_price:.4f} <= {self.current_position['stop_loss']:.4f}"
            elif current_price >= self.current_position['take_profit']:
                should_close = True
                reason = f"多头止盈: {current_price:.4f} >= {self.current_position['take_profit']:.4f}"
        else:
            if current_price >= self.current_position['stop_loss']:
                should_close = True
                reason = f"空头止损: {current_price:.4f} >= {self.current_position['stop_loss']:.4f}"
            elif current_price <= self.current_position['take_profit']:
                should_close = True
                reason = f"空头止盈: {current_price:.4f} <= {self.current_position['take_profit']:.4f}"
        
        if should_close:
            await self._close_position(reason)
    
    def _calculate_dynamic_leverage(self, data: pd.Series, side: str) -> float:
        """计算动态杠杆"""
        base_leverage = 1.0
        
        # 基于信号强度调整杠杆
        smc_signal = data.get('smc_signal', 0)
        signal_confluence = data.get('signal_confluence', 0)
        atr_normalized = data.get('atr_normalized', 0.5)
        
        # 信号强度系数
        if side == 'long' and smc_signal > 0.3:
            leverage_multiplier = min(2.0, 1 + smc_signal)
        elif side == 'short' and smc_signal < -0.3:
            leverage_multiplier = min(2.0, 1 + abs(smc_signal))
        else:
            leverage_multiplier = 1.0
        
        # 信号汇聚系数
        if signal_confluence > 0.5:
            leverage_multiplier *= 1.2
        
        # 波动率调整（高波动率降低杠杆）
        if atr_normalized > 0.7:
            leverage_multiplier *= 0.8
        elif atr_normalized < 0.3:
            leverage_multiplier *= 1.1
        
        final_leverage = base_leverage * leverage_multiplier
        return min(max(final_leverage, 1.0), 5.0)  # 限制在1-5x之间
    
    def _calculate_dynamic_stop_loss(self, entry_price: float, side: str, data: pd.Series) -> float:
        """计算动态止损价位"""
        atr = data.get('atr', entry_price * 0.02)  # 默认2%
        atr_multiplier = 2.0
        
        if side == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:
            stop_loss = entry_price + (atr * atr_multiplier)
        
        return stop_loss
    
    def _calculate_dynamic_take_profit(self, entry_price: float, side: str, data: pd.Series) -> float:
        """计算动态止盈价位"""
        atr = data.get('atr', entry_price * 0.02)
        atr_multiplier = 3.0  # 风险收益比1:1.5
        
        # 检查是否有阻力/支撑位
        bb_upper = data.get('bb_upper', 0)
        bb_lower = data.get('bb_lower', 0)
        
        if side == 'long':
            atr_target = entry_price + (atr * atr_multiplier)
            if bb_upper > 0 and bb_upper < atr_target:
                take_profit = bb_upper * 0.99  # 稍微保守一点
            else:
                take_profit = atr_target
        else:
            atr_target = entry_price - (atr * atr_multiplier)
            if bb_lower > 0 and bb_lower > atr_target:
                take_profit = bb_lower * 1.01
            else:
                take_profit = atr_target
        
        return take_profit
    
    async def _place_market_order(self, symbol: str, side: str, amount: float, leverage: float = 1.0) -> Optional[Dict]:
        """下市价单"""
        try:
            if self.dry_run:
                return None
            
            # 设置杠杆
            await self.exchange.set_leverage(leverage, symbol)
            
            # 下单
            order = await self.exchange.create_market_order(symbol, side, amount)
            
            self.logger.info(f"✅ 订单执行成功: {side} {amount:.4f} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"❌ 订单执行失败: {e}")
            return None
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        total_return = (self.daily_stats['current_balance'] - self.daily_stats['start_balance']) / self.daily_stats['start_balance']
        
        return {
            'start_balance': self.daily_stats['start_balance'],
            'current_balance': self.daily_stats['current_balance'],
            'total_pnl': self.daily_stats['total_pnl'],
            'total_return': total_return,
            'trades_count': self.daily_stats['trades_count'],
            'current_position': self.current_position.copy(),
            'ws_connected': self.ws_connected
        }

    async def _check_enhanced_risk_control_live(self, current_price: float) -> Tuple[bool, str, Dict]:
        """
        ✅ 实盘风控检查 - 与交易环境保持一致
        
        Returns:
            (should_exit, exit_reason, exit_info)
        """
        if self.current_position['side'] is None:
            return False, "", {}
        
        exit_info = {}
        current_pnl_pct = self._calculate_current_pnl_pct_live(current_price)
        
        # 1. ✅ 基础止损检查
        if self.risk_control_state['stop_loss_price'] > 0:
            if ((self.current_position['side'] == 'long' and current_price <= self.risk_control_state['stop_loss_price']) or
                (self.current_position['side'] == 'short' and current_price >= self.risk_control_state['stop_loss_price'])):
                
                self.risk_control_state['stop_loss_triggered_count'] += 1
                exit_info = {
                    'exit_type': 'stop_loss',
                    'exit_price': current_price,
                    'stop_loss_price': self.risk_control_state['stop_loss_price'],
                    'pnl_pct': current_pnl_pct,
                    'triggered_count': self.risk_control_state['stop_loss_triggered_count']
                }
                self.logger.info(f"🛑 实盘止损触发: 价格={current_price:.4f}, 止损线={self.risk_control_state['stop_loss_price']:.4f}, 亏损={current_pnl_pct*100:.2f}%")
                return True, "stop_loss", exit_info
        
        # 2. ✅ 基础止盈检查
        if self.risk_control_state['take_profit_price'] > 0:
            if ((self.current_position['side'] == 'long' and current_price >= self.risk_control_state['take_profit_price']) or
                (self.current_position['side'] == 'short' and current_price <= self.risk_control_state['take_profit_price'])):
                
                self.risk_control_state['take_profit_triggered_count'] += 1
                exit_info = {
                    'exit_type': 'take_profit',
                    'exit_price': current_price,
                    'take_profit_price': self.risk_control_state['take_profit_price'],
                    'pnl_pct': current_pnl_pct,
                    'triggered_count': self.risk_control_state['take_profit_triggered_count']
                }
                self.logger.info(f"🎯 实盘止盈触发: 价格={current_price:.4f}, 止盈线={self.risk_control_state['take_profit_price']:.4f}, 盈利={current_pnl_pct*100:.2f}%")
                return True, "take_profit", exit_info
        
        # 3. ✅ 移动止损检查
        if self.enable_trailing_stop and self.risk_control_state['trailing_stop_active']:
            # 更新历史最高盈利
            if current_pnl_pct > self.risk_control_state['highest_profit']:
                self.risk_control_state['highest_profit'] = current_pnl_pct
                # 更新移动止损线
                self._update_trailing_stop_live(current_price)
            
            # 检查移动止损触发
            if self.risk_control_state['trailing_stop_price'] > 0:
                if ((self.current_position['side'] == 'long' and current_price <= self.risk_control_state['trailing_stop_price']) or
                    (self.current_position['side'] == 'short' and current_price >= self.risk_control_state['trailing_stop_price'])):
                    
                    self.risk_control_state['trailing_stop_triggered_count'] += 1
                    exit_info = {
                        'exit_type': 'trailing_stop',
                        'exit_price': current_price,
                        'trailing_stop_price': self.risk_control_state['trailing_stop_price'],
                        'highest_profit': self.risk_control_state['highest_profit'],
                        'pnl_pct': current_pnl_pct,
                        'triggered_count': self.risk_control_state['trailing_stop_triggered_count']
                    }
                    self.logger.info(f"📈 实盘移动止损触发: 价格={current_price:.4f}, 移动止损线={self.risk_control_state['trailing_stop_price']:.4f}")
                    return True, "trailing_stop", exit_info
        
        # 4. ✅ 最大单笔亏损检查
        if current_pnl_pct < -self.max_single_loss:
            exit_info = {
                'exit_type': 'max_single_loss',
                'exit_price': current_price,
                'pnl_pct': current_pnl_pct,
                'max_single_loss': self.max_single_loss
            }
            self.logger.warning(f"💥 实盘单笔最大亏损触发: 当前亏损={current_pnl_pct*100:.2f}%, 限制={self.max_single_loss*100:.1f}%")
            return True, "max_single_loss", exit_info
        
        # 5. ✅ 日内最大亏损检查
        daily_loss_pct = self.risk_control_state['daily_pnl'] / self.daily_stats['start_balance']
        if daily_loss_pct < -self.max_daily_loss:
            exit_info = {
                'exit_type': 'daily_loss_limit',
                'exit_price': current_price,
                'daily_pnl': self.risk_control_state['daily_pnl'],
                'daily_loss_pct': daily_loss_pct,
                'daily_loss_limit': self.max_daily_loss
            }
            self.logger.warning(f"📉 实盘日内最大亏损触发: 日内亏损={daily_loss_pct*100:.2f}%, 限制={self.max_daily_loss*100:.1f}%")
            return True, "daily_loss_limit", exit_info
        
        return False, "", {}
    
    def _calculate_current_pnl_pct_live(self, current_price: float) -> float:
        """计算当前持仓盈亏百分比"""
        if self.current_position['side'] is None:
            return 0.0
        
        entry_price = self.current_position['entry_price']
        price_change = (current_price - entry_price) / entry_price
        
        if self.current_position['side'] == 'long':
            return price_change
        else:  # short
            return -price_change
    
    def _update_trailing_stop_live(self, current_price: float):
        """更新移动止损线 - 实盘版本"""
        if not self.risk_control_state['trailing_stop_active'] or self.current_position['side'] is None:
            return
        
        # 激活移动止损的条件检查
        current_pnl_pct = self._calculate_current_pnl_pct_live(current_price)
        if not self.risk_control_state['trailing_stop_active'] and current_pnl_pct >= self.trailing_stop_activation:
            self.risk_control_state['trailing_stop_active'] = True
            self.logger.info(f"🔄 实盘移动止损激活: 当前盈利={current_pnl_pct*100:.2f}%, 激活阈值={self.trailing_stop_activation*100:.1f}%")
        
        if self.risk_control_state['trailing_stop_active']:
            # 计算新的移动止损线
            if self.current_position['side'] == 'long':  # 多头
                new_trailing_stop = current_price * (1 - self.trailing_stop_distance)
                if new_trailing_stop > self.risk_control_state['trailing_stop_price']:
                    self.risk_control_state['trailing_stop_price'] = new_trailing_stop
                    self.logger.debug(f"📈 实盘多头移动止损更新: {new_trailing_stop:.4f}")
            else:  # 空头
                new_trailing_stop = current_price * (1 + self.trailing_stop_distance)
                if new_trailing_stop < self.risk_control_state['trailing_stop_price'] or self.risk_control_state['trailing_stop_price'] == 0:
                    self.risk_control_state['trailing_stop_price'] = new_trailing_stop
                    self.logger.debug(f"📉 实盘空头移动止损更新: {new_trailing_stop:.4f}")
    
    def _set_initial_stop_loss_take_profit_live(self, entry_price: float, side: str):
        """设置初始止盈止损价位 - 实盘版本"""
        # 使用固定比例设置
        if side == 'long':
            self.risk_control_state['stop_loss_price'] = entry_price * (1 - self.base_stop_loss)
            self.risk_control_state['take_profit_price'] = entry_price * (1 + self.base_take_profit)
        else:  # short
            self.risk_control_state['stop_loss_price'] = entry_price * (1 + self.base_stop_loss)
            self.risk_control_state['take_profit_price'] = entry_price * (1 - self.base_take_profit)
        
        # 初始化移动止损
        if self.enable_trailing_stop:
            self.risk_control_state['trailing_stop_price'] = self.risk_control_state['stop_loss_price']
            self.risk_control_state['trailing_stop_active'] = False
            self.risk_control_state['highest_profit'] = 0.0
        
        # 记录设置
        self.risk_control_state['position_entry_time'] = time.time()
        
        self.logger.info(f"🎯 实盘设置止盈止损: 入场={entry_price:.4f}, 止损={self.risk_control_state['stop_loss_price']:.4f}, 止盈={self.risk_control_state['take_profit_price']:.4f}")

# 异步运行示例
async def main():
    """主函数，用于测试实时交易功能"""
    print("🚀 实时交易系统测试")
    
    # 配置参数
    model_path = "models/ppo_sol_trading_model.zip"  # 需要先训练模型
    
    # 创建实时交易器（模拟模式）
    trader = LiveTrader(
        model_path=model_path,
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True,
        dry_run=True  # 模拟模式
    )
    
    try:
        # 启动实时交易
        await trader.start_live_trading()
    except KeyboardInterrupt:
        print("\n🛑 接收到停止信号")
        await trader.stop_live_trading()

if __name__ == "__main__":
    asyncio.run(main()) 