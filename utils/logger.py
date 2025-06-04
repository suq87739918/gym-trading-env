"""
日志管理模块
提供统一的日志记录功能
"""
import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional
from .config import get_config

class Logger:
    """日志管理类"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.config = get_config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.get('LOG_LEVEL', 'INFO')))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str]):
        """设置日志处理器"""
        formatter = logging.Formatter(self.config.get('LOG_FORMAT'))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            log_dir = self.config.get('LOG_DIR', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_path = os.path.join(log_dir, log_file)
            
            # 使用RotatingFileHandler避免日志文件过大
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """调试信息"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """一般信息"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告信息"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误信息"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常信息（包含堆栈跟踪）"""
        self.logger.exception(message, **kwargs)

class TradingLogger:
    """交易专用日志记录器"""
    
    def __init__(self):
        self.config = get_config()
        self.general_logger = Logger('trading', 'trading.log')
        self.trade_logger = Logger('trades', 'trades.log')
        self.error_logger = Logger('errors', 'errors.log')
    
    # 添加通用日志方法，使其与Logger接口兼容
    def debug(self, message: str, **kwargs):
        """调试信息"""
        self.general_logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """一般信息"""
        self.general_logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告信息"""
        self.general_logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误信息"""
        self.error_logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误"""
        self.error_logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常信息（包含堆栈跟踪）"""
        self.error_logger.exception(message, **kwargs)
    
    def log_trade(self, action: str, symbol: str, price: float, 
                  quantity: float, position_type: str = None, **kwargs):
        """记录交易信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_info = {
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'position_type': position_type,
            **kwargs
        }
        
        message = f"交易执行 - {action}: {symbol} @ {price}, 数量: {quantity}"
        if position_type:
            message += f", 类型: {position_type}"
        
        self.trade_logger.info(message)
        return trade_info
    
    def log_signal(self, signal_type: str, symbol: str, signal_strength: float, 
                   indicators: dict = None, **kwargs):
        """记录信号信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"信号检测 - {signal_type}: {symbol}, 强度: {signal_strength:.3f}"
        if indicators:
            indicator_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, (int, float)) 
                                     else f"{k}={v}" for k, v in indicators.items()])
            message += f", 指标: {indicator_str}"
        
        self.general_logger.info(message)
    
    def log_performance(self, metrics: dict):
        """记录性能指标"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"性能更新 - "
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) 
                               else f"{k}={v}" for k, v in metrics.items()])
        message += metric_str
        
        self.general_logger.info(message)
    
    def log_risk_event(self, event_type: str, details: dict):
        """记录风险事件"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"风险事件 - {event_type}: "
        detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        message += detail_str
        
        self.error_logger.warning(message)
    
    def log_error(self, error_type: str, error_message: str, **kwargs):
        """记录错误信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"错误发生 - {error_type}: {error_message}"
        if kwargs:
            detail_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            message += f", 详情: {detail_str}"
        
        self.error_logger.error(message)

# 全局日志实例
def get_logger(name: str, log_file: Optional[str] = None) -> Logger:
    """获取日志实例"""
    return Logger(name, log_file)

def get_trading_logger() -> TradingLogger:
    """获取交易日志实例"""
    return TradingLogger() 