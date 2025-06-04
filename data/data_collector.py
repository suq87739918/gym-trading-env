"""
数据收集器模块
负责从Binance API获取SOL/USDT历史K线数据
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple
import ccxt
from binance.client import Client
from utils.config import get_config
from utils.logger import get_logger

class DataCollector:
    """数据收集器类"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger('DataCollector', 'data_collector.log')
        
        # 初始化Binance客户端
        try:
            if self.config.get('BINANCE_TESTNET'):
                self.client = Client(
                    self.config.get('BINANCE_API_KEY'),
                    self.config.get('BINANCE_SECRET_KEY'),
                    testnet=True
                )
            else:
                self.client = Client(
                    self.config.get('BINANCE_API_KEY'),
                    self.config.get('BINANCE_SECRET_KEY')
                )
            
            # 初始化CCXT交换所（备用方案）
            self.exchange = ccxt.binance({
                'apiKey': self.config.get('BINANCE_API_KEY'),
                'secret': self.config.get('BINANCE_SECRET_KEY'),
                'sandbox': self.config.get('BINANCE_TESTNET'),
                'enableRateLimit': True,
            })
            
        except Exception as e:
            self.logger.error(f"初始化Binance客户端失败: {e}")
            self.client = None
            self.exchange = None
    
    def get_historical_data(self, symbol: str = None, timeframe: str = None,
                          start_date: str = None, end_date: str = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        获取历史数据 - 优化版，默认获取最近3个月数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期  
            limit: 数据条数限制
        """
        symbol = symbol or self.config.get('SYMBOL')
        timeframe = timeframe or self.config.get('TIMEFRAME')
        
        # ✅ 修改：默认获取最近3个月数据（约6500条15分钟数据）
        if start_date is None and end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')  # 3个月前
            limit = 6500  # 3个月的15分钟数据约6500条
            self.logger.info(f"📅 使用最近3个月数据进行训练: {start_date} 到 {end_date}")
        
        try:
            self.logger.info(f"📊 开始获取 {symbol} {timeframe} 历史数据...")
            self.logger.info(f"📅 时间范围: {start_date} 到 {end_date}")
            self.logger.info(f"📈 预计获取 {limit} 条数据")
            
            df = None
            
            # 优先使用Binance API
            if self.client:
                self.logger.info("🔑 使用Binance API获取数据...")
                df = self._get_data_with_binance(symbol, timeframe, start_date, end_date, limit)
            
            # 如果Binance失败，使用CCXT
            if df is None or df.empty:
                self.logger.info("🔄 使用CCXT获取数据...")
                df = self._get_data_with_ccxt(symbol, timeframe, start_date, end_date)
            
            if df is not None and not df.empty:
                # 数据预处理
                df = self._preprocess_data(df)
                self.logger.info(f"✅ 成功获取 {len(df)} 条历史数据")
                return df
            else:
                self.logger.warning("⚠️ 获取历史数据失败，返回空DataFrame")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ 获取历史数据失败: {e}")
            return pd.DataFrame()
    
    def _get_data_with_binance(self, symbol: str, timeframe: str, 
                              start_date: str, end_date: str, limit: int) -> Optional[pd.DataFrame]:
        """使用python-binance获取数据"""
        if not self.client:
            return None
        
        try:
            # 转换时间格式
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # 转换时间间隔格式
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
            }
            
            interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_15MINUTE)
            
            # 分批获取数据
            all_data = []
            current_start = start_ts
            
            while current_start < end_ts:
                # 获取当前批次数据
                klines = self.client.get_historical_klines(
                    symbol, interval, current_start, min(current_start + limit * self._get_interval_ms(timeframe), end_ts)
                )
                
                if not klines:
                    break
                
                # 转换为DataFrame
                df_batch = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                all_data.append(df_batch)
                
                # 更新下一批次的开始时间
                current_start = int(klines[-1][6]) + 1  # 使用close_time + 1
                
                # 避免请求过于频繁
                time.sleep(0.1)
                
                self.logger.info(f"已获取 {len(all_data)} 批数据，当前时间: {datetime.fromtimestamp(current_start/1000)}")
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"使用python-binance获取数据失败: {e}")
            return None
    
    def _get_data_with_ccxt(self, symbol: str, timeframe: str, 
                           start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """使用CCXT获取数据"""
        if not self.exchange:
            return None
        
        try:
            # 转换时间格式
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # 获取数据
            ohlcv_data = []
            current_since = since
            
            while current_since < until:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                
                if not ohlcv:
                    break
                
                ohlcv_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + self._get_interval_ms(timeframe)
                
                # 避免请求过于频繁
                time.sleep(self.exchange.rateLimit / 1000)
            
            if ohlcv_data:
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"使用CCXT获取数据失败: {e}")
            return None
    
    def _get_interval_ms(self, timeframe: str) -> int:
        """获取时间间隔的毫秒数"""
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return interval_map.get(timeframe, 15 * 60 * 1000)
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 转换数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 转换时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        
        # 去除重复数据
        df = df[~df.index.duplicated(keep='last')]
        
        # 按时间排序
        df = df.sort_index()
        
        # 填补缺失值
        df = df.fillna(method='ffill')
        
        # 只保留必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        
        # 移除异常值（价格为0或负数）
        df = df[(df['close'] > 0) & (df['volume'] >= 0)]
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None):
        """保存数据到文件"""
        if filename is None:
            symbol = self.config.get('SYMBOL')
            timeframe = self.config.get('TIMEFRAME')
            filename = f"{symbol}_{timeframe}_data.csv"
        
        # 创建数据目录
        data_dir = self.config.get('DATA_DIR')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath)
        
        self.logger.info(f"数据已保存到: {filepath}")
        return filepath
    
    def load_data(self, filename: str = None) -> pd.DataFrame:
        """从文件加载数据"""
        if filename is None:
            symbol = self.config.get('SYMBOL')
            timeframe = self.config.get('TIMEFRAME')
            filename = f"{symbol}_{timeframe}_data.csv"
        
        data_dir = self.config.get('DATA_DIR')
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"从文件加载了 {len(df)} 条数据记录")
            return df
        else:
            self.logger.warning(f"数据文件不存在: {filepath}")
            return pd.DataFrame()
    
    def get_latest_data(self, symbol: str = None, timeframe: str = None, 
                       count: int = 100) -> pd.DataFrame:
        """获取最新的K线数据"""
        symbol = symbol or self.config.get('SYMBOL')
        timeframe = timeframe or self.config.get('TIMEFRAME')
        
        try:
            if self.client:
                # 转换时间间隔格式
                interval_map = {
                    '1m': Client.KLINE_INTERVAL_1MINUTE,
                    '5m': Client.KLINE_INTERVAL_5MINUTE,
                    '15m': Client.KLINE_INTERVAL_15MINUTE,
                    '30m': Client.KLINE_INTERVAL_30MINUTE,
                    '1h': Client.KLINE_INTERVAL_1HOUR,
                    '4h': Client.KLINE_INTERVAL_4HOUR,
                    '1d': Client.KLINE_INTERVAL_1DAY,
                }
                
                interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_15MINUTE)
                
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=count
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    return self._preprocess_data(df)
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.exception(f"获取最新数据时发生错误: {e}")
            return pd.DataFrame()

def main():
    """主函数，用于直接运行数据收集"""
    collector = DataCollector()
    
    # 获取历史数据
    df = collector.get_historical_data()
    
    if not df.empty:
        # 保存数据
        filepath = collector.save_data(df)
        print(f"数据收集完成，共 {len(df)} 条记录，已保存到: {filepath}")
        
        # 显示数据摘要
        print("\n数据摘要:")
        print(df.describe())
        print(f"\n时间范围: {df.index.min()} 到 {df.index.max()}")
    else:
        print("数据收集失败")

if __name__ == "__main__":
    main() 