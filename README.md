# SOL/USDT 中频合约交易策略

基于强化学习PPO算法的SOL/USDT中频合约交易策略，集成PO3/SMC信号与经典技术指标。

## 项目特点

- 🧠 **强化学习驱动**: 使用PPO算法进行策略训练
- 📊 **技术指标融合**: RSI、布林带、EMA等经典指标
- 🎯 **SMC信号集成**: PO3阶段、BOS、Order Block等市场结构信号
- 🔄 **实时交易支持**: 支持Binance API实时交易
- 📈 **可视化分析**: 详细的回测分析与图表展示
- 🛡️ **风险控制**: 完善的止损止盈和资金管理机制

## 项目结构

```
├── data/                    # 数据模块
│   ├── data_collector.py   # 数据收集器
│   ├── technical_indicators.py  # 技术指标计算
│   └── smc_signals.py      # SMC信号计算
├── environment/             # 交易环境
│   └── trading_env.py      # Gym交易环境
├── models/                  # 模型相关
│   ├── ppo_trainer.py      # PPO训练器
│   └── model_evaluation.py # 模型评估
├── trading/                 # 交易模块
│   ├── live_trader.py      # 实时交易器
│   └── risk_manager.py     # 风险管理
├── analysis/                # 分析模块
│   ├── backtest.py         # 回测分析
│   └── explainability.py   # 可解释性分析
├── utils/                   # 工具模块
│   ├── config.py           # 配置管理
│   └── logger.py           # 日志管理
└── main.py                 # 主程序入口
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥：
```bash
cp .env.example .env
# 编辑 .env 文件，填入Binance API密钥
```

3. 数据准备：
```bash
python -m data.data_collector
```

4. 训练模型：
```bash
python main.py --mode train
```

5. 回测评估：
```bash
python main.py --mode backtest
```

6. 实时交易：
```bash
python main.py --mode live
```

## 配置说明

所有配置项都在 `utils/config.py` 中统一管理，支持通过环境变量或配置文件进行调整。

## 注意事项

- 本项目仅供学习研究使用
- 实际交易存在风险，请谨慎使用
- 建议先在模拟环境中充分测试 