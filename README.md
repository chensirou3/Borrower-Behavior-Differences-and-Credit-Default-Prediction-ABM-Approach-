# ACR: 信贷风控代理模型 | Agent-Based Credit Risk Modeling
## 基于代理建模的信贷风控与行为画像特征研究 | Agent-Based Credit Risk Modeling with Digital Profile Proxies

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**基于代理的信贷风险建模框架，集成行为画像特征的本科可发表研究项目。**  
**A comprehensive agent-based modeling framework for credit risk analysis with behavioral digital profile proxies.**

## 项目概述 | Project Overview

本项目实现了一个模块化的信贷风险仿真系统，主要特点：  
This project implements a modular credit risk simulation system with key features:

- **代理建模 | Agent-Based Modeling**：基于个体特质的借款人行为仿真 | Individual trait-based borrower behavior simulation
- **画像代理 | Digital Proxies**：通过弱相关映射将潜在特质转换为可观测的行为画像 | Converting latent traits to observable behavioral profiles through weak correlation mapping
- **环境周期 | Economic Cycles**：十年经济周期的利率和宏观环境变化 | Interest rate and macro environment changes in 10-year economic cycles
- **风控建模 | Risk Modeling**：传统基线特征 vs 增强画像特征的比较 | Comparison of traditional baseline features vs enhanced profile features
- **公平性分析 | Fairness Analysis**：不同人群的模型公平性评估 | Model fairness evaluation across different populations

### 核心发现 | Key Findings

基于50,000借款人×30年的541万+事件 | Based on 5.4+ million events from 50,000 borrowers over 30 years:

- **🚀 AUC提升+5.2%** (0.561 → 0.590) | **+5.2% AUC improvement**
- **📉 违约率降低** 所有批准阈值下 | **Lower default rates** at all approval thresholds  
- **⚖️ 公平性改善** TPR差距减少 | **Improved fairness** with reduced TPR gaps
- **🎯 周期稳健性** 宽松/紧缩制度下一致 | **Cycle robustness** consistent across loose/tight regimes

## 快速开始 | Quick Start

### 安装 | Installation

```bash
# 安装依赖 | Install dependencies
pip install -r requirements.txt

# 开发模式安装 | Install in development mode
pip install -e .
```

### 基础运行 | Basic Usage

```bash
# 运行默认仿真实验 | Run default simulation experiment
acr run-sim --config configs/experiment.yaml

# 评估模型性能 | Evaluate model performance
acr eval-models outputs/run_YYYYMMDD_HHMMSS/

# 生成可视化图表 | Generate visualization charts
acr plots outputs/run_YYYYMMDD_HHMMSS/

# 运行质量诊断 | Run quality diagnostics
acr fix-plots outputs/run_YYYYMMDD_HHMMSS/

# 参数覆盖示例 | Parameter override examples
acr run-sim --set population.N=10000 --set timeline.T=120
```

### 配置定制 | Configuration

主要配置参数 | Main configuration parameters:

```yaml
# 人口规模和时间跨度 | Population size and timeline
population:
  N: 50000              # 借款人数 | Number of borrowers
timeline:
  T: 360                # 月数，30年 | Months, 30 years

# 特质分布（独立截断正态）| Trait distributions (independent truncated normal)
traits:
  gamma:  { mean: 2.0,  sd: 0.6,  min: 0.5,  max: null }    # 风险偏好 | Risk appetite
  beta:   { mean: 0.90, sd: 0.08, min: 0.60, max: 1.00 }   # 财务纪律 | Financial discipline
  kappa:  { mean: 0.50, sd: 0.25, min: 0.00, max: 1.50 }   # 行为波动 | Behavioral volatility
  omega:  { mean: 0.00, sd: 0.80, min: null, max: null }    # 冲击敏感性 | Shock sensitivity
  eta:    { mean: 0.70, sd: 0.20, min: 0.00, max: 1.00 }   # 学习能力 | Learning ability

# 画像代理映射 | Digital proxy mappings
proxies:
  noise_sd: 0.12        # 噪声标准差 | Noise standard deviation
  mapping:
    night_active_ratio:   { kappa: +0.50, beta: -0.20, intercept: 0.20, clip: [0.0, 1.0] }
    session_std:          { kappa: +0.80, intercept: 0.50, min: 0.01 }
    task_completion_ratio:{ kappa: -0.40, beta: -0.20, intercept: 0.85, clip: [0.0, 1.0] }
    spending_volatility:  { kappa: +0.50, beta: -0.20, omega: +0.30, intercept: 0.30, min: 0.01 }

# 经济环境 | Economic environment
environment:
  sine:
    enabled: true
    period: 120         # 10年周期 | 10-year cycle
    ar1_rho: 0.2        # AR(1)系数 | AR(1) coefficient
    noise_sd: 0.05      # 噪声标准差 | Noise standard deviation
```

## 模块架构 | Module Architecture

- `acr.config`: 配置解析与校验 | Configuration parsing and validation
- `acr.traits`: 特质采样与原型 | Trait sampling and prototypes
- `acr.proxies`: 画像代理映射 | Digital proxy mapping
- `acr.environment`: 环境周期与机制 | Environment cycles and mechanisms
- `acr.agents`: 借款人代理 | Borrower agents
- `acr.bank`: 银行决策策略 | Bank decision policies
- `acr.dgp`: 数据生成过程 | Data generation process
- `acr.simulation`: 主仿真循环 | Main simulation loop
- `acr.features`: 特征集构造 | Feature set construction
- `acr.models`: 模型训练管道 | Model training pipeline
- `acr.evaluation`: 评估指标与公平性 | Evaluation metrics and fairness
- `acr.viz`: 可视化与质量保证 | Visualization and quality assurance
- `acr.io`: 输入输出管理 | Input/output management
- `acr.cli`: 命令行界面 | Command-line interface

### 核心数据格式 | Core Data Formats

所有输出采用开放格式 | All outputs use open formats:
- **事件数据 | Event Data**: `events.csv` (主要分析数据 | Main analysis data)
- **指标结果 | Metrics Results**: `metrics.csv` (性能指标 | Performance metrics)
- **公平性分析 | Fairness Analysis**: `fairness.json` (公平性指标 | Fairness metrics)
- **图表文件 | Chart Files**: PNG格式 | PNG format (可发表质量 | Publication quality)
- **配置快照 | Configuration Snapshot**: `config.yaml` (完整可复现 | Complete reproducibility)

## 数据格式

所有输出采用开放格式：
- 事件数据：`events.csv`
- 指标结果：`metrics.csv`
- 公平性分析：`fairness.json`
- 图表：PNG格式
- 配置快照：`manifest.json`

## 扩展指南

### 添加新的特质采样器

```python
from acr.traits.sampler import TraitSampler

class CustomTraitSampler(TraitSampler):
    def sample(self, N: int, rng: np.random.Generator) -> pd.DataFrame:
        # 实现自定义采样逻辑
        pass
```

### 修改环境机制

```python
from acr.environment.cycles import build_sine_env

def build_custom_env(T: int, cfg, rng) -> EnvSeries:
    # 实现自定义环境逻辑
    pass
```

### 自定义银行策略

```python
from acr.bank.policy import DecisionPolicy

class CustomDecisionPolicy(DecisionPolicy):
    def approve(self, scores: np.ndarray, mode: str, q_or_tau: float) -> np.ndarray:
        # 实现自定义决策逻辑
        pass
```

## 开发阶段

- **阶段 0** (当前): 基础MVE - 独立特质、弱相关画像、基础模型
- **阶段 1**: 银行反馈机制
- **阶段 2**: 相关特质与原型
- **阶段 3**: 复杂环境机制
- **阶段 4**: 公平性优化

## 测试

```bash
pytest tests/
```

## 许可证

MIT License
