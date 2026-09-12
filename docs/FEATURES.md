# PASM-Lite · 全部功能总览

> **零 token 认知智能体 · 开源教学版**（MIT）
> 本仓库 = **PASM 系统介绍 + 教学最小实现**，一体化（纯 Markdown）。
>
> 本文穷举 PASM-Lite 的**全部功能**：它实现了哪几块、能当什么用、每个文件提供什么能力。

---

## 一、它是什么

**PASM-Lite 是 PASM 的开源教学版**：把七层仿生认知架构压缩成一个可运行的最小实现，
去掉情绪 / 性格 / 元认知等生产模块，只保留 **"零 token 认知循环"的主干**。

> **核心命题**：AI 的"思考"一定要生成 token 吗？
> PASM 的回答是：感知 → 工作记忆 → 情景记忆检索 → 世界模型"想象" → 规划 → 行动，
> **全程连续潜在向量，零 token**。

**定位说明**：Lite 是**教学骨架，不是性能基线**。演示重点是"机制跑通 + 记忆可见"，不是刷分。

---

## 二、实现了哪几块

| 层 | 名字 | Lite 实现 | 说明 |
|---|---|---|---|
| 0 | 感知编码 | ✅ `AutoEncoder` | 27 维观测 → 自编码器压缩成 8 维潜在向量 `z` |
| 1 | 工作记忆 | ✅ `WorkMemory` | 最近 4 步上下文 `c`，注意力槽位 |
| 2 | 情景记忆 | ✅ `Episodic` | KNN 检索相似经历 + **事件驱动写入**（只在惊讶/奖励异常时写） |
| 3 | 世界模型 | ✅ `WorldModel` | GRU 在潜在空间预测下一状态 `z′` 与奖励 `r` |
| — | 规划 | ✅ `Planner` | 采样 16 条动作序列在**世界模型里"想象"**，挑累计奖励最高的一条执行 |
| — | 学习层 | ✅ `LearningLayer` | 关联式（Hebbian）经验吸附，可热插拔 |
| 4 | 情绪 / 多巴胺-血清素 | — | 完整引擎有（消融显示价值 11.7 奖励差） |
| 5 | 性格先验与发育可塑性 | — | 完整引擎有 |
| 6 | 元认知 / 全局工作空间 | — | 完整引擎有 |

**能力自描述**：`capabilities()` 会**如实申报**——教学版只覆盖主干四层，
情绪/性格/发育/元认知一律报 `False`，可用 `missing_vs()` 列出"相对完整引擎还缺什么"。

---

## 三、能当什么用（四种用法）

### 3.1 学习材料 —— 10 分钟读懂

单文件、约 380 行、无抽象层。改一改就玩：

```bash
pip install torch          # CPU 即可
python pasm_lite.py
```

```python
GridWorld(n_food=12)   # 让食物更密 → 观察行为变化
Episodic(k=5)          # 调检索宽度
Planner(n_seq=32)      # 增加"想象"预算
```

### 3.2 可运行的最小引擎 —— 插上去就能用

`engine.py` 实现了完整契约，`create("pasm-lite")` 即得一个合规引擎：

```python
from engine_api import create

eng = create("pasm-lite")            # 换成 create("pasm") 即换生产引擎，调用代码不变
eng.reset_episode()
action, report = eng.act(obs)
eng.learn(reward=r, next_obs=obs2)
print(eng.info().to_dict())          # 我是谁
print(eng.capabilities().layers())   # 我会什么
print(eng.snapshot()["memory"])      # 记忆现状
```

### 3.3 接口契约的对照样本 —— 换引擎不改代码

仓库自带 `engine_api.py`（与 PASM 核心 `pasm/engine_api.py` **同源镜像**，纯标准库零依赖）。
它把"引擎该有什么"固化成可校验契约，Lite 是这份契约的**参考实现**。

**意义**：上层依赖接口而非实现。今天用教学版跑通流程，明天换生产引擎，**调用代码一行都不用改**。

### 3.4 环境与学习层的可插拔示范

```python
from engine import LiteEngine

# ① 换世界：非网格环境也能跑
eng = LiteEngine(env_name="toy-vector")   # 连续向量世界：obs_dim=3, n_actions=2

# ② 换学习层：运行时热插拔（契约 + 潜维 + 动作数 三项校验）
ok, msg = eng.attach_learning(MyLearningLayer())
print(eng.learning_tier)                  # full / teaching / 自定义
```

> 核心引擎的 `LearningEngine`（离散动作 + 性格设计）可直接插进 Lite 引擎 ——
> 这就是"**同一接口两档实现**"的实证。

---

## 四、文件级功能清单

| 文件 | 提供的能力 |
|---|---|
| **`pasm_lite.py`** | 教学主干：`GridWorld` / `AutoEncoder` / `WorkMemory` / `Episodic` / `WorldModel` / `Planner` / `Agent`（含 `freeze_encoder` 编码器冻结）+ 命令行闭环 |
| **`engine.py`** | `LiteEngine` 引擎实现：契约六件套 + `warmup()` / `run_episode()` / `freeze_vae()` / `attach_learning()` / 快照七区块 / 存档读档（含档位校验） |
| **`engine_api.py`** | 同源契约镜像（api 1.1）：`EngineInfo` / `Capabilities` / `conforms` / `as_engine` / `Registry` / `create` / `create_best` / `capability_gap` / `normalize_snapshot` |
| **`envs.py`** | 环境插件：`grid-10x10`（网格）+ `toy-vector`（连续向量，非网格）；`register_env` / `make_env` / `env_names` / `env_conforms` |
| **`learning.py`** | 学习层（teaching 档，契约 `pasm.learning/1.0`）：`update(z,a,r)` / `bias(z)` / `recall_action(z)` / `info` / `capabilities` / `learn` / `state` / `apply_state` |
| **`mathlab.py`** | 数学脑（纯 numpy）：线性回归(R²) / 多元回归 / 描述统计 / 相关矩阵 / 线性方程组 / 矩阵特征 / 拓扑排序 / 环检测 / 最短路(Dijkstra) / 连通分量 / 马尔可夫稳态 / 表格分析 |
| **`verify_swap.py`** | 跨档互换验证：两档契约校验 + 同构驱动 + 运行时换档（含维度不符被拒、档位不符存档报错等边界） |
| **`index.md`** | 纯文字项目介绍页（Gitee 直接渲染） |
| **`docs/pasm_ecosystem.md`** | 完整能力清单、可度量效果、三仓生态与文档地图 |

### 4.1 环境契约（`envs.py`）

- **必需**：`reset() -> obs`、`step(action) -> 任意`
- **可选**：`observe()` / `close()` / `spec()` / `obs_dim` / `n_actions`
- **自动适配**：引擎读环境申报的维度，自动调整感知、世界模型与规划器
- **登记新世界**：
  ```python
  from envs import register_env
  register_env("我的世界", lambda **kw: MyEnv(**kw), info={"obs_dim": 5, "n_actions": 3})
  ```

### 4.2 学习层契约（`learning.py`）

契约五件套：`info` / `capabilities` / `learn` / `bias` / `state`（另加 `apply_state` 恢复）

| 档位 | 实现 | 特点 |
|---|---|---|
| **teaching** | 本仓 `learning.py` | 连续向量关联式学习，轻量 |
| **full** | `pasm.cognitive.learning.LearningEngine` | 离散动作标签 + 性格设计 + 反馈微调 |

两档**同接口、可互换**，`state()` / `apply_state()` 支持跨档状态迁移。

---

## 五、命令速查

```bash
# 跑教学闭环（默认 10 局）
python pasm_lite.py

# 作为引擎跑：3 局、预热 150 步
python engine.py 3 150

# 换非网格世界
python engine.py 3 200 toy-vector

# 跨档互换验证（19 项检查）
python verify_swap.py

# 引擎自检
python -c "import engine; print(engine.LiteEngine().snapshot())"
```

---

## 六、诚实边界（重要）

- Lite 是**教学骨架**：演示"机制跑通 + 记忆可见"，**不是**性能基线
- 强行为表现依赖奖励信号工程、探索策略、记忆阈值调优等细节
- 无指导学习、无预训练权重、无大规模调参
- **最佳练习**：对照第二节表格，把 Lite 逐步"喂"成完整引擎，每加一块看效果变化

---

## 七、生态与对接

| 仓库 | 定位 | 可见性 |
|---|---|---|
| **PASM** | 核心引擎（七层 + 认知皮层） | 私有 |
| **pasm-qclaw** | PASM Studio 桌面产品发行（Releases + 更新通道） | 公开（MIT） |
| **PASM-Lite**（本仓） | 零 token 认知最小教学实现 | 公开（MIT） |

- **提 issue**：架构方向、教学改进、零 token 认知猜想都欢迎
- **完整引擎 / 授权 / POC / 定制**（NPC、客服情绪层、陪伴养成等）：issue 留言对接
- **双系统（PASM + DeepSeek）**：完整引擎提供 OpenAI 兼容 API，可与任意 LLM 组合为 System1/System2

---

## English Summary

**PASM-Lite** is the open teaching edition of PASM — a **zero-token cognitive
agent** in which the entire decision loop runs in continuous latent vectors.
It keeps only the backbone of the seven-layer architecture: perception encoding
(VAE), working memory, episodic memory with event-driven writes, a GRU world
model, and a planner that "imagines" candidate action sequences before acting.

**What you get**

1. **A readable minimal implementation** — a single ~380-line file you can read
   in ten minutes and modify freely.
2. **A compliant engine** — `engine.py` implements the shared engine contract,
   so `create("pasm-lite")` yields a drop-in engine. Swap it for `create("pasm")`
   and **no calling code changes**.
3. **A reference contract** — `engine_api.py` is a same-source mirror of the PASM
   core contract (pure standard library, zero dependencies).
4. **Pluggable worlds** — non-grid environments work out of the box
   (`toy-vector`: obs_dim=3, n_actions=2); dimensional adaptation is automatic.
5. **A pluggable learning layer** — the `pasm.learning/1.0` contract makes the
   teaching tier and the full `LearningEngine` **interchangeable at runtime**.
6. **A math brain** — pure-numpy regression, statistics, correlation, linear
   systems, topological sort, shortest path, connected components and Markov
   steady state.
7. **Verification tooling** — `verify_swap.py` proves contract compliance and
   cross-tier interchangeability in one command.

Lite is honest about its scope: it is a **teaching skeleton, not a performance
baseline**. The point is visible mechanics and inspectable memory.

---

*PASM-Lite 全部功能总览 · 与代码同步维护。*
