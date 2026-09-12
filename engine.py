"""PASM-Lite 接口引擎（Engine API 实现）。

把教学版 `pasm_lite.py` 里那个 Agent 包装成本仓库**对外的统一引擎接口**：
调用方只依赖 `engine_api`（已随本仓库一同提供），不需要了解教学版的
`(z, c, m, key)` 内部管线；将来换成生产引擎（pasm / pasm-light）时，
调用代码一行都不用改。

    from engine_api import create
    eng = create("pasm-lite")            # 或直接 LiteEngine()
    eng.reset_episode()
    action, report = eng.act(obs)
    eng.learn(obs, action, next_obs, reward)

设计要点
--------
· **契约对齐**：本类满足 `engine_api.REQUIRED_METHODS` 全部 6 项，
  并用 `conforms()` 自我校验（`python engine.py` 会打印结果）。
· **管道隐藏**：教学版 `act()` 返回 5 元组、`learn()` 需要 8 个参数，
  引擎在这里把中间量暂存起来，对外只保留 §契约里的标准签名。
· **诚实申报**：`capabilities()` 如实标注教学版只覆盖七层中的主干四层。
· **可发现**：导入即注册进 `engine_api.REGISTRY`，`create("pasm-lite")` 可用。

v0.2.0 新增
-----------
· **环境插件化**：环境改从 `engine_api` 的环境注册表里取（`envs.py` 登记），
  支持 `LiteEngine(env_name="toy-vector")` 换**非网格**世界；引擎会读环境
  申报的 `obs_dim / n_actions` 自动调整感知、世界模型与规划器的维度。
· **`freeze_vae()` 语义**：冻结感知编码器（自编码器），让潜空间定型，
  之后只训练世界模型与学习层 —— 与完整引擎的同名方法语义对齐。

v0.2.1 新增
-----------
· **学习层热插拔**：`engine.learning` 暴露当前学习层，`attach_learning(layer)`
  可换成任意满足 `pasm.learning/1.0` 契约的实现（契约校验 + 潜维/动作数对齐，
  不过则拒绝且不动原实现）。内部用 `_LearningAdapter` 抹平两档差异，
  于是核心档 `LearningEngine`、教学档 `LearningLayer`、第三方实现三者可互换 ——
  这正是"同一接口两档实现"在引擎侧的兑现点。
· **持久化走契约**：`save()` 存 `learning.state()`、`load()` 调 `apply_state()`，
  换档后存档照样能读（旧快照保留兼容分支）。

依赖：torch（CPU 即可）。运行：`python engine.py [轮数] [预热步数] [环境名]`
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                 # 保证同目录的 engine_api / pasm_lite / envs 可导入
    sys.path.insert(0, _HERE)

import torch

from engine_api import (API_VERSION, ENV_REGISTRY, Capabilities, EngineInfo,
                        REGISTRY, conforms, env_conforms, make_env,
                        normalize_snapshot)
from envs import register_builtin_envs
from learning import LEARNING_API, LearningLayer, learning_conforms
from pasm_lite import SEED, Agent

ENGINE_NAME = "pasm-lite"
ENGINE_VERSION = "0.2.1"

#: 环境不可用时的兜底维度（与教学版默认网格世界一致）
_DEFAULT_OBS_DIM = 27
_DEFAULT_N_ACT = 4

INFO = EngineInfo(
    name=ENGINE_NAME,
    version=ENGINE_VERSION,
    kind="lite",
    description="零 token 认知循环教学引擎（自编码感知 / 工作记忆 / 情景记忆 / GRU 世界模型 / 采样规划）",
    deps=("torch",),
)


class _LearningAdapter:
    """把任意契约学习层适配成教学版内部期望的形态（不改原对象）。

    教学版内部只用到三件事：`bias(z) -> tensor`、`update(z,a,r)`、`len()`；
    而契约层对外统一的是 `learn()` 与可能返回 list 的 `bias()`。
    这层薄转发让**核心档 LearningEngine 也能直接插进教学引擎**，
    是"同一接口两档实现"在引擎侧的兑现点。
    """

    def __init__(self, layer, n_act: int = None):
        self._layer = layer
        self._n_act = int(n_act) if n_act else None
        # 离散标签档（如核心 LearningEngine）需要把整数动作映射回标签；
        # 向量档（教学档）order 不存在，保持整数动作原样。
        order = getattr(layer, "order", None)
        self._order = list(order) if order else None

    def __getattr__(self, name):
        if name.startswith("_"):             # 防 `_layer` 尚未赋值时无限递归
            raise AttributeError(name)
        return getattr(self._layer, name)

    def bias(self, z=None, n_act=None):
        v = self._layer.bias(z) if z is not None else self._layer.bias(None)
        t = v if torch.is_tensor(v) else torch.tensor(list(v), dtype=torch.float32)
        want = int(n_act or self._n_act or 0)
        if want and t.numel() != want:       # 动作数对齐（截断 / 补零）
            t = t[:want] if t.numel() > want else torch.cat(
                [t, torch.zeros(want - t.numel())])
        return t

    def _action_name(self, a):
        """整数动作 → 学习层认得的动作表示（标签档映射回标签）。"""
        if self._order:
            return self._order[int(a) % len(self._order)]
        return int(a)

    def update(self, z, a, r):
        return self._layer.learn(z, self._action_name(a), float(r))

    def __len__(self):
        try:
            return len(self._layer)
        except TypeError:
            return 0


def _caps(extra: dict) -> Capabilities:
    return Capabilities(
        perception=True,          # 层0 自编码器
        working_memory=True,      # 层1 最近 K 步
        memory=True,              # 层2 情景记忆 KNN 检索
        world_model=True,         # 层3 GRU 预测 + 采样规划
        emotion=False,            # 教学范围外
        personality=False,
        development=False,
        metacognition=False,
        global_workspace=False,
        symbolic=False,
        llm=False,
        multimodal=False,
        persistence=True,         # save/load 已实现
        extra=extra,
    )


#: 模块级默认能力表（注册表用；实例会按实际环境覆盖 extra）
CAPS = _caps({"note": "教学最小实现：七层中保留主干四层"})


class LiteEngine:
    """教学版引擎：对外统一接口，对内包住 `pasm_lite.Agent`。"""

    def __init__(self, seed: int = SEED, env=None, env_name: str = None,
                 warmup: int = 600, latent: int = 8,
                 n_act: int = None, obs_dim: int = None, hidden: int = 24):
        self.seed = int(seed)

        # ---- 环境：实例优先，其次按注册表名字创建，最后退回默认环境 ----
        self.env_name = env_name
        if env is not None:
            self.env = env
            self.env_name = env_name or type(env).__name__
        else:
            register_builtin_envs()           # 幂等
            self.env = make_env(env_name)     # env_name=None → 注册表默认环境
            self.env_name = env_name or (ENV_REGISTRY.default or "?")

        good, problems = env_conforms(self.env)
        if not good:
            raise ValueError("环境不符合契约（%s）：%s"
                             % (self.env_name, "; ".join(problems)))

        # ---- 维度：环境申报优先，参数可覆盖（这是"可换非网格环境"的关键）----
        self.obs_dim = int(obs_dim or getattr(self.env, "obs_dim", _DEFAULT_OBS_DIM))
        self.n_act = int(n_act or getattr(self.env, "n_actions", _DEFAULT_N_ACT))
        self.latent = int(latent)
        self.hidden = int(hidden)

        self.agent = Agent(obs_dim=self.obs_dim, latent=self.latent,
                           n_act=self.n_act, hidden=self.hidden)
        self.warmup_steps = int(warmup)
        self._last = None            # act() 暂存的内部管线 (obs, z, c, m, key, action)
        self._learn_flag = True      # 本步是否学习（来自 act(learning=...)）
        self._steps = 0
        self._episodes = 0
        self._last_reward = 0.0
        self._closed = False
        self._vae_frozen = False

        self._info = EngineInfo(name=ENGINE_NAME, version=ENGINE_VERSION,
                                kind="lite", description=INFO.description,
                                deps=("torch",))
        self._caps = _caps({"env": self.env_name, "obs_dim": self.obs_dim,
                            "n_actions": self.n_act,
                            "note": "教学最小实现：七层中保留主干四层"})

    # ==================== 自描述 ====================
    def info(self) -> EngineInfo:
        return self._info

    def capabilities(self) -> Capabilities:
        return self._caps

    # ==================== 生命周期 ====================
    def reset_episode(self):
        """开新一局：重置环境、工作记忆与 GRU 隐状态，返回初始观测。"""
        obs = self.env.reset()
        self.agent.wmem.reset()
        self.agent.h = torch.zeros(self.hidden)
        self._last = None
        self._episodes += 1
        self._last_reward = 0.0
        return obs

    def obs(self):
        """当前观测（调用方不想自己管环境时用）。"""
        return self.env.observe()

    # ==================== 决策 / 学习 ====================
    def act(self, obs=None, learning: bool = True):
        """决策：返回 (action, report)。

        `learning=False` 表示本步只决策不学习，随后的 `learn()` 会被跳过
        （完整引擎用它做"冻结推理"，教学版在此对齐语义）。
        """
        obs = self.env.observe() if obs is None else obs
        self._learn_flag = bool(learning)
        a, z, c, m, key = self.agent.act(obs)
        self._last = (obs, z, c, m, key, int(a))
        report = {
            "z": [round(v, 4) for v in z.detach().tolist()],
            "skills": len(self.agent.learner),
            "episodic": len(self.agent.mem),
        }
        return int(a), report

    def learn(self, obs=None, action=None, next_obs=None, reward: float = 0.0,
              report=None) -> dict:
        """学习：用上一步 `act()` 暂存的内部量更新世界模型与关联式学习层。"""
        if self._last is None:
            return {"learned": False, "reason": "本步没有待学习的 act()"}
        o, z, c, m, key, a = self._last
        self._last = None
        self._steps += 1
        if not self._learn_flag:
            return {"learned": False, "reason": "act(learning=False)"}
        nxt = next_obs if next_obs is not None else (obs if obs is not None else o)
        act = int(a if action is None else action)
        self.agent.learn(o, z, c, m, key, act, nxt, float(reward))
        self._last_reward = float(reward)
        return {"learned": True,
                "skills": len(self.agent.learner),
                "episodic": len(self.agent.mem)}

    # ==================== 学习层热插拔 ====================
    @property
    def learning(self):
        """当前学习层（契约对象：info / capabilities / learn / bias / state）。"""
        return self.agent.learner

    @property
    def learning_tier(self) -> str:
        """当前学习层档位（full / teaching / 第三方自定义）。"""
        return getattr(self.agent.learner, "TIER", "?")

    def attach_learning(self, layer, strict: bool = True):
        """替换学习层 —— 「学习层可替换」的落地点。

        任何满足 `pasm.learning/1.0` 契约的对象都能顶上：默认是教学档
        `learning.LearningLayer`，也可以是外部实现（核心档 / 第三方）。
        校验三件事，任一不过就拒绝，且**保持原实现不动**：

          ① 契约一致 —— `learning_conforms(layer)`
          ② 潜维对齐 —— `latent` 相同，否则世界模型给的 z 与偏好对不上
          ③ 动作数对齐 —— `n_act` 相同，否则 one-hot 越界

        返回 (是否换成功, 说明)。
        """
        ok, probs = learning_conforms(layer, strict=strict)
        if not ok:
            return False, "学习层不符合 %s 契约：%s" % (LEARNING_API, "; ".join(probs))
        for attr, want in (("latent", self.latent), ("n_act", self.n_act)):
            got = getattr(layer, attr, None)
            if got is not None and int(got) != int(want):
                return False, "学习层 %s=%s 与引擎 %s=%s 不匹配" % (attr, got, attr, want)
        self.agent.learner = _LearningAdapter(layer, n_act=self.n_act)
        return True, "学习层已替换为 %s[%s]" % (type(layer).__name__, self.learning_tier)

    # ==================== 感知编码器冻结 ====================
    def freeze_vae(self):
        """冻结感知编码器（自编码器），让潜在空间定型。

        为什么需要它：感知编码器在持续训练时，潜在空间会一直漂移，
        已经学到的世界模型/情景记忆会"失去参考系"（昨天学到的 z 和今天
        不再可比）。冻结之后，编码器只作确定性的观测→向量映射，
        后续训练只更新世界模型与学习层 —— 这也是完整引擎里 `freeze_vae()`
        的语义，教学版在此对齐。
        """
        self.agent.freeze_encoder()
        self._vae_frozen = True
        return self

    def unfreeze_vae(self):
        """解冻感知编码器（需要继续适应新环境/新观测分布时用）。"""
        self.agent.unfreeze_encoder()
        self._vae_frozen = False
        return self

    @property
    def vae_frozen(self) -> bool:
        return bool(self.agent.encoder_frozen)

    # ==================== 状态 ====================
    def snapshot(self) -> dict:
        """状态快照：七个规范区块齐全（教学版没有的区块如实填 None）。"""
        ag = self.agent
        return normalize_snapshot({
            "step": self._steps,
            "personality": None,          # 教学版无性格模块（见 capabilities）
            "emotion": None,
            "development": {"episodes": self._episodes,
                            "note": "教学版无发育模块，此处仅记轮次"},
            "memory": {"episodic": len(ag.mem),
                       "skills": len(ag.learner),
                       "working_slots": len(ag.wmem.buf)},
            "global_workspace": None,
            "last_retrieval": {"last_reward": round(self._last_reward, 4)},
            "perception": {"frozen": self.vae_frozen,
                           "obs_dim": self.obs_dim,
                           "latent": self.latent},
            "engine": self._info.to_dict(),
            "env": {"name": self.env_name, "n_actions": self.n_act},
        })

    # ==================== 持久化 ====================
    def save(self, path: str) -> str:
        torch.save({
            "api": API_VERSION, "engine": ENGINE_NAME,
            "seed": self.seed,
            "ae": self.agent.ae.state_dict(),
            "wm": self.agent.wm.state_dict(),
            "learning": self.agent.learner.state(),   # 契约统一：两档学习层都能存
            "learning_tier": self.learning_tier,      # 记档位，防止读到别的档上
            "steps": self._steps, "episodes": self._episodes,
            "env_name": self.env_name,
            "obs_dim": self.obs_dim, "n_act": self.n_act,
            "latent": self.latent, "hidden": self.hidden,
            "vae_frozen": self.vae_frozen,
        }, path)
        return path

    def load(self, path: str):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.agent.ae.load_state_dict(blob["ae"])
        self.agent.wm.load_state_dict(blob["wm"])
        st = blob.get("learning")
        if st is not None:                   # 新格式：交给学习层自己恢复（可换档）
            was = blob.get("learning_tier")
            if was and was != self.learning_tier:
                raise ValueError(
                    "存档里的学习层档位=%s，当前=%s；请先 attach_learning() 换上同档再 load()"
                    % (was, self.learning_tier))
            self.agent.learner.apply_state(st)
        else:                                # 旧快照兼容（v0.2.0 之前）
            self.agent.learner.W = blob["learner_W"]
            self.agent.learner.keys = list(blob.get("learner_keys", []))
            self.agent.learner.vals = list(blob.get("learner_vals", []))
        self._steps = int(blob.get("steps", 0))
        self._episodes = int(blob.get("episodes", 0))
        if blob.get("vae_frozen"):
            self.freeze_vae()
        return self

    def close(self):
        self._closed = True
        closer = getattr(self.env, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:                         # noqa: BLE001
                pass

    # ==================== 便捷方法（非契约要求） ====================
    def _env_step(self, a):
        """统一的 step 适配：教学版是 3 元组，完整引擎是 4 元组，都能吃。"""
        out = self.env.step(a)
        if isinstance(out, (tuple, list)):
            if len(out) >= 4:
                return out[0], out[1], out[2]
            if len(out) == 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], False
        return out, 0.0, False

    def warmup(self, steps: int = None):
        """世界模型预热：随机逛若干步，先把"动作→下一状态→奖励"拟合个大概。

        不预热的话规划器一开始等于掷骰子（教学版极简实现，无在线自适应）。
        编码器被 `freeze_vae()` 冻结时，本步只更新世界模型。
        """
        steps = self.warmup_steps if steps is None else int(steps)
        ag = self.agent
        obs = self.env.reset()
        for _ in range(steps):
            a = int(torch.randint(0, ag.n_act, (1,)).item())
            nxt, r, done = self._env_step(a)
            z = ag.ae.encode(obs)
            a_oh = torch.nn.functional.one_hot(torch.tensor(a), ag.n_act).float()
            h, zp, rp = ag.wm(z, a_oh, torch.zeros(ag.latent), ag.h)
            loss = (ag.ae.encode(nxt) - zp).pow(2).mean() \
                + 0.2 * (r - float(rp.detach())) ** 2
            ag.opt.zero_grad()
            loss.backward()
            ag.opt.step()
            ag.h = h.detach()
            obs = nxt if not done else self.env.reset()
        return self

    def run_episode(self, steps: int = 200) -> dict:
        """跑完一局，返回统计（供脚本/看板直接读）。"""
        self.reset_episode()
        total = 0.0
        n = 0
        for _ in range(steps):
            a, _ = self.act()
            nxt, r, done = self._env_step(a)
            self.learn(reward=r, next_obs=nxt)
            total += r
            n += 1
            if done:
                break
        return {"reward": round(total, 3), "steps": n,
                "eaten": getattr(self.env, "eaten", 0),
                "crashes": getattr(self.env, "crashes", 0)}


# 导入即注册：使 create("pasm-lite") 可用
REGISTRY.register(ENGINE_NAME, lambda **cfg: LiteEngine(**cfg), info=INFO, replace=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    episodes = int(argv[0]) if argv else 25
    warmup = int(argv[1]) if len(argv) > 1 else 600
    env_name = argv[2] if len(argv) > 2 else None

    eng = LiteEngine(warmup=warmup, env_name=env_name)
    ok, problems = conforms(eng, strict=True)
    print("引擎接口一致性:", "通过" if ok else "不通过 -> %s" % problems)
    print("引擎自述:", json.dumps(eng.info().to_dict(), ensure_ascii=False))
    print("已具备能力:", eng.capabilities().layers())
    print("可选环境:", ENV_REGISTRY.names(), "| 当前:",
          eng.env_name, "| obs_dim=%d n_actions=%d" % (eng.obs_dim, eng.n_act))

    print("世界模型预热 %d 步 …" % warmup)
    eng.warmup()
    eng.freeze_vae()
    print("感知编码器已冻结（freeze_vae）: ", eng.vae_frozen)

    hist = []
    for ep in range(episodes):
        st = eng.run_episode()
        hist.append(st["reward"])
        if ep % 10 == 0 or ep == episodes - 1:
            print("episode %d: reward %+.2f | 能量 %s | 情景记忆 %d 条 | 学习层技能 %d"
                  % (ep, st["reward"], st["eaten"],
                     eng.snapshot()["memory"]["episodic"],
                     len(eng.agent.learner)))
    half = max(1, episodes // 5)
    print("\n平均奖励(末%d集): %+.2f（前%d集 %+.2f）"
          % (half, sum(hist[-half:]) / half, half, sum(hist[:half]) / half))
    print("快照:", json.dumps(eng.snapshot()["memory"], ensure_ascii=False))
    print("演示要点：同一个引擎接口既能跑教学版，也能换生产引擎/换环境——调用方代码不变。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
