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

依赖：torch（CPU 即可）。运行：`python engine.py [轮数] [预热步数]`
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                 # 保证同目录的 engine_api / pasm_lite 可导入
    sys.path.insert(0, _HERE)

import torch

from engine_api import (API_VERSION, Capabilities, EngineInfo, REGISTRY,
                        conforms, normalize_snapshot)
from pasm_lite import SEED, Agent, GridWorld

ENGINE_NAME = "pasm-lite"
ENGINE_VERSION = "0.1.0"

INFO = EngineInfo(
    name=ENGINE_NAME,
    version=ENGINE_VERSION,
    kind="lite",
    description="零 token 认知循环教学引擎（自编码感知 / 工作记忆 / 情景记忆 / GRU 世界模型 / 采样规划）",
    deps=("torch",),
)

CAPS = Capabilities(
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
    extra={"env": "GridWorld(10×10)", "note": "教学最小实现：七层中保留主干四层"},
)


class LiteEngine:
    """教学版引擎：对外统一接口，对内包住 `pasm_lite.Agent`。"""

    def __init__(self, seed: int = SEED, env=None, warmup: int = 600,
                 latent: int = 8, n_act: int = 4):
        self.seed = int(seed)
        self.env = env if env is not None else GridWorld()
        self.agent = Agent()
        self.warmup_steps = int(warmup)
        self._last = None            # act() 暂存的内部管线 (obs, z, c, m, key, action)
        self._learn_flag = True      # 本步是否学习（来自 act(learning=...)）
        self._steps = 0
        self._episodes = 0
        self._last_reward = 0.0
        self._closed = False

    # ==================== 自描述 ====================
    def info(self) -> EngineInfo:
        return INFO

    def capabilities(self) -> Capabilities:
        return CAPS

    # ==================== 生命周期 ====================
    def reset_episode(self):
        """开新一局：重置环境、工作记忆与 GRU 隐状态，返回初始观测。"""
        obs = self.env.reset()
        self.agent.wmem.reset()
        self.agent.h = torch.zeros(24)
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
            "engine": INFO.to_dict(),
        })

    # ==================== 持久化 ====================
    def save(self, path: str) -> str:
        torch.save({
            "api": API_VERSION, "engine": ENGINE_NAME,
            "seed": self.seed,
            "ae": self.agent.ae.state_dict(),
            "wm": self.agent.wm.state_dict(),
            "learner_W": self.agent.learner.W,
            "learner_keys": self.agent.learner.keys,
            "learner_vals": self.agent.learner.vals,
            "steps": self._steps, "episodes": self._episodes,
        }, path)
        return path

    def load(self, path: str):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.agent.ae.load_state_dict(blob["ae"])
        self.agent.wm.load_state_dict(blob["wm"])
        self.agent.learner.W = blob["learner_W"]
        self.agent.learner.keys = list(blob.get("learner_keys", []))
        self.agent.learner.vals = list(blob.get("learner_vals", []))
        self._steps = int(blob.get("steps", 0))
        self._episodes = int(blob.get("episodes", 0))
        return self

    def close(self):
        self._closed = True

    # ==================== 便捷方法（非契约要求） ====================
    def warmup(self, steps: int = None):
        """世界模型预热：随机逛若干步，先把"动作→下一状态→奖励"拟合个大概。

        不预热的话规划器一开始等于掷骰子（教学版极简实现，无在线自适应）。
        """
        steps = self.warmup_steps if steps is None else int(steps)
        obs = self.env.reset()
        for _ in range(steps):
            a = int(torch.randint(0, 4, (1,)).item())
            nxt, r, done = self.env.step(a)
            z = self.agent.ae.encode(obs)
            a_oh = torch.nn.functional.one_hot(torch.tensor(a), 4).float()
            h, zp, rp = self.agent.wm(z, a_oh, torch.zeros(8), self.agent.h)
            loss = (self.agent.ae.encode(nxt) - zp).pow(2).mean() \
                + 0.2 * (r - float(rp.detach())) ** 2
            self.agent.opt.zero_grad()
            loss.backward()
            self.agent.opt.step()
            self.agent.h = h.detach()
            obs = nxt if not done else self.env.reset()
        return self

    def run_episode(self, steps: int = 200) -> dict:
        """跑完一局，返回统计（供脚本/看板直接读）。"""
        self.reset_episode()
        total = 0.0
        n = 0
        for _ in range(steps):
            a, _ = self.act()
            nxt, r, done = self.env.step(a)
            self.learn(reward=r, next_obs=nxt)
            total += r
            n += 1
            if done:
                break
        return {"reward": round(total, 3), "steps": n,
                "eaten": self.env.eaten, "crashes": self.env.crashes}


# 导入即注册：使 create("pasm-lite") 可用
REGISTRY.register(ENGINE_NAME, lambda **cfg: LiteEngine(**cfg), info=INFO, replace=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    episodes = int(argv[0]) if argv else 25
    warmup = int(argv[1]) if len(argv) > 1 else 600

    eng = LiteEngine(warmup=warmup)
    ok, problems = conforms(eng, strict=True)
    print("引擎接口一致性:", "通过" if ok else "不通过 -> %s" % problems)
    print("引擎自述:", json.dumps(eng.info().to_dict(), ensure_ascii=False))
    print("已具备能力:", eng.capabilities().layers())
    print("世界模型预热 %d 步 …" % warmup)
    eng.warmup()

    hist = []
    for ep in range(episodes):
        st = eng.run_episode()
        hist.append(st["reward"])
        if ep % 10 == 0 or ep == episodes - 1:
            print(f"episode {ep}: reward {st['reward']:+.2f} | 能量 {st['eaten']} | "
                  f"情景记忆 {eng.snapshot()['memory']['episodic']} 条 | "
                  f"学习层技能 {len(eng.agent.learner)}")
    half = max(1, episodes // 5)
    print(f"\n平均奖励(末{half}集): {sum(hist[-half:])/half:+.2f}"
          f"（前{half}集 {sum(hist[:half])/half:+.2f}）")
    print("快照:", json.dumps(eng.snapshot()["memory"], ensure_ascii=False))
    print("演示要点：同一个引擎接口既能跑教学版，也能换生产引擎——调用方代码不变。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
