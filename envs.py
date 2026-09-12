"""PASM-Lite 环境插件（Env Plugins）。

为什么需要这一层
----------------
`engine.py` 的 `LiteEngine(env=...)` 早就能接收任意环境对象，但"能接收"和
"能换"是两件事：没有注册表，调用方仍然得自己 import 具体环境类，
环境一变就要改代码；而且引擎内部写死了 `obs_dim=27 / n_actions=4`，
换个维度的世界直接崩。

本模块把环境做成**可发现、可创建、可自述**的插件：

    from envs import make_env, env_names, register_env   # 或从 engine_api 用同一套名字
    eng = LiteEngine(env_name="toy-vector")              # 换个世界，一行搞定
    register_env("我的世界", lambda **kw: MyEnv(**kw),
                 info={"obs_dim": 12, "n_actions": 5})

环境契约（与 `engine_api.ENV_REQUIRED_METHODS` 一致）
----------------------------------------------------
    必需：reset() -> obs ；step(action) -> (next_obs, reward, done)
    可选：observe() / close() / spec() / obs_dim / n_actions / eaten / crashes

其中 `obs_dim` 与 `n_actions` 会被 `LiteEngine` 读取，用来在换环境时
自动调整感知/世界模型/规划器的维度 —— 这正是"可换非网格环境"的关键。
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict

try:                                  # 与引擎共用同一套注册表（契约层提供）
    from engine_api import (ENV_REGISTRY, env_conforms, env_names,
                            env_registry, make_env, register_env)
except Exception:                     # pragma: no cover - 兼容旧版镜像
    env_names, env_registry, make_env, register_env = (None,) * 4   # type: ignore
    ENV_REGISTRY = None                                            # type: ignore
    env_conforms = None                                            # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: 网格世界的维度（与 pasm_lite.GridWorld.observe 保持一致）
GRID_OBS_DIM = 27
GRID_N_ACTIONS = 4


# ================================================== 内置环境 1：网格世界
def _make_grid(**cfg):
    """网格世界（教学版默认环境，10×10 收集能量块）。"""
    from pasm_lite import GridWorld
    return GridWorld(**cfg)


# ================================================== 内置环境 2：非网格环境
class VectorToy:
    """一个**非网格**的最小环境：把一维状态维持在原点附近。

    存在意义：证明引擎真的能换世界 —— 它的观测维度是 3、动作空间是 2，
    与网格世界（27 / 4）完全不同；如果引擎里还写死着 27 和 4，这里会立刻炸。

        obs = [x, v/2, t/T]      3 维连续观测
        act = 0 向后推 / 1 向前推
        r   = -|x|               越靠近 0 分越高
    """

    obs_dim = 3
    n_actions = 2

    def __init__(self, horizon: int = 100, seed: int = 0):
        self.horizon = int(horizon)
        self.rng = random.Random(seed)
        self.x = 0.0
        self.v = 0.0
        self.t = 0
        self.eaten = 0
        self.crashes = 0

    def reset(self):
        self.x = self.rng.uniform(-1.0, 1.0)
        self.v = 0.0
        self.t = 0
        self.eaten = self.crashes = 0
        return self.observe()

    def observe(self):
        import torch
        return torch.tensor([self.x, self.v / 2.0,
                             self.t / max(1, self.horizon)])

    def step(self, a):
        self.t += 1
        self.v = 0.9 * self.v + 0.1 * (1.0 if int(a) == 1 else -1.0)
        self.x = max(-2.0, min(2.0, self.x + self.v))
        if abs(self.x) < 0.1:
            self.eaten += 1
        self.crashes = self.t - self.eaten
        r = -abs(self.x)
        return self.observe(), r, self.t >= self.horizon

    def spec(self) -> Dict[str, Any]:
        return {"name": "toy-vector", "obs_dim": self.obs_dim,
                "n_actions": self.n_actions, "horizon": self.horizon,
                "kind": "non-grid"}

    def close(self):
        pass


def _make_toy(**cfg):
    return VectorToy(**cfg)


# ================================================== 注册（导入即生效）
def register_builtin_envs(force: bool = False) -> None:
    """登记本仓库自带的环境。重复调用安全。"""
    if register_env is None:                      # 契约层不可用：静默跳过
        return

    existing = set(env_names() if env_names else [])
    if force or "grid-10x10" not in existing:
        register_env("grid-10x10", _make_grid, info={
            "obs_dim": GRID_OBS_DIM, "n_actions": GRID_N_ACTIONS,
            "description": "10×10 网格世界：收集能量块（教学默认环境）",
            "kind": "grid", "source": "pasm_lite.GridWorld"},
            replace=True, default=True)
    if force or "toy-vector" not in existing:
        register_env("toy-vector", _make_toy, info={
            "obs_dim": VectorToy.obs_dim, "n_actions": VectorToy.n_actions,
            "description": "非网格最小环境：把一维状态维持在原点附近",
            "kind": "non-grid", "source": "envs.VectorToy"},
            replace=True)


register_builtin_envs()


def selftest() -> bool:
    """自检：注册表可用、两个内置环境都符合契约、维度申报正确。"""
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("  x %s" % msg)
        else:
            print("  v %s" % msg)

    print("envs selftest")
    if ENV_REGISTRY is None:
        print("  x 契约层不可用（engine_api 未导入）")
        return False

    names = env_names()
    check("grid-10x10" in names, "网格世界已注册：%s" % names)
    check("toy-vector" in names, "非网格环境已注册")

    for nm in ("grid-10x10", "toy-vector"):
        env = make_env(nm)
        good, probs = env_conforms(env)
        check(good, "%s 符合环境契约（问题：%s）" % (nm, probs or "无"))
        obs = env.reset()
        nxt, r, done = env.step(0)
        check(hasattr(obs, "__len__") and hasattr(nxt, "__len__"),
              "%s reset/step 返回观测" % nm)
        dim = getattr(env, "obs_dim", None)
        check(dim is None or len(obs) == dim,
              "%s 申报 obs_dim=%s 与实际一致" % (nm, dim))

    spec = env_registry()
    check(isinstance(spec, dict) and len(spec) >= 2, "env_registry() 自述清单可用")
    print("envs selftest:", "通过" if ok else "失败")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
