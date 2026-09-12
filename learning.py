"""pasm-lite 学习层（教学档，与 pasm.cognitive.learning 同源思想）

教学版把"学习"做到最轻：用连续向量状态 z 与一个「动作偏好」矩阵做
关联式(Hebbian-like)学习 —— 在经常获得正奖励的状态附近，强化那个动作。
同时维护一个「技能记忆」：把「状态模式 → 成功动作」存下来，遇到相似状态直接调用。

── 学习层契约（LEARNING_API = "pasm.learning/1.0"，与核心档同一接口）──────
本文件是契约的**教学档实现**（tier="teaching", kind="vector"）：

  同一接口（LearningLayer 协议）   info / capabilities / learn / bias / state
  两档实现
   ├ full      pasm.cognitive.learning.LearningEngine   离散动作 + 性格设计
   └ teaching  LearningLayer（本文件）                    连续向量关联式

差别只在内部表示：核心档是「动作 → 权重」字典，本档是「状态 × 动作」权重矩阵。
调用方按同一组方法使用，`learning_conforms()` 一键校验两档是否可互换。

纯 torch / 标准库，无新依赖。运行：python learning.py 看自测。
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

# ============================================================ 学习层契约
LEARNING_API = "pasm.learning/1.0"
LEARNING_API_VERSION = "1.0"

TIER_TEACHING = "teaching"   # 教学档：连续向量关联式
KIND_VECTOR = "vector"

#: 契约必需方法（与核心档 LEARNING_REQUIRED_METHODS 保持一致）
LEARNING_REQUIRED_METHODS = ("info", "capabilities", "learn", "bias", "state")
LEARNING_OPTIONAL_METHODS = ("update", "recall_action", "reset", "apply_state", "spec")


@dataclass
class LearningInfo:
    """学习层自描述（与 pasm.cognitive.learning.LearningInfo 同构）。"""

    api: str = LEARNING_API
    version: str = LEARNING_API_VERSION
    name: str = "LearningLayer"
    tier: str = TIER_TEACHING
    kind: str = KIND_VECTOR
    summary: str = ""

    def to_dict(self) -> dict:
        return {"api": self.api, "version": self.version, "name": self.name,
                "tier": self.tier, "kind": self.kind, "summary": self.summary}


def _as_mapping(obj):
    """把 info()/state() 的返回值归一成 dict。"""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    fn = getattr(obj, "to_dict", None)
    if callable(fn):
        try:
            d = fn()
            return dict(d) if isinstance(d, Mapping) else None
        except Exception:
            return None
    return None


def learning_conforms(obj, strict: bool = False):
    """学习层结构一致性检查 → (是否通过, 问题清单)。

    （与 pasm.cognitive.learning.learning_conforms 同源，教学版保持轻量自包含。）
    """
    problems = []
    for m in LEARNING_REQUIRED_METHODS:
        if not callable(getattr(obj, m, None)):
            problems.append("缺少必需方法 %s()" % m)
    for m in LEARNING_OPTIONAL_METHODS:
        if hasattr(obj, m) and not callable(getattr(obj, m)):
            problems.append("可选方法 %s 存在但不可调用" % m)
    if problems:
        return False, problems

    try:
        info = _as_mapping(obj.info())
        if info is None:
            problems.append("info() 应返回 Mapping 或带 to_dict() 的对象")
        elif not info.get("api"):
            problems.append("info() 缺少 api 字段")
        if _as_mapping(obj.capabilities()) is None:
            problems.append("capabilities() 应返回 Mapping")
    except Exception as ex:
        problems.append("info()/capabilities() 调用出错：%s" % ex)

    if strict and not problems:
        try:
            if obj.bias(None) is None:
                problems.append("bias() 不应返回 None")
        except Exception as ex:
            problems.append("bias() 调用出错：%s" % ex)
        try:
            if _as_mapping(obj.state()) is None:
                problems.append("state() 应返回 Mapping")
        except Exception as ex:
            problems.append("state() 调用出错：%s" % ex)

    return (not problems), problems


# ============================================================ 教学档实现
class LearningLayer:
    """轻量关联式学习层：状态→动作偏好 + 技能记忆召回。

    满足 `LearningLayer` 契约 —— teaching 档：连续向量关联式学习。
    """

    TIER = TIER_TEACHING
    KIND = KIND_VECTOR

    def __init__(self, latent: int = 8, n_act: int = 4, lr: float = 0.05, mem_cap: int = 256):
        self.latent = latent
        self.n_act = n_act
        self.lr = lr
        # W: 线性关联权重，z @ W 得到各动作偏好偏置
        self.W = torch.zeros(latent, n_act)
        self.keys: list = []          # 技能记忆：状态关键向量
        self.vals: list = []          # 对应成功动作
        self.mem_cap = mem_cap

    # ---------------------------------------------------- 契约：自描述
    def info(self) -> LearningInfo:
        return LearningInfo(
            summary="潜维 %d / 动作 %d / 技能记忆 %d 条"
                    % (self.latent, self.n_act, len(self.keys)))

    def capabilities(self) -> dict:
        """能力表：调用方可据此决定"这块活能不能交给它"。"""
        return {
            "api": LEARNING_API,
            "tier": self.TIER,
            "design": False,            # 不做性格设计（核心档才有）
            "explore": False,           # 探索交给上层策略
            "feedback": True,           # 奖励驱动的关联式学习
            "redesign": False,
            "stateful_vector": True,    # 偏好依赖连续状态向量
            "skills_memory": True,      # 含"状态→动作"技能记忆
            "persistent": False,        # 由调用方决定是否落盘
        }

    def spec(self) -> str:
        """一行人类可读摘要（调试 / 日志用）。"""
        return "%s[%s] 潜维=%d 动作=%d 技能记忆=%d" % (
            type(self).__name__, self.TIER, self.latent, self.n_act, len(self.keys))

    # ---------------------------------------------------- 契约：统一经验入口
    def learn(self, z=None, action=None, reward=None, **kwargs):
        """吸收一次经验（契约方法）——与核心档 `LearningEngine.learn` **同名同义**：

            learn(z, action, reward)                —— 在情境 z 下做 action 得到 reward
            learn(z=z, action=a, reward=r)
            （兼容别名：obs / a / r）
        """
        if z is None:
            z = kwargs.get("z", kwargs.get("obs"))
        if action is None:
            action = kwargs.get("action", kwargs.get("a"))
        if reward is None:
            reward = kwargs.get("reward", kwargs.get("r"))
        if z is None or action is None or reward is None:
            raise ValueError("learn() 需要 (z, action, reward)")
        return self.update(z, action, reward)

    def state(self) -> dict:
        """可持久化状态（契约方法）——tensor 全部转成 list，可直接 json.dump。"""
        return {"api": LEARNING_API, "tier": self.TIER,
                "latent": self.latent, "n_act": self.n_act,
                "lr": self.lr, "mem_cap": self.mem_cap,
                "W": self.W.tolist(),
                "keys": [k.tolist() for k in self.keys],
                "vals": list(self.vals)}

    def apply_state(self, d) -> "LearningLayer":
        """从 state() 的产物恢复（链式）。"""
        d = _as_mapping(d) or {}
        self.latent = int(d.get("latent", self.latent))
        self.n_act = int(d.get("n_act", self.n_act))
        self.lr = float(d.get("lr", self.lr))
        self.mem_cap = int(d.get("mem_cap", self.mem_cap))
        W = d.get("W")
        if W:
            self.W = torch.tensor(W, dtype=torch.float32)
        keys = d.get("keys") or []
        self.keys = [torch.tensor(k, dtype=torch.float32) for k in keys]
        self.vals = [int(v) for v in (d.get("vals") or [])]
        return self

    def reset(self, keep_design: bool = True) -> "LearningLayer":
        """清空习得的关联权重与技能记忆（保持接口与核心档一致）。"""
        self.W = torch.zeros(self.latent, self.n_act)
        self.keys, self.vals = [], []
        return self

    # ---------------------------------------------------- 学习 / 使用
    def update(self, z, a, r):
        """关联式学习：正奖励强化 (z,a)，负奖励削弱。"""
        z = z.detach()
        if r == 0:
            return
        sign = 1.0 if r > 0 else -1.0
        delta = sign * self.lr * torch.outer(z, F.one_hot(torch.tensor(int(a)), self.n_act).float())
        self.W += delta
        if r > 0.5:                  # 显著正奖励才记入技能记忆
            self._remember(z, int(a))

    def _remember(self, z, a):
        if len(self.keys) >= self.mem_cap:
            self.keys.pop(0)
            self.vals.pop(0)
        self.keys.append(z.detach().clone())
        self.vals.append(int(a))

    def bias(self, z=None, n_act: Optional[int] = None):
        """给定状态，返回各动作的偏好偏置（叠加到选择分数上）。

        z=None 时用零向量（表示"不看状态"的中性偏好），便于契约校验与降级。
        """
        if z is None:
            z = torch.zeros(self.latent)
        v = self.W.t() @ z.detach()
        if n_act is not None and v.numel() > int(n_act):
            v = v[:int(n_act)]
        return v

    def recall_action(self, z, k: int = 3):
        """相似状态的成功动作（多数投票）；无记忆/不相似返回 None。"""
        if not self.keys:
            return None
        ks = torch.stack(self.keys)
        sim = F.cosine_similarity(z.detach().unsqueeze(0), ks)   # 形状 [N]，不要取 [0]
        acts = [self.vals[i] for i in sim.topk(min(k, len(ks))).indices if sim[i] > 0.8]
        if not acts:
            return None
        return Counter(acts).most_common(1)[0][0]

    def __len__(self):
        return len(self.keys)


def selftest() -> bool:
    """自带回归：契约 + 关联式学习生效 + 状态往返。"""
    ll = LearningLayer()
    ok, probs = learning_conforms(ll, strict=True)
    if not ok:
        print("  学习层契约不满足：%s" % probs)
        return False

    z = torch.randn(8)
    for _ in range(20):
        ll.update(z + 0.01 * torch.randn(8), 2, 1.0)
    learned = int(ll.bias(z).argmax()) == 2 and len(ll) > 0

    recalled = ll.recall_action(z) == 2

    # 状态往返：新实例读回同一份权重与记忆
    st = ll.state()
    ll2 = LearningLayer().apply_state(st)
    same = (torch.allclose(ll2.W, ll.W) and ll2.vals == ll.vals
            and len(ll2) == len(ll)
            and int(ll2.bias(z).argmax()) == int(ll.bias(z).argmax()))

    # learn() 契约入口应等价于 update()
    ll3 = LearningLayer().apply_state(st)
    ll3.learn(z, 2, 1.0)
    entry = torch.allclose(ll3.W, ll.W + ll.lr * torch.outer(z, F.one_hot(torch.tensor(2), 4).float()))
    return bool(learned and recalled and same and entry)


def demo():
    ll = LearningLayer()
    z = torch.randn(8)
    # 在状态 z 附近反复给动作 2 正奖励 → 偏好应偏向 2
    for _ in range(20):
        ll.update(z + 0.01 * torch.randn(8), 2, 1.0)
    b = ll.bias(z)
    print("状态 z 下动作偏好:", [round(x, 3) for x in b.tolist()])
    print("偏好动作:", int(b.argmax()), "(期望接近 2)")
    print("召回相似状态动作:", ll.recall_action(z), "｜ 技能记忆条数:", len(ll))
    print("自描述:", ll.info().to_dict())
    print("能力表:", ll.capabilities())


if __name__ == "__main__":
    demo()
    print("learning 层自测:", "✅ 通过" if selftest() else "❌ 失败")
