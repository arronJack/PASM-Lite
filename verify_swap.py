# -*- coding: utf-8 -*-
"""跨档互换验证：核心档 LearningEngine ↔ 教学档 LearningLayer 插进同一引擎。

证明三件事：
  ① 两档都满足同一契约 `pasm.learning/1.0`（learning_conforms strict 通过）
  ② 同一段"调用方代码"跑两档，输出结构同构（能互换）
  ③ 教学引擎可在运行时换学习层，换完继续 act/learn/save/load 全通

运行：`python verify_swap.py`
核心仓位置默认取同级目录的 PASM，可用环境变量 `PASM_CORE` 覆盖。
"""
import os
import sys
import tempfile

LITE = os.path.dirname(os.path.abspath(__file__))
CORE = os.environ.get("PASM_CORE") or os.path.join(os.path.dirname(LITE), "PASM")
for p in (LITE, CORE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from engine import LiteEngine
import learning as LITE_MOD
from pasm.cognitive import learning as CORE_MOD

FAILS = []


def check(name, cond, detail=""):
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (" — " + str(detail) if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 66)
print("① 契约一致性")
core = CORE_MOD.LearningEngine(seed=5)
lite = LITE_MOD.LearningLayer()
ok_c, p_c = CORE_MOD.learning_conforms(core, strict=True)
ok_l, p_l = LITE_MOD.learning_conforms(lite, strict=True)
check("核心档满足契约", ok_c, p_c)
check("教学档满足契约", ok_l, p_l)
check("API 标识一致", CORE_MOD.LEARNING_API == LITE_MOD.LEARNING_API,
      CORE_MOD.LEARNING_API)
check("必需方法集一致",
      set(CORE_MOD.LEARNING_REQUIRED_METHODS) == set(LITE_MOD.LEARNING_REQUIRED_METHODS),
      CORE_MOD.LEARNING_REQUIRED_METHODS)


print("=" * 66)
print("② 同一段调用方代码跑两档（只依赖契约方法）")
# 调用方只认契约：info / capabilities / learn / bias / state —— 两档各跑一遍
def drive_core():
    l = CORE_MOD.LearningEngine(seed=7)
    l.design({"energy": .9, "play": .8, "temper": .3}, 3,
             ["wave", "ball", "dance", "spin", "think"])
    l.learn("praise")
    z = torch.zeros(8)
    return dict(api=l.info().api, tier=l.info().tier, kind=l.info().kind,
                caps=sorted(l.capabilities()), state=sorted(l.state()),
                bias=[round(v, 3) for v in l.bias(z)], n=len(l))


def drive_lite():
    l = LITE_MOD.LearningLayer()
    z = torch.zeros(8)
    l.learn(z, 2, 1.0)
    return dict(api=l.info().api, tier=l.info().tier, kind=l.info().kind,
                caps=sorted(l.capabilities()), state=sorted(l.state()),
                bias=[round(v, 3) for v in l.bias(z).tolist()], n=len(l))


rc, rl = drive_core(), drive_lite()
check("同 API 标识", rc["api"] == rl["api"], rc["api"])
check("能力表键同构", rc["caps"] == rl["caps"], rc["caps"])
check("state 键同构（含 api 字段）", "api" in rc["state"] and "api" in rl["state"])
check("档位可区分（full / teaching）", rc["tier"] != rl["tier"],
      "%s vs %s" % (rc["tier"], rl["tier"]))
check("learn 后偏好长度 = 动作数", len(rc["bias"]) == 5 and len(rl["bias"]) == 4,
      "%s / %s" % (len(rc["bias"]), len(rl["bias"])))

print("=" * 66)
print("③ 教学引擎运行时换学习层")
eng = LiteEngine(seed=1, env_name="grid-10x10")
check("默认档 = teaching", eng.learning_tier == "teaching", eng.learning_tier)

core2 = CORE_MOD.LearningEngine(seed=11)
core2.design({"energy": .9, "play": .8, "temper": .3}, 3,
             ["wave", "ball", "dance", "spin", "think"])
ok, msg = eng.attach_learning(core2)
check("换成核心档成功", ok, msg)
check("引擎档位已变 full", eng.learning_tier == "full", eng.learning_tier)

eng.reset_episode()
for i in range(40):
    a, rep = eng.act()
    eng.learn(reward=(1.0 if i % 3 == 0 else -0.2))
snap = eng.snapshot()
check("换档后 act/learn 全通", snap["memory"]["episodic"] > 0,
      "skills=%s episodic=%s" % (snap["memory"]["skills"], snap["memory"]["episodic"]))
check("学习真的落进新档（整数动作已映射成标签）", len(core2) > 0,
      "%d 条反馈增量" % len(core2))

# 维度不匹配应被拒绝且不动原实现
bad = LITE_MOD.LearningLayer(latent=99, n_act=7)
ok2, msg2 = eng.attach_learning(bad)
check("维度不符被拒", not ok2, msg2)
check("拒绝后原实现未动", eng.learning_tier == "full", eng.learning_tier)

# 换档后存档 → 同档读回
tmp = os.path.join(tempfile.gettempdir(), "_pasm_lite_swap.pt")
eng.save(tmp)
eng_b = LiteEngine(seed=2, env_name="grid-10x10")
eng_b.attach_learning(CORE_MOD.LearningEngine(seed=11))
eng_b.load(tmp)
check("换档后 save/load 往返", eng_b.learning_tier == "full")

# 档位不符应明确报错（不静默串档）
eng_c = LiteEngine(seed=3, env_name="grid-10x10")
try:
    eng_c.load(tmp)
    check("档位不符应报错", False, "未报错（静默串档）")
except ValueError as ex:
    check("档位不符明确报错", "档位" in str(ex), str(ex)[:60])
os.remove(tmp)

print("=" * 66)
print("结果：", "✅ 全部通过" if not FAILS else "❌ 失败 %d 项：%s" % (len(FAILS), FAILS))
sys.exit(1 if FAILS else 0)
