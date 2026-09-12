"""PASM 引擎接口契约（Engine API）—— 纯标准库，零依赖。

【同源镜像】本文件是 PASM 核心仓库 `pasm/engine_api.py` 的同源镜像，
随 PASM-Lite 一同分发，使本仓库不必依赖私有核心包即可使用统一引擎接口。
两份内容应保持一致，改动请同步另一份（`diff` 除本段说明外应无差异）。

为什么需要这一层
----------------
PASM 现在有多套"大脑"实现，它们彼此可替换：

    · pasm.agent.PASMAgent   完整七层引擎（torch，科研/训练/API 服务）
    · pasm.light.PASMAgent   纯 Python 轻量认知体（性格/情绪动力学）
    · PASM-Lite              公开教学版（自编码/工作记忆/情景记忆/世界模型/规划）

过去三者靠"口口相传"保持签名一致，没有一份**可校验的正式契约**，
于是出现"某个能力只在某个实现里、别的实现根本不知道"的漂移
（真实事故：符号层/记忆路由只落在桌面端，核心包从未包含）。

本模块把三者**事实上已经一致**的那套方法固化为契约，并提供：

    EngineInfo / Capabilities  自描述 —— 我是谁、我能做什么
    Engine 协议 + conforms()   可校验 —— 结构对不上会被逐条点出来
    as_engine()                适配器 —— 把任意"大脑"包成标准引擎（不改原对象）
    Registry                   可发现 —— 按名字创建引擎

契约（必需）
------------
    info()                                     -> EngineInfo
    capabilities()                             -> Capabilities
    reset_episode()                            -> Any
    act(obs, learning=True)                    -> (action, report: dict)
    learn(obs, action, next_obs, reward, report=None) -> dict
    snapshot()                                 -> dict   # 含 SNAPSHOT_SECTIONS 七个区块

契约（可选，有则更好）
----------------------
    save(path) / load(path) / freeze_vae() / close()

零依赖：只用标准库，任何环境（含无 torch 的纯 Python 环境）都能 import。
运行 `python -m pasm.engine_api` 或 `engine_api.selftest()` 自检。
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "API_VERSION", "SNAPSHOT_SECTIONS", "REQUIRED_METHODS", "OPTIONAL_METHODS",
    "EngineInfo", "Capabilities", "Engine", "WrappedEngine", "Registry",
    "REGISTRY", "register", "create", "available", "engine_info",
    "as_engine", "conforms", "normalize_snapshot", "selftest",
]

API_VERSION = "1.0"

# 规范快照区块：任何引擎的 snapshot() 都应包含这些键（教学版没有的填 None，
# 这样统一看板/监控可以无差别读取任意引擎的状态）。
SNAPSHOT_SECTIONS: Tuple[str, ...] = (
    "step", "personality", "emotion", "development",
    "memory", "global_workspace", "last_retrieval",
)

REQUIRED_METHODS: Tuple[str, ...] = (
    "info", "capabilities", "reset_episode", "act", "learn", "snapshot",
)
OPTIONAL_METHODS: Tuple[str, ...] = ("save", "load", "freeze_vae", "close")


# ============================================================ 自描述
@dataclass
class EngineInfo:
    """引擎自述：名字/版本/类型/依赖。"""

    name: str
    version: str = "0"
    kind: str = "generic"        # full | light | lite | generic
    description: str = ""
    api: str = API_VERSION
    deps: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Capabilities:
    """引擎能力表（对照七层架构；没有的层保持 False，诚实申报）。"""

    perception: bool = False          # 层0 感知编码
    working_memory: bool = False      # 层1 工作记忆
    memory: bool = False              # 层2 情景/语义/程序记忆
    world_model: bool = False         # 层3 世界模型（含规划）
    emotion: bool = False             # 层4 情绪与动机
    personality: bool = False         # 层5 性格先验
    development: bool = False         # 层5 发育可塑性
    metacognition: bool = False       # 层6 元认知
    global_workspace: bool = False    # 全局工作空间广播
    symbolic: bool = False            # 符号推理层
    llm: bool = False                 # 接入大模型
    multimodal: bool = False          # 多模态
    persistence: bool = False         # 支持存档/读档
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def layers(self) -> List[str]:
        """已具备的能力名（不含 extra）。"""
        d = self.to_dict()
        return [k for k, v in d.items() if k != "extra" and v is True]

    def missing_vs(self, other: "Capabilities") -> List[str]:
        """相对 `other` 缺哪些能力——用于"降级说明"（如用 Lite 顶替完整引擎时）。"""
        mine, oth = self.to_dict(), other.to_dict()
        return [k for k, v in oth.items()
                if k != "extra" and v is True and not mine.get(k)]

    @classmethod
    def of(cls, **kw) -> "Capabilities":
        """只取认识的字段，忽略多余键（便于从 dict/配置构造）。"""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in kw.items() if k in known})


@runtime_checkable
class Engine(Protocol):
    """引擎接口（结构化协议，无需继承即可满足）。"""

    def info(self) -> EngineInfo: ...
    def capabilities(self) -> Capabilities: ...
    def reset_episode(self) -> Any: ...
    def act(self, obs: Any, learning: bool = True) -> Any: ...
    def learn(self, obs: Any, action: Any, next_obs: Any, reward: float,
              report: Optional[dict] = None) -> dict: ...
    def snapshot(self) -> dict: ...


# ============================================================ 校验 / 归一化
def conforms(obj: Any, strict: bool = False) -> Tuple[bool, List[str]]:
    """结构一致性检查 → (是否通过, 问题清单)。

    `strict=True` 时会额外调用一次 snapshot() 确认它返回 dict
    （注意：部分引擎在首次 act() 前调用 snapshot 会抛异常，故非严格模式不调用）。
    """
    problems: List[str] = []
    for m in REQUIRED_METHODS:
        if not callable(getattr(obj, m, None)):
            problems.append("缺少必需方法 %s()" % m)
    for m in OPTIONAL_METHODS:
        if hasattr(obj, m) and not callable(getattr(obj, m)):
            problems.append("可选方法 %s 存在但不可调用" % m)
    if problems:
        return False, problems

    try:
        info = obj.info()
        if not isinstance(info, EngineInfo):
            problems.append("info() 应返回 EngineInfo，实得 %s" % type(info).__name__)
        caps = obj.capabilities()
        if not isinstance(caps, Capabilities):
            problems.append("capabilities() 应返回 Capabilities，实得 %s" % type(caps).__name__)
    except Exception as ex:                      # noqa: BLE001
        problems.append("info()/capabilities() 调用出错：%s" % ex)

    if strict and not problems:
        try:
            if not isinstance(obj.snapshot(), dict):
                problems.append("snapshot() 应返回 dict")
        except Exception as ex:                  # noqa: BLE001
            problems.append("snapshot() 调用出错：%s" % ex)

    return (not problems), problems


def normalize_snapshot(d: Any) -> dict:
    """把任意引擎的快照补齐成规范区块（缺的填 None，多出来的原样保留）。"""
    if not isinstance(d, dict):
        return {k: None for k in SNAPSHOT_SECTIONS}
    out: Dict[str, Any] = {k: d.get(k) for k in SNAPSHOT_SECTIONS}
    for k, v in d.items():
        out.setdefault(k, v)
    return out


def _filter_kwargs(fn: Callable, kwargs: dict) -> dict:
    """按签名过滤关键字实参（兼容不同实现的可选参数差异）。"""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


# ============================================================ 适配器
class WrappedEngine:
    """把任意符合契约的"大脑"对象适配成标准引擎（不改动原对象）。

    额外收益：`act()` 的 report 会被记下并在 `learn()` 缺省时自动复用，
    于是调用方不必自己搬运 (z, c, m, key) 这类内部管线。
    """

    def __init__(self, brain: Any, info: Optional[EngineInfo] = None,
                 capabilities: Optional[Capabilities] = None):
        object.__setattr__(self, "_brain", brain)
        object.__setattr__(self, "_info", info or EngineInfo(name=type(brain).__name__))
        object.__setattr__(self, "_caps", capabilities or Capabilities())
        object.__setattr__(self, "_last_report", None)

    # ---- 自描述 ----
    def info(self) -> EngineInfo:
        return self._info

    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def brain(self) -> Any:
        """被包装的原始对象（需要拿内部状态时用）。"""
        return self._brain

    def __getattr__(self, name: str) -> Any:
        # 其余方法（含引擎私有能力）直通原始对象
        return getattr(object.__getattribute__(self, "_brain"), name)

    # ---- 契约方法 ----
    def reset_episode(self, *a, **kw) -> Any:
        fn = getattr(self._brain, "reset_episode", None)
        return fn(*a, **kw) if callable(fn) else None

    def act(self, obs: Any, learning: bool = True):
        fn = self._brain.act
        r = fn(obs, **_filter_kwargs(fn, {"learning": learning}))
        if isinstance(r, tuple) and len(r) == 2:
            action, report = r
            self._last_report = report if isinstance(report, dict) else None
            return action, self._last_report
        return r, None

    def learn(self, obs: Any, action: Any, next_obs: Any, reward: float,
              report: Optional[dict] = None) -> dict:
        fn = self._brain.learn
        rep = report if report is not None else self._last_report
        out = fn(obs, action, next_obs, reward,
                 **_filter_kwargs(fn, {"report": rep}))
        return out if isinstance(out, dict) else {"result": out}

    def snapshot(self) -> dict:
        fn = getattr(self._brain, "snapshot", None)
        return normalize_snapshot(fn() if callable(fn) else None)

    def save(self, path: str):
        return self._brain.save(path)

    def load(self, path: str):
        return self._brain.load(path)

    def freeze_vae(self):
        fn = getattr(self._brain, "freeze_vae", None)
        return fn() if callable(fn) else None

    def close(self):
        fn = getattr(self._brain, "close", None)
        return fn() if callable(fn) else None


def as_engine(brain: Any, info: Optional[EngineInfo] = None,
              capabilities: Optional[Capabilities] = None) -> WrappedEngine:
    """把任意"大脑"包成标准引擎。`brain` 已含 info()/capabilities() 时可省略后者。"""
    if info is None and callable(getattr(brain, "info", None)):
        got = brain.info()
        if isinstance(got, EngineInfo):
            info = got
    if capabilities is None and callable(getattr(brain, "capabilities", None)):
        got = brain.capabilities()
        if isinstance(got, Capabilities):
            capabilities = got
    return WrappedEngine(brain, info, capabilities)


# ============================================================ 注册表
EngineFactory = Callable[..., Any]


class Registry:
    """按名字发现 / 创建引擎。同名重复注册会报错（除非 replace=True）。"""

    def __init__(self) -> None:
        self._factories: Dict[str, EngineFactory] = {}
        self._meta: Dict[str, EngineInfo] = {}

    # ---- 注册 ----
    def register(self, name: str, factory: Optional[EngineFactory] = None,
                 info: Optional[EngineInfo] = None, replace: bool = False):
        if factory is None:                      # 允许当装饰器用：@REGISTRY.register("x")
            def _deco(f: EngineFactory):
                self.register(name, f, info, replace)
                return f
            return _deco
        if name in self._factories and not replace:
            raise KeyError("引擎 %r 已注册（replace=True 可覆盖）" % name)
        self._factories[name] = factory
        if info is not None:
            self._meta[name] = info
        return factory

    # ---- 发现 ----
    def names(self) -> List[str]:
        """已注册 + 本机可导入的内置引擎，去重排序。"""
        out = set(self._factories)
        for n in _BUILTINS:
            if n not in out and _builtin_importable(n):
                out.add(n)
        return sorted(out)

    def info(self, name: str) -> Optional[EngineInfo]:
        if name in self._meta:
            return self._meta[name]
        if name in _BUILTINS:
            return _builtin_info(name)
        return None

    def spec(self) -> Dict[str, Any]:
        """全部可用引擎的自述（给界面/文档/`--list-engines` 用）。"""
        out: Dict[str, Any] = {}
        for n in self.names():
            inf = self.info(n)
            out[n] = inf.to_dict() if inf else {"name": n}
        return out

    # ---- 创建 ----
    def create(self, name: Optional[str] = None, **cfg) -> Any:
        if name is None:
            pool = self.names()
            if not pool:
                raise RuntimeError("没有任何可用引擎")
            name = pool[0]
        if name in self._factories:
            return self._factories[name](**cfg)
        if name in _BUILTINS and _builtin_importable(name):
            return _builtin_factory(name)(**cfg)
        raise KeyError("未知引擎 %r（可用：%s）" % (name, ", ".join(self.names()) or "无"))


REGISTRY = Registry()
register = REGISTRY.register
create = REGISTRY.create
available = REGISTRY.names
engine_info = REGISTRY.info


# ---- 内置引擎（延迟导入：不在 import 本模块时就把 torch 拉起来）----
_BUILTINS: Dict[str, Tuple[str, str, str, Dict[str, Any]]] = {
    # name: (模块, 类名, kind, 能力开关)
    "pasm": ("pasm.agent", "PASMAgent", "full", {
        "perception": True, "working_memory": True, "memory": True,
        "world_model": True, "emotion": True, "personality": True,
        "development": True, "metacognition": True, "global_workspace": True,
        "persistence": True,
    }),
    "pasm-light": ("pasm.light", "PASMAgent", "light", {
        "personality": True, "emotion": True, "development": True,
    }),
}


def _builtin_importable(name: str) -> bool:
    mod = _BUILTINS[name][0]
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _builtin_info(name: str) -> EngineInfo:
    mod, cls, kind, caps = _BUILTINS[name]
    desc = "完整七层认知引擎" if kind == "full" else "纯 Python 轻量认知体"
    return EngineInfo(name=name, version="", kind=kind, description=desc, deps=("torch",))


def _builtin_factory(name: str) -> EngineFactory:
    mod, cls, kind, capflags = _BUILTINS[name]

    def _make(**cfg: Any) -> Any:
        m = importlib.import_module(mod)
        brain = getattr(m, cls)(**cfg)
        return as_engine(
            brain,
            info=EngineInfo(name=name, version="", kind=kind,
                            description=_builtin_info(name).description,
                            deps=("torch",)),
            capabilities=Capabilities.of(**capflags),
        )

    return _make


# ============================================================ 自检
class _Dummy:
    """最小合规引擎（selftest 用，零依赖）。"""

    def info(self): return EngineInfo(name="dummy", version="0")
    def capabilities(self): return Capabilities(perception=True)
    def reset_episode(self): return None
    def act(self, obs, learning=True): return 0, {"obs": obs}
    def learn(self, obs, action, next_obs, reward, report=None):
        return {"learned": True, "report": report}
    def snapshot(self): return {"step": 1}


def selftest() -> bool:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            print("  ✗ %s" % msg)
        else:
            print("  ✓ %s" % msg)

    print("engine_api selftest (api %s)" % API_VERSION)
    d = _Dummy()
    pass1, probs = conforms(d, strict=True)
    check(pass1, "合规引擎通过校验（问题：%s）" % (probs or "无"))

    pass2, probs2 = conforms(object())
    check((not pass2) and len(probs2) >= len(REQUIRED_METHODS),
          "不合规对象被拦下并列出 %d 条问题" % len(probs2))

    check(normalize_snapshot({"step": 3})["step"] == 3, "快照归一化保留已有区块")
    check(normalize_snapshot({"step": 3})["emotion"] is None, "缺失区块补 None")
    check(normalize_snapshot(None)["memory"] is None, "非 dict 输入安全降级")

    w = as_engine(d)
    a, rep = w.act("o", learning=True)
    check(a == 0 and rep == {"obs": "o"}, "适配器 act 透传 report")
    learned = w.learn("o", a, "o2", 1.0)
    check(learned.get("learned") is True and learned.get("report") == {"obs": "o"},
          "适配器 learn 自动复用上一次 report")

    caps = Capabilities(perception=True, memory=True)
    check(caps.missing_vs(Capabilities(perception=True, memory=True, llm=True)) == ["llm"],
          "missing_vs 能算出降级缺口")

    check(isinstance(available(), list), "可用引擎列表：%s" % (available() or "无"))
    print("engine_api selftest:", "通过" if ok else "失败")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
