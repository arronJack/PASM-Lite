# -*- coding: utf-8 -*-
"""PASM mathlab —— 离散与线性分析脑（Phase A1，v0.27 前置）。

纯 numpy 实现，无重依赖，桌面版可直接打包。供三类调用方使用：
1. 表格/数据干活：回归、相关、离群点、描述统计（genxls 续改时"分析一下"）
2. agent_team：流程依赖拓扑校验（环检测）、排程能量评估
3. quantum_strategy（Phase C）：马氏链稳态、特征值等数值底座

设计原则：所有函数返回纯 Python 原生类型（dict/list/float），
不向上层泄漏 numpy 类型，方便 JSON 落盘与提示词注入。
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "fit_linear", "fit_multi_linear", "describe", "corr_matrix",
    "solve_linear_system", "matrix_facts", "topo_sort", "has_cycle",
    "dijkstra", "connected_components", "markov_steady", "choose",
    "analyze_table",
]


def _f(x) -> float:
    """numpy 标量 → 原生 float（NaN/Inf 归 0；复数取实部，避免告警）。"""
    v = float(x.real if isinstance(x, (complex, np.complexfloating)) else x)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return round(v, 6)


# ---------------------------------------------------------------- 线性回归

def fit_linear(x: Sequence[float], y: Sequence[float]) -> dict:
    """一元最小二乘：y = a·x + b，返回斜率/截距/R²/预测函数所需的全部量。"""
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    if xs.size != ys.size or xs.size < 2:
        return {"ok": False, "why": "至少需要 2 对数据"}
    if float(np.std(xs)) == 0.0:
        return {"ok": False, "why": "x 全相同，无法拟合"}
    a, b = np.polyfit(xs, ys, 1)
    pred = a * xs + b
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    r = float(np.corrcoef(xs, ys)[0, 1]) if ss_tot > 0 else 0.0
    return {
        "ok": True, "slope": _f(a), "intercept": _f(b),
        "r": _f(r), "r2": _f(r2),
        "strength": ("强" if abs(r) >= 0.8 else "中等" if abs(r) >= 0.5
                     else "弱" if abs(r) >= 0.3 else "几乎不相关"),
        "direction": "正相关" if r > 0 else "负相关" if r < 0 else "无",
        "residual_max": _f(np.max(np.abs(ys - pred))),
        "predict": lambda v: _f(a * float(v) + b),
    }


def fit_multi_linear(X: Sequence[Sequence[float]], y: Sequence[float],
                     names: Optional[List[str]] = None) -> dict:
    """多元最小二乘 y = X·w（带截距列），小样本直接正规方程。"""
    M = np.asarray(X, float)
    yy = np.asarray(y, float)
    if M.ndim != 2 or M.shape[0] != yy.size or M.shape[0] < 2:
        return {"ok": False, "why": "X/y 形状不匹配或样本不足"}
    A = np.column_stack([np.ones(M.shape[0]), M])          # 截距列
    w, *_ = np.linalg.lstsq(A, yy, rcond=None)
    pred = A @ w
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    names = names or [f"x{i+1}" for i in range(M.shape[1])]
    coefs = [{"name": names[i], "coef": _f(w[i + 1])}
             for i in range(M.shape[1])]
    coefs.sort(key=lambda d: abs(d["coef"]), reverse=True)
    return {"ok": True, "intercept": _f(w[0]), "coefficients": coefs,
            "r2": _f(r2)}


# ---------------------------------------------------------------- 描述统计

def describe(series: Sequence[float], name: str = "") -> dict:
    """描述统计 + IQR/3σ 离群点。"""
    a = np.asarray([v for v in series if v is not None], float)
    if a.size == 0:
        return {"ok": False, "why": "空序列"}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    iqr = float(q3 - q1)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mu, sd = float(np.mean(a)), float(np.std(a))
    outliers = [ _f(v) for v in a if v < lo or v > hi ]
    z_out = ([_f(v) for v in a if sd > 0 and abs((v - mu) / sd) > 3]
             if sd > 0 else [])
    return {"ok": True, "name": name, "n": int(a.size),
            "mean": _f(mu), "std": _f(sd),
            "min": _f(np.min(a)), "q1": _f(q1), "median": _f(med),
            "q3": _f(q3), "max": _f(np.max(a)), "iqr": _f(iqr),
            "outliers_iqr": outliers, "outliers_3sigma": z_out}


def corr_matrix(rows: Sequence[Sequence[float]],
                names: Optional[List[str]] = None) -> dict:
    """相关系数矩阵 + 按相关性排序的 Top 对（供提示词直接引用）。"""
    M = np.asarray(rows, float)
    if M.ndim != 2 or M.shape[0] < 3 or M.shape[1] < 2:
        return {"ok": False, "why": "至少 3 行 2 列"}
    names = names or [f"列{i+1}" for i in range(M.shape[1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(M, rowvar=False)
    pairs = []
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            r = C[i, j]
            if math.isnan(float(r)):
                continue
            pairs.append({"a": names[i], "b": names[j], "r": _f(r)})
    pairs.sort(key=lambda d: abs(d["r"]), reverse=True)
    return {"ok": True, "names": names,
            "matrix": [[_f(v) for v in row] for row in C],
            "top_pairs": pairs[:6]}


# ---------------------------------------------------------------- 线性代数

def solve_linear_system(A: Sequence[Sequence[float]],
                        b: Sequence[float]) -> dict:
    """Gauss 消元解 Ax=b；奇异则返回判据。"""
    M = np.asarray(A, float)
    bb = np.asarray(b, float)
    if M.ndim != 2 or M.shape[0] != bb.size:
        return {"ok": False, "why": "形状不匹配"}
    rank_a = int(np.linalg.matrix_rank(M))
    rank_ab = int(np.linalg.matrix_rank(np.column_stack([M, bb])))
    if rank_a < rank_ab:
        return {"ok": False, "why": "无解（矛盾方程）",
                "rank": rank_a, "rank_aug": rank_ab}
    if rank_a < M.shape[1]:
        return {"ok": False, "why": "无穷多解（欠定）",
                "rank": rank_a, "unknowns": int(M.shape[1])}
    x = np.linalg.solve(M, bb)
    return {"ok": True, "solution": [_f(v) for v in x]}


def matrix_facts(A: Sequence[Sequence[float]]) -> dict:
    """行列式/秩/逆（可逆时）/特征值（实对称取实部）。"""
    M = np.asarray(A, float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return {"ok": False, "why": "仅支持方阵"}
    det = _f(np.linalg.det(M))
    out = {"ok": True, "n": int(M.shape[0]), "det": det,
           "rank": int(np.linalg.matrix_rank(M)),
           "invertible": abs(det) > 1e-12,
           "eigenvalues": [_f(v) for v in np.linalg.eigvals(M)]}
    if out["invertible"]:
        out["inverse"] = [[_f(v) for v in row]
                          for row in np.linalg.inv(M)]
    return out


# ---------------------------------------------------------------- 离散：图

def topo_sort(n: int, edges: Sequence[Tuple[int, int]]) -> dict:
    """Kahn 拓扑排序；有环返回 cycle=True（流程依赖校验用）。"""
    adj: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return {"ok": len(order) == n, "order": order,
            "cycle": len(order) != n}


def has_cycle(n: int, edges: Sequence[Tuple[int, int]]) -> bool:
    return topo_sort(n, edges)["cycle"]


def dijkstra(graph: Dict[int, List[Tuple[int, float]]],
             src: int) -> dict:
    """标准 Dijkstra，graph: {节点: [(邻居, 权重), ...]}。"""
    dist: Dict[int, float] = {src: 0.0}
    pq: List[Tuple[float, int]] = [(0.0, src)]
    prev: Dict[int, int] = {}
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in graph.get(u, []):
            nd = d + float(w)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    paths = {}
    for v in dist:
        if v == src:
            continue
        p, cur = [v], v
        while cur in prev:
            cur = prev[cur]
            p.append(cur)
        paths[str(v)] = list(reversed(p))
    return {"ok": True, "dist": {str(k): _f(v) for k, v in dist.items()},
            "paths": paths}


def connected_components(n: int, edges: Sequence[Tuple[int, int]]) -> dict:
    """无向图连通分量（团队/关系聚类用）。"""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    comps: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(sorted(comp))
    return {"ok": True, "count": len(comps), "components": comps}


# ---------------------------------------------------------------- 离散：其他

def markov_steady(P: Sequence[Sequence[float]], iters: int = 2000) -> dict:
    """行随机矩阵的平稳分布（幂迭代；不收敛时按周期阵近似）。"""
    M = np.asarray(P, float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return {"ok": False, "why": "需方阵"}
    if np.any(np.abs(M.sum(axis=1) - 1) > 1e-6):
        return {"ok": False, "why": "不是行随机矩阵（每行和须为 1）"}
    v = np.ones(M.shape[0]) / M.shape[0]
    for _ in range(iters):
        v = v @ M
    return {"ok": True, "stationary": [_f(x) for x in v]}


def choose(n: int, k: int) -> int:
    return math.comb(n, k)


# ---------------------------------------------------------------- 一体化入口

def analyze_table(header: List[str], rows: List[List],
                  target: Optional[str] = None) -> dict:
    """表格一键分析：数值列描述统计 + 相关 Top 对 + （指定目标列时）回归。

    供 genxls/表格续改的"分析一下"直接调用；返回 dict 的 summary 字段
    是一段可直接拼进提示词/回复的中文摘要。
    """
    num_cols: Dict[str, List[float]] = {}
    for j, h in enumerate(header):
        vals = []
        for r in rows:
            if j < len(r):
                try:
                    vals.append(float(str(r[j]).replace(",", "")
                                      .replace("%", "").strip()))
                except (ValueError, TypeError):
                    vals = []
                    break
        if len(vals) >= 3:
            num_cols[str(h)] = vals
    if not num_cols:
        return {"ok": False, "why": "没有可解析的数值列（≥3 行）"}

    names = list(num_cols)
    if target is None and len(names) >= 2:
        target = names[-1]          # 未指定目标列时，默认拿最后一列当回归目标
    stats = [describe(v, n) for n, v in num_cols.items()]
    mat = [ [num_cols[n][i] for n in names]
            for i in range(min(len(v) for v in num_cols.values())) ]
    corr = corr_matrix(mat, names)
    summary_parts = []
    for s in stats:
        line = (f"「{s['name']}」均值 {_f(s['mean'])}、中位数 {_f(s['median'])}、"
                f"标准差 {_f(s['std'])}、范围 [{_f(s['min'])}, {_f(s['max'])}]")
        if s["outliers_iqr"]:
            line += f"、离群点 {s['outliers_iqr'][:4]}"
        summary_parts.append(line)
    if corr.get("top_pairs"):
        ps = corr["top_pairs"][:3]
        rel = "；".join(f"{p['a']}×{p['b']} r={p['r']}" for p in ps)
        summary_parts.append(f"相关性最强的几对：{rel}")
    reg = None
    if target and target in num_cols and len(names) >= 2:
        xs_names = [n for n in names if n != target]
        xs = [[num_cols[n][i] for n in xs_names]
              for i in range(min(len(v) for v in num_cols.values()))]
        ys = num_cols[target]
        reg = fit_multi_linear(xs, ys, xs_names)
        if reg.get("ok"):
            cs = "、".join(f"{c['name']}系数{c['coef']}" for c in
                           reg["coefficients"][:3])
            summary_parts.append(
                f"以「{target}」为目标的多因素拟合：R²={reg['r2']}；{cs}")
    return {"ok": True, "numeric_columns": names, "stats": stats,
            "correlation": corr, "regression": reg,
            "summary": "\n".join(summary_parts)}
