"""PASM-Lite：零 token 认知循环的教学最小实现（单文件 ~250 行）

它展示 PASM 的核心思想——AI 的"思考"不需要生成 token：
  感知(编码为潜在向量) → 工作记忆 → 情景记忆检索 → 世界模型"想象" → 规划 → 行动
一切内部状态都是连续向量，仅在需要时输出动作。

适合：教学/入门/二次开发试验。生产级完整引擎（情绪/性格/发育/元认知/
全局广播/睡眠巩固/三库记忆/ANN 索引/OpenAI 兼容 API）为私有项目，
不在本仓库提供。

依赖：torch（CPU 即可）。运行：python pasm_lite.py
"""
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7
random.seed(SEED); torch.manual_seed(SEED)


# ================= 环境：10×10 网格，收集能量块 =================
class GridWorld:
    def __init__(self, size=10, n_obs=10, n_food=5):
        self.size, self.n_obs, self.n_food = size, n_obs, n_food
        self.reset()

    def reset(self):
        self.agent = [self.size // 2] * 2
        self.obstacles = set()
        while len(self.obstacles) < self.n_obs:
            self.obstacles.add((random.randrange(self.size), random.randrange(self.size)))
        self.foods = set()
        while len(self.foods) < self.n_food:
            p = (random.randrange(self.size), random.randrange(self.size))
            if p not in self.obstacles and list(p) != self.agent:
                self.foods.add(p)
        self.steps = self.eaten = self.crashes = 0
        return self.observe()

    def observe(self):
        """5×5 局部视野 + 归一化坐标 = 27 维。"""
        x, y = self.agent
        view = torch.full((5, 5), -0.5)          # 越界=墙
        for i in range(5):
            for j in range(5):
                wx, wy = x - 2 + i, y - 2 + j
                if 0 <= wx < self.size and 0 <= wy < self.size:
                    view[i, j] = 0.5 if (wx, wy) in self.obstacles \
                        else 1.0 if (wx, wy) in self.foods else 0.0
        return torch.cat([view.flatten(), torch.tensor([x / self.size, y / self.size])])

    def step(self, a):
        self.steps += 1
        mv = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[a]
        nx, ny = self.agent[0] + mv[0], self.agent[1] + mv[1]
        r = -0.01
        if (0 <= nx < self.size and 0 <= ny < self.size
                and (nx, ny) not in self.obstacles):
            self.agent = [nx, ny]
        else:
            r -= 0.1
            self.crashes += 1
        if tuple(self.agent) in self.foods:
            r += 1.0
            self.foods.remove(tuple(self.agent))
            self.eaten += 1
            while len(self.foods) < self.n_food:
                p = (random.randrange(self.size), random.randrange(self.size))
                if p not in self.obstacles and list(p) != self.agent:
                    self.foods.add(p)
        done = self.steps >= 200
        return self.observe(), r, done


# ================= 层 0：感知自编码器（潜在向量空间） =================
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=27, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 48), nn.ReLU(), nn.Linear(48, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 48), nn.ReLU(), nn.Linear(48, in_dim))

    def encode(self, x):
        return self.enc(x)                      # 确定性潜在向量 z

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z), z


# ================= 层 1：工作记忆（最近 K 步上下文） =================
class WorkMemory:
    def __init__(self, k=4):
        self.buf = deque(maxlen=k)

    def push(self, z):
        self.buf.append(z.detach())

    def context(self):
        return torch.stack(list(self.buf)).mean(0) if self.buf else torch.zeros(8)

    def reset(self):
        self.buf.clear()


# ================= 层 2：情景记忆（KNN 检索 + 事件驱动写入） =================
class Episodic:
    def __init__(self, cap=2000, k=3):
        self.keys = []
        self.vals = []
        self.cap, self.k = cap, k

    def query(self, key):
        if not self.keys:
            return torch.zeros(8)
        ks = torch.stack(self.keys)
        sim = F.cosine_similarity(key.unsqueeze(0), ks)
        top = sim.topk(min(self.k, len(ks)))
        w = F.softmax(top.values * 5.0, 0)
        vals = torch.stack(self.vals)[top.indices]
        m = (vals[:, 4:12] * w.unsqueeze(1)).sum(0)   # value 的"下一状态"段加权
        return m

    def write(self, key, value, r):
        if len(self.keys) >= self.cap:
            del self.keys[0], self.vals[0]
        self.keys.append(key.detach())
        self.vals.append(value.detach())

    def __len__(self):
        return len(self.keys)


# ================= 层 3：世界模型（预测 z′ 与 r） =================
class WorldModel(nn.Module):
    def __init__(self, latent=8, hidden=24):
        super().__init__()
        self.gru = nn.GRUCell(latent + 4 + latent, hidden)
        self.head_z = nn.Linear(hidden, latent)
        self.head_r = nn.Linear(hidden, 1)

    def forward(self, z, a, m, h):
        h = self.gru(torch.cat([z, a, m]), h)
        return h, self.head_z(h), self.head_r(h)


# ================= 规划器：世界模型里的"想象"（简单采样评估） =================
class Planner:
    def __init__(self, wm, n_seq=16, horizon=4):
        self.wm, self.n_seq, self.horizon = wm, n_seq, horizon

    @torch.no_grad()
    def choose(self, z, m, h):
        best_a, best_g = 0, -1e9
        for _ in range(self.n_seq):
            seq = [random.randrange(4) for _ in range(self.horizon)]
            hh, zz, g = h.clone(), z.clone(), 0.0
            for t, a in enumerate(seq):
                hh, zz, rr = self.wm(zz, F.one_hot(torch.tensor(a), 4).float(), m, hh)
                g += (0.9 ** t) * float(rr)
            if g > best_g:
                best_g, best_a = g, seq[0]
        return best_a


# ================= 智能体主循环 =================
class Agent:
    def __init__(self):
        self.ae = AutoEncoder()
        self.wm = WorldModel()
        self.wmem = WorkMemory()
        self.mem = Episodic()
        self.planner = Planner(self.wm)
        self.opt = torch.optim.Adam(list(self.ae.parameters()) + list(self.wm.parameters()), lr=3e-3)
        self.h = torch.zeros(24)

    def act(self, obs):
        z = self.ae.encode(obs)
        self.wmem.push(z)
        c = self.wmem.context()
        key = torch.cat([z, c, self.h])
        m = self.mem.query(key)
        # h 来自上一步真实转移后的上下文；规划与训练使用同一 h 起点（一致性）
        if random.random() < 0.15:          # 简单 ε 探索，防止动作分布锁死
            a = random.randrange(4)
            return a, z, c, m, key
        a = self.planner.choose(z, m, self.h)
        return a, z, c, m, key

    def learn(self, obs, z, c, m, key, a, next_obs, r):
        z2 = self.ae.encode(next_obs)
        a_oh = F.one_hot(torch.tensor(a), 4).float()
        h, zp, rp = self.wm(z, a_oh, m, self.h)
        loss = (z2 - zp).pow(2).mean() + 0.2 * (r - float(rp)) ** 2 \
            + (obs - self.ae.dec(z)).pow(2).mean() * 0.5
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # 事件驱动写入：惊讶（预测误差大）或异常奖励
        err = float((z2.detach() - zp.detach()).pow(2).mean())
        if err > 0.05 or abs(r) > 0.15:
            self.mem.write(key, torch.cat([a_oh, z2.detach(), torch.tensor([r])]), r)
        self.h = h.detach()


def main():
    env = GridWorld()
    ag = Agent()
    # 世界模型预热：随机逛 600 步，先把"动作→下一状态→奖励"拟合个大概，
    # 否则规划器一开始等于掷骰子（与完整引擎的 warmup 同理，教学版极简实现）
    print("世界模型预热 …")
    obs = env.reset()
    for _ in range(600):
        a = random.randrange(4)
        nxt, r, done = env.step(a)
        z, c, m, key = ag.ae.encode(obs), torch.zeros(8), torch.zeros(8), None
        a_oh = F.one_hot(torch.tensor(a), 4).float()
        h, zp, rp = ag.wm(z, a_oh, m, ag.h)
        loss = (ag.ae.encode(nxt) - zp).pow(2).mean() + 0.2 * (r - float(rp)) ** 2
        ag.opt.zero_grad(); loss.backward(); ag.opt.step()
        ag.h = h.detach()
        obs = nxt if not done else env.reset()

    hist = []
    for ep in range(25):
        obs = env.reset()
        ag.wmem.reset()
        ag.h = torch.zeros(24)
        total = 0.0
        for _ in range(200):
            a, z, c, m, key = ag.act(obs)
            nxt, r, done = env.step(a)
            ag.learn(obs, z, c, m, key, a, nxt, r)
            total += r
            obs = nxt
            if done:
                break
        hist.append(total)
        if ep % 10 == 0 or ep == 39:
            print(f"episode {ep}: reward {total:+.2f} | 能量 {env.eaten} | "
                  f"情景记忆 {len(ag.mem)} 条")
    print(f"\n平均奖励(末5集): {sum(hist[-5:])/5:+.2f}（前5集 {sum(hist[:5])/5:+.2f}）")
    print("演示要点：AI 无 token、纯连续向量循环即可探索并积累情景记忆；")
    print("完整引擎在此基础上加 情绪/性格/发育/元认知/睡眠巩固/三库记忆/API 等。")


if __name__ == "__main__":
    main()
