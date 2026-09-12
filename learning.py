"""pasm-lite 学习层（起步版，与 pasm.cognitive.learning 同源思想）

教学版把"学习"做到最轻：用连续向量状态 z 与一个「动作偏好」矩阵做
关联式(Hebbian-like)学习 —— 在经常获得正奖励的状态附近，强化那个动作。
同时维护一个「技能记忆」：把「状态模式 → 成功动作」存下来，遇到相似状态直接调用。

纯 torch / 标准库，无新依赖。运行：python learning.py 看自测。
"""

import torch
import torch.nn.functional as F
from collections import Counter


class LearningLayer:
    """轻量关联式学习层：状态→动作偏好 + 技能记忆召回。"""

    def __init__(self, latent: int = 8, n_act: int = 4, lr: float = 0.05, mem_cap: int = 256):
        self.latent = latent
        self.n_act = n_act
        self.lr = lr
        # W: 线性关联权重，z @ W 得到各动作偏好偏置
        self.W = torch.zeros(latent, n_act)
        self.keys: list = []          # 技能记忆：状态关键向量
        self.vals: list = []          # 对应成功动作
        self.mem_cap = mem_cap

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

    def bias(self, z):
        """给定状态，返回各动作的偏好偏置（叠加到选择分数上）。"""
        return self.W.t() @ z.detach()

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


if __name__ == "__main__":
    demo()
