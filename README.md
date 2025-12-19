# RLHF_learn


TRPO：用优化问题的约束限制新策略更新步幅，使策略梯度方法训练更加稳定

PPO：用clip函数限制新策略更新步幅，使策略梯度方法能够在工程上落地使用

DPO：核心思想是跳过奖励模型训练，直接利用人类偏好对 来优化策略模型。

GRPO（deepseek）：用组内平均回报值替代Crictic网络，极大降低大模型强化学习训练开销

DAPO（字节）：提高clip上界、动态采样、token级梯度聚合、引入规则奖励、移除KL，优化了GRPO中的多个缺陷

GSPO（qwen）：序列级重要性采样对齐序列级reward，解决token级噪声和MoE的routing replay问题

$max_{\theta} \ \mathbb{E}\Big[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A(s,a) \Big]$
