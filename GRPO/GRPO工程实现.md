# GRPO工程实现

Deepseek 提出

![image.png](image.png)

## 先验知识

### 1. 初始化

1. **策略模型**：初始化 $\pi_\theta$（通常是预训练 LM 或 SFT 后的模型）。
2. **奖励模型**：初始化 $r_\phi$，用于对输出打分。（也可以基于规则）
3. **参考模型**：设定 $\pi_{\text{ref}}$（一般就是初始 SFT 模型），用于计算 KL 惩罚，防止策略跑飞。

---

### 2. 采样输出（对每个问题 q）

1. 从当前策略 $\pi_{\text{old}}$ 中采样一组输出： $\{o_1, o_2, \ldots, o_G\} \sim \pi_{\text{old}}(O|q)$
2. 用奖励模型打分，得到每个输出的标量奖励：$r_i = r_\phi(o_i),\quad i=1,\ldots,G$

---

### 3. 组内奖励归一化（Grouped Normalization）

对同一个问题 q 的 **一组 G 个样本** 做归一化：

1. 平均奖励：$\text{mean}(r) = \frac{1}{G}\sum_{i=1}^G r_i$
2. 标准差：$\text{std}(r) = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \text{mean}(r))^2}$
3. 标准化后的奖励：$\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

---

### 4. 计算优势函数

对第 i 个输出的所有时间步 t：

- 使用 **同一个标量** 作为所有时间步的优势：$\hat{A}_{i,t} = \tilde{r}_i$

> 若是过程监督（Process Supervision），则可以对每个中间推理步骤分别给奖励，再据此构造更细粒度的 $A_{i,t}$。（过程奖励）
> 

---

### 5. 策略优化目标（GRPO 核心）

定义 ratio（策略比）：$\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}$

PPO 式裁剪： $\text{clip}(\rho_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)$

**GRPO 目标函数**（略去期望外的 KL 项写法上的细节，大意如下）：优化时最大化 $J_{\text{GRPO}}(\theta)$，等价于对 −J 做梯度下降。

$$
J_{\text{GRPO}}(\theta)
= \mathbb{E}_{q \sim P(Q),\; \{o_i\}_{i=1}^G \sim \pi_{\text{old}}(O|q)}
\Bigg[
\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\big(
\rho_{i,t}(\theta)\hat{A}_{i,t},
\text{clip}(\rho_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_{i,t}
\big)
\;-\;
\beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
\Bigg]
$$

- $\varepsilon$：裁剪参数，控制更新步长，避免崩溃。
- β：KL 权重，约束策略不偏离参考模型 $\pi_{\text{ref}}$。
- $D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$：策略与参考模型之间的 KL 散度。$D_{\text{KL}}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$
- $|o_i|$：第 i 条回答有多少个 token，用来在“这条回答内部”对所有 token 求平均；
- **G**：对同一个问题采了多少条回答，用来在“这一组回答之间”再求一次平均。

对G和O做如下解释：对**同一个问题 q**，我们从当前策略里采样 **G = 3** 个回答：

- 第 1 个回答 $o_1$：长度 ∣o1∣=4 个 token；token 序列：$w_{1,1}, w_{1,2}, w_{1,3}, w_{1,4}$

$\frac{1}{|o_1|}\sum_{t=1}^{|o_1|}(\cdots)
= \frac{1}{4}\big[(\cdots)_{1,1}+(\cdots)_{1,2}+(\cdots)_{1,3}+(\cdots)_{1,4}\big]$

---

### 6. 迭代训练流程

1. **更新奖励模型**（如果是在线更新的场景）：
    - 一轮训练后，用最新采样的数据继续微调奖励模型 $r_\phi$（很多设置会把它固定不训练，这一步可选）。
2. **更新策略模型**：
    - 用上一步算出的优势和目标函数，对 $\pi_\theta$ 进行多步梯度更新。
3. **重复步骤 2–5**：
    - 不断采样 → 评分 → 归一化奖励 → 计算优势 → 优化策略
    - 直到策略收敛或达到预定步数。

### clip与KL是否冲突？

- **clip：**防止每一步更新太激进，保证优化稳定；约束的是“这次更新”相对于“上次策略”的变化幅度。
- **KL：**防止模型长期“学坏”，过度迎合奖励模型、偏离原始语言能力与人类分布；约束的是“当前策略”相对于“参考模型（SFT）”的整体距离。

## GSM8K数据集

- GSM8K 的中文数学题数据集：`prompt` = 中文题目 `question_zh-cn`；`answer` = 参考答案 `answer_only`
- Trainer 后面会用到 `prompt` 做输入，用 `answer` 作为奖励函数里的「参考答案」。

```python
{
    'question_zh-cn': '纳塔利娅在 4 月份向 48 个朋友出售了视频片段，然后在 5 月份售出了一半的视频片段。娜塔莉亚在四月和五月总共卖出了多少个视频？',
    'answer_only': '72',
}
```

### 从huggingface下载数据集

git clone [https://huggingface.co/datasets/swulling/gsm8k_chinese](https://huggingface.co/datasets/swulling/gsm8k_chinese)

# GRPO实战（单个 batch 的视角）

把上面的步骤压缩成一行「调用时间线」，就是：

1. `__main__`：准备 args / writer / tokenizer / model / dataset
2. `GRPOTrainer(...)`：初始化策略、奖励函数、优化器、缓存
3. `trainer.train()`
4. `DataLoader` 提供 `batch`（prompt+answer）
5. `generate_experiences(batch)`
    - → `generate_samples(batch)`
    - → `model.generate(...)` 生成每个 prompt 的 K 个回复
    - → `get_action_log_probs(model, ...)` 得到旧 log_prob
    - → 调各个 `reward_func` 得到 reward
    - → 在组内标准化，得到 advantage（GRPO 核心）
6. （积累若干 batch 后）`train_step(model, inputs, optimizer, step)`
7. `compute_loss(model, inputs)`
    - → `get_action_log_probs` 得到当前策略 log_prob
    - → 和旧 log_prob 计算 PPO ratio + clip，结合 action_mask 聚合成每条回答的 loss；
    - → 对 4 条回答取平均，得到一个标量 loss；
8. `loss.backward()` → `optimizer.step()`，更新策略 πθ。

**实际案例**

样本原始数据是：

```python
{
    'question_zh-cn': '纳塔利娅在 4 月份向 48 个朋友出售了视频片段，然后在 5 月份售出了一半的视频片段。娜塔莉亚在四月和五月总共卖出了多少个视频？',
    'answer_only': '72',
}
```

---

## 1. DataLoader 产出的 batch 长什么样？

`GSM8KDataset.__getitem__` 会把字段名改成：

```python
def __getitem__(self, index):
    sample = self.data[index]
    answer = sample['answer_only']          # '72'
    prompt = sample['question_zh-cn']       # 那句中文题目
    return {'prompt': prompt, 'answer': answer}

假设 batch_size = 1，DataLoader 产出为
batch = {
    'prompt': [
        '纳塔利娅在 4 月份向 48 个朋友出售了视频片段，然后在 5 月份售出了一半的视频片段。娜塔莉亚在四月和五月总共卖出了多少个视频？'
    ],
    'answer': [
        '72'
    ]
}
```

接下来，`trainer.train()` 里会做：

```python
for idx, batch in enumerate(dataloader):
    inputs = self.generate_experiences(batch)
    ...
```

## 2. generate_experiences(batch)：先采样、再算 reward & advantage

### 2.1 generate_samples：同一题目生成 K 条回答

```python
samples_list = self.generate_samples(batch)

```

### 2.1.1 取出 prompt / answer

```python
prompts = [prompt for prompt in inputs['prompt']]
# => ['那句中文题目']

answers = [answer for answer in inputs['answer']]
# => ['72']

```

### 2.1.2 拼 chat 模板文本

对这一条题目，代码会构造：

```python
input_text = tokenizer.**apply_chat_template**(
    [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': '那句中文题目'}
    ],
    add_generation_prompt=True,
    tokenize=False
)
```

`input_text` 大致是这样的结构（示意）：

```
<System>
按照如下格式回答问题：
<think>...</think>
<answer>...</answer>

<User>
纳塔利娅在 4 月份向 48 个朋友出售了视频片段...

<Assistant>  # add_generation_prompt=True 会自动加一个 assistant 开头
```

### 2.1.3 同一个 prompt 复制 K 份 → 一个 group

设 `num_generations = 4`：

```python
inputs_tok = tokenizer(
    [input_text] * 4,
    max_length=self.args.max_prompt_length,
    truncation=True,
    return_tensors='pt'
)
prompt_ids = inputs_tok['input_ids']
# 形状: [4, L_prompt]
```

这里的意思是：**这一道题，我们让模型一次生成 4 个不同回答**，构成一个 group。

### 2.1.4 调用 model.generate 生成回答

```python
prompt_response_ids = self.model.generate(
    **inputs_tok.to(device),
    max_new_tokens=self.args.max_generate_length,
    temperature=0.9,
    top_p=1,
    top_k=50,
)
# 形状: [4, L_prompt + L_resp]
```

这一步结束后：

- 每一行是一条「完整对话」：`[prompt_tokens, response_tokens]`
- 总共有 4 行，对应 4 条不同回答。

### 2.1.5 切出 response，以及各种 mask

```python
attention_mask = (prompt_response_ids.ne(pad_token_id)).long()
# [4, L_full]

response_ids = prompt_response_ids[:, prompt_ids.size(1):]
# [4, L_resp]  # 只保留回答部分

action_mask = (
    (response_ids.ne(eos_token_id)) &
    (response_ids.ne(pad_token_id))
).long()
num_actions = action_mask.size(1)  # = L_resp

```

然后打包成一个 `Samples`：

```python
samples = Samples(
    prompt_response_ids=prompt_response_ids,  # [4, L_full]
    response_ids=response_ids,                # [4, L_resp]
    prompt='那句中文题目',
    answer='72',
    attention_mask=attention_mask,           # [4, L_full]
    action_mask=action_mask,                 # [4, L_resp]
    num_actions=L_resp,
    response_length=L_resp,
)
给出一个伪造案例方便理解
假设<pad> 的 id = 0；<eos> 的 id = 2
prompt 长度 = 10 个 token（L_prompt = 10）
每条回复长度（截断后） = 6 个 token（L_resp = 6）
⇒ 整条序列总长 L_full = 10 + 6 = 16
num_generations = 4，也就是 一次生成 4 条回答
samples = Samples(
    prompt_response_ids = torch.tensor([
        [101, 10, 11, 12, 13, 14, 15, 16, 17, 18, 201, 202, 203,  72, 204, 2],
        [101, 10, 11, 12, 13, 14, 15, 16, 17, 18, 211, 212, 213,  60, 214, 2],
        [101, 10, 11, 12, 13, 14, 15, 16, 17, 18, 221, 222,  48, 223, 224, 2],
        [101, 10, 11, 12, 13, 14, 15, 16, 17, 18, 231, 232, 233, 120, 234, 2],
    ], dtype=torch.long),                     # [4,16]

    response_ids = torch.tensor([
        [201, 202, 203,  72, 204,2],
        [211, 212, 213,  60, 214,2],
        [221, 222,  48, 223, 224,2],
        [231, 232, 233, 120, 234,2],
    ], dtype=torch.long),                     # [4,6]

    prompt = (
        "纳塔利娅在 4 月份向 48 个朋友出售了视频片段，然后在 5 月份售出了一半的视频片段。"
        "娜塔莉亚在四月和五月总共卖出了多少个视频？"
    ),

    answer = "72",
		#假设每条序列都刚好长度 16，没有 padding
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.long),                     # [4, 16]
		#回复中哪些 token 参与 loss
    action_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
    ], dtype=torch.long),                     # [4, 6]
    num_actions = 6,
    response_length = 6,
)
```

因为 batch_size=1，所以 `samples_list` 里就只有一个 `Samples`。

---

### 2.2 回到 generate_experiences：算 log_prob、reward、advantage

现在 `samples_list` 有一个元素，循环中取出：

```python
prompt_response_ids = samples.prompt_response_ids  # [4, L_full]
response_ids        = samples.response_ids         # [4, L_resp]
attention_mask      = samples.attention_mask      # [4, L_full]
action_mask         = samples.action_mask         # [4, L_resp]
num_actions         = samples.num_actions         # L_resp
prompt              = samples.prompt              # 题目字符串
answer              = samples.answer              # '72'

```

### 2.2.1 计算旧策略下每个 token 的 log_prob（old_action_log_probs）

```python
old_action_log_probs = self.get_action_log_probs(
    self.model, prompt_response_ids, attention_mask, num_actions
)
# 形状: [4, L_resp]
```

`get_action_log_probs` 内部做的事：

1. 把整个序列 `prompt_response_ids` 喂进模型算 logits；
2. 对每个位置做 log_softmax，得到 `log_probs[:, :-1, :]`；
3. 用 `gather` 取出「真实下一个 token」对应的 log_prob；
4. 只保留最后 `num_actions` 个，也就是回答部分的 log_prob：

```python
# 简化理解：每条回答的一串 token 的 logπ_old(a_t | s_t)
old_action_log_probs[i] = [
    log π_old( token_1 ), log π_old( token_2 ), ..., log π_old( token_L_resp )
]
```

### 2.2.2 把 4 条回答 decode 出来，准备 reward 计算

```python
response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
# 举个假想例子，4 条回答可能是：
# [
#   "<think>...48 + 24 = 72...</think><answer>72</answer>",
#   "<think>...48 + 12 = 60...</think><answer>60</answer>",
#   "<think>...算错了...</think><answer>48</answer>",
#   "<think>...胡说八道...</think><answer>120</answer>",
# ]

prompt_texts = [prompt] * 4
answers      = [answer] * 4  # ['72', '72', '72', '72']
```

### 2.2.3 调用 reward_funcs

代码中你传入了 4 个 Python 函数：

```python
reward_funcs = [
    correctness_reward,
    digit_reward,
    hard_format_reward,
    mark_reward,
]
```

generate_experiences 里对每个 reward 做：

```python
rewards_per_func = torch.zeros(4, 4)  # [num_funcs, num_generations]

for i, reward_func in enumerate(reward_funcs):
    output_reward_func = reward_func(
        prompts=prompt_texts,
        responses=response_texts,
        answers=answers
    )
    # output_reward_func 是长度为 4 的 Python list，例如：
    # correctness_reward -> [1.0, 0.0, 0.0, 0.0]
    # digit_reward      -> [1.0, 0.5, 0.5, 0.0]
    # hard_format_reward-> [1.0, 0.0, 0.0, 0.0]
    # mark_reward       -> [0.9, 0.4, 0.2, 0.1]
    rewards_per_func[i] = torch.tensor(output_reward_func)
```

> 上面这些数值只是示意：
> 
> - 第一条回答完全正确、格式也好 → 奖励高；
> - 第二条算错，但是有过程、数字格式对 → 奖励中等；
> - 第三条只写了 48 → 更低；
> - 第四条乱写 → 最低。

### 2.2.4 多个 reward 加权合并

如果你没设置 `reward_weights`，代码会默认：

```python
self.args.reward_weights = [1.0, 1.0, 1.0, 1.0]
```

所以总 reward：

```python
# rewards_per_func: 形状 [4, 4]（4 个 reward 函数 × 4 条回答）
# reward_weights:   [1,1,1,1] → [4,1] 展开

# 逐行乘权重后按 func 维度求和 → [4]
rewards = (rewards_per_func * weights.unsqueeze(1)).sum(dim=0)
```

按上面的示意值，可能得到：

```python
# 每条回答的总 reward（示例）
rewards = tensor([3.9, 0.9, 0.7, 0.1])  # [r1, r2, r3, r4]
```

### 2.2.5 组内标准化 → GRPO 的 advantage

对这 4 个 reward 计算均值和标准差：

```python
mean_group_rewards = rewards.mean()
std_group_rewards  = rewards.std()

advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8)
# 得到形状 [4]，比如示意：
# tensor([ 1.3, -0.3, -0.2, -0.8])
```

含义是：

- 第 1 条回答比组内平均好很多 → 优势为正且最大；
- 其他几条都比平均差 → 优势为负，更新时会被「压下去」。

这就是 **group-relative policy optimization** 的核心：

优势完全依据「同一题中各回答的相对好坏」，不需要 value function。

---

### 2.3 generate_experiences 最终返回什么？

对于这道题（batch 里只有这一题），最后返回一个字典：

```python
inputs = {
    "prompt_response_ids":  # [4, L_full]   4 条完整对话 token
    "attention_mask":       # [4, L_full]
    "action_mask":          # [4, L_resp]   回复部分可训练 token 的 mask
    "old_action_log_probs": # [4, L_resp]   采样时策略的 log_prob
    "ref_action_log_probs": None            # 因为 beta=0
    "advantages":           # [4]           每条回答一个 advantage
}
```

这个 `inputs` 会被塞进 `input_buffer`，之后交给 `train_step` / `compute_loss`。

---

## 3. train_step(inputs)：把这批 experience 用来更新模型

训练循环里：

```python
self.train_step(self.model, inputs, self.optimizer, step)
```

### 3.1 train_step 外壳

```python
def train_step(self, model, inputs, optimizer, step):
    model.train()
    loss = self.compute_loss(model, inputs)
    loss = loss / self.args.gradient_accumulation_steps
    loss.backward()
    ...
```

重点是 `compute_loss(model, inputs)`，我们继续往下看。

---

## 4. compute_loss(model, inputs)：PPO + GRPO 的 loss

```python
prompt_response_ids = inputs['prompt_response_ids']  # [4, L_full]
attention_mask      = inputs['attention_mask']       # [4, L_full]
action_mask         = inputs['action_mask']          # [4, L_resp]
num_actions         = action_mask.size(1)            # L_resp

# 1️⃣ 再算一次当前策略下的 log_prob（更新后的 πθ）
action_log_probs = self.get_action_log_probs(
    model, prompt_response_ids, attention_mask, num_actions
)
# [4, L_resp]
```

### 4.1 old_action_log_probs & ratio

```python
advantages = inputs['advantages']  # [4]

old_action_log_probs = (
    inputs['old_action_log_probs']
    if self.args.num_iterations > 1
    else action_log_probs.detach()
)
```

- 你当前设置 `num_iterations = 1`，所以：
    - old_action_log_probs = action_log_probs.detach()
    - 数值上是一样的，但 old 是常数，不参与梯度。

PPO 的 ratio：

```python
coef_1 = torch.exp(action_log_probs - old_action_log_probs)
# 对当前设定，初始更新时这一项 ≈ 1（因为新旧策略很接近）

coef_2 = torch.clamp(coef_1, 1 - clip_eps, 1 + clip_eps)
# clip_eps = 0.2 → ratio 被限制在 [0.8, 1.2]
```

### 4.2 把句子级优势广播到 token 级

```python
per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # [4,L_resp]
per_token_loss2 = coef_2 * advantages.unsqueeze(1)
per_token_loss  = -torch.min(per_token_loss1, per_token_loss2)
```

- advantages 是 `[4]`，`unsqueeze(1)` 后变成 `[4,1]`，每条回答所有 token 共用一个优势 A_i。
- `min(...)` 就是标准 PPO 的 clipped objective（要最大化的东西加了负号变成要最小化）。

然后只在回答 token 上生效：

```python
per_token_loss = per_token_loss * action_mask  # pad / eos 位置为 0
```

> 直观理解：
> 
> - 那条优势为正的回答（例如完全正确那条），其 token 的 loss 会导向**增大**对应 token 的概率；
> - 优势为负的回答，其 loss 导向**减小**对应 token 的概率；
> - 并且如果 ratio 偏离太多，会被 clip 限制住。

### 4.3 聚合成最终标量 loss

```python
# 每条回答：把所有有效 token 的损失求平均
loss_per_seq = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)
# 然后对 4 条回答再求平均
loss = loss_per_seq.mean()
return loss
```

这时，这个 loss 就是「针对这道题目及其 4 条回答的 GRPO + PPO 目标」。

回到 `train_step`：

```python
loss.backward()   # 产生梯度
optimizer.step()  # 更新策略模型参数 πθ
```

**结果：**

- 对这道视频题目，模型会学到「那条 reward 最高（优势最大的）回答」的 token 分布；
- 同时会「打压」其它 reward 低的回答；
- 下次再遇到类似题目，更倾向于生成类似高 reward 的回答（比如计算出 4 月 48，5 月 24，总共 72 这种）。

---
