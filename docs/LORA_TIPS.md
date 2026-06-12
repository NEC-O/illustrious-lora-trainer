# LoRA 微调技巧手册

> 适用于 SDXL / Illustrious-XL 风格 LoRA 训练（3060 6G 环境）。
> 文档持续更新，遇到实际效果就回来补。

---

## 1. 核心概念：什么叫"甜点区间"

训练 LoRA 是在 **"拟合"和"泛化"之间找平衡**：

| 训练量 | 模型状态 | 出图表现 |
|---|---|---|
| **太少**（欠拟合） | 还没学到底层风格 | 触发词几乎不起作用 |
| **甜点区间** ✅ | 风格学会了，但没"死记"图 | 换 prompt/构图/主体都能保留风格 |
| **太多**（过拟合） | 把训练图的具体细节都"背"下来了 | 只会画训练图的构图/姿态；换 prompt 风格就崩 |

**甜点区间**指 loss 还没触底、但视觉上风格已经稳定**那段时间**。

判别方法：
- 训练 log 里 loss 还在缓慢下降（没 plateau）→ 还在甜点附近
- 训练图 1:1 重现（含背景/姿态/构图）→ 过拟合了
- 用 ComfyUI 跑非训练 prompt：风格明显 + 主体/构图自由 → 甜点内

---

## 2. 数据集：图片数 × epoch × step

### 关键公式

```
总 step × gradient_accumulation = 有效图像前向次数
epoch/图 = 有效图像前向次数 / 图片数
```

### 经验值

| 数据规模 | 推荐 epoch/图 | 风格 LoRA 适配度 |
|---|---|---|
| < 30 张 | 30~50 | 风险高，容易过拟合 |
| 50~80 张 | 20~30 | **甜点区间** ✅ |
| 80~150 张 | 15~25 | **最佳甜点** ✅ |
| 150+ 张 | 10~20 | 数据多时反而要少训 |
| 300+ 张 | 5~15 | 当数据集用，少训防饱和 |

### 快速估算（3060 6G，512x768, grad_accum=2）

| 方案 | 耗时 | epoch/图（80 张） | 评价 |
|---|---|---|---|
| 1000 step | ~1h | 25 epoch | 甜点 ✅ |
| 1500 step | ~1.5h | 37 epoch | 偏上限 |
| 2000 step | ~2h | 50 epoch | 过拟合风险 |
| 2500 step | ~2.5h | 62 epoch | 几乎必过拟合 |

### 实用建议

- **加图片永远比加 step 好**
- 不知道什么时候停？看 loss + 看 checkpoint 实际出图
- **80 张图 + 1200 step** 比 **30 张图 + 2400 step** 强 5 倍

---

## 3. Caption 写法

### 三种主流（详细示例见 `gen_captions.py` 顶部 docstring）

#### (1) Constant caption（纯风格 LoRA）

```text
masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 1girl, solo
```

所有图共用同一段，模型只学"风格"。**新手入门首选**。

#### (2) Trigger + 主体词（风格 + 主体多样性）

```text
masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, <主体>
```

`<主体>` 按图实际换：`1girl, solo` / `1boy, solo` / `2girls` / `1girl, cat` / `landscape` 等。

效果：LoRA 不会和"1girl"强绑定，推理时换 boy/landscape 也能保持风格。

#### (3) Trigger + 简单描述（风格 + 部分属性）

```text
masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 1girl, red hair, white dress
```

每张图带少量"主体+属性"，让 LoRA 学风格时也理解一些"什么颜色""什么服饰"等通用概念。

### 触发词 (Trigger words) 的选取原则

- **独特**：避免用 `1girl`, `long hair` 这种通用词
- **具体**：`thick paint, impasto` 比 `painterly` 更精准
- **多词组合**：2~4 个核心风格词一起用，误触率低
- **避免与基础模型原生词冲突**：SDXL 已有 `anime`，加 `anime style` 强化

### 通用 tag 模板（适合插画风格）

```text
masterpiece, best quality, high quality, ultra detailed, anime, <风格词1>, <风格词2>, ..., <主体>, <属性>
```

---

## 4. LoRA 参数：rank / alpha / scaling

### 核心概念

```python
LoraConfig(
    r=8,              # rank：低秩矩阵的"中间维度"
    lora_alpha=16,    # alpha：scaling 系数
    target_modules=[...],  # 注入位置
    init_lora_weights="gaussian",
)
# 训练时实际 scaling = alpha / rank
# 推理时 LoRA 强度 = scaling × ComfyUI strength 滑块
```

### rank 选取

| rank | 容量 | 适用 | 显存 |
|---|---|---|---|
| 4 | 极小 | 简单风格、显存紧 | +0.5GB |
| 8 | 小 | **风格 LoRA 甜点** ✅ | +0.7GB |
| 16 | 中 | 复杂风格/概念/人物 | +1.2GB |
| 32 | 大 | 复杂多概念 | +2GB |
| 64+ | 极大 | 已接近全量微调 | +3GB+ |

**风格 LoRA 强烈推荐 8~16**。rank=4 容易学不到位。

### alpha 选取

**通用规则：`alpha = 2 × rank`** 是甜点：

| rank | 推荐 alpha | 训练时 scaling |
|---|---|---|
| 4 | 8 | 2.0 |
| 8 | 16 | 2.0 |
| 16 | 32 | 2.0 |
| 32 | 64 | 2.0 |

alpha 越大训练时 LoRA 越"强势"，学习越快，但容易过拟合。
alpha = rank 时，scaling = 1，训练温和，泛化好但学得慢。
alpha > 2 × rank 时，scaling > 2，容易学过头。

### 提取 LoRA 时的 scaling 处理

`lora_to_comfyui.py` 已经把 `scaling = alpha / rank` 烤进 `lora_up.weight`。
所以 ComfyUI 里 strength 滑块 = 1.0 时 = 原始训练强度。

如果想 ComfyUI 加载后更"温和"，可以在脚本里改 `alpha` 参数重新烤：
```bash
python lora_to_comfyui.py -i input.safetensors -o output.safetensors --alpha 4 --rank 8
# alpha=4, rank=8, scaling=0.5  →  效果减半
```

---

## 5. 学习率 (LR)

### 经验值

| 场景 | 推荐 LR | 备注 |
|---|---|---|
| **风格 LoRA 入门** | `3.5e-4` | 主流甜点 |
| 保守/防过拟合 | `1e-4 ~ 2e-4` | 慢但稳 |
| 激进/快速实验 | `5e-4 ~ 1e-3` | 容易过拟合 |
| 大数据集 (>200 图) | `5e-4 ~ 1e-3` | 数据多可以快 |
| LoRA+ (rank-aware LR) | 不同层用不同 LR | 进阶技巧 |

### LR 调度（warmup + cosine）

- **warmup_steps = 总 step 的 5~10%**（1000 step → warmup 50~100）
- 主调度：**cosine annealing**（余弦退火）
- **不要**用 constant LR（容易震荡/晚停）

当前 `train_lora.py` 配置：
- warmup 100 step + cosine 到 0 → **这个组合很标准，OK**

### 早停 (Early Stop)

如果 loss 长时间不下降（连续 200 step 波动 < 1%），可以**手动 Ctrl+C 停**：
- 之前已经存了 checkpoint-X 选最优的
- 不要硬跑完所有 step

---

## 6. Batch Size & Gradient Accumulation

### 当前配置

```bash
--gradient_accumulation_steps 2
```

- 物理 batch_size = 1（受 6G 显存限制）
- 逻辑 batch_size = 1 × 2 = 2（每 2 个 sample 一次 optimizer 更新）
- 实际 step = 总 sample 数 / 2

### 影响

| grad_accum | 等效 batch | 优点 | 缺点 |
|---|---|---|---|
| 1 | 1 | 训练快，loss 噪声大 | 不稳定，梯度震荡 |
| 2~4 | 2~4 | **稳定甜点** ✅ | 速度略慢 |
| 8~16 | 8~16 | 最稳定 | 慢，所需 step 翻倍 |

**3060 6G + 512x768 推荐 2~4**。当前 2 是合理选择。

---

## 7. 分辨率

### 常见选择

| 分辨率 | 显存 | 训练速度 | 适用 |
|---|---|---|---|
| **512x768** | 4GB | 1x | 显存紧、**风格 LoRA 入门** ✅ |
| 768x1024 | 6GB | 0.8x | 数据集是 4:3 时 |
| 1024x1024 | 8GB+ | 0.6x | 方图数据集 |
| 1024x1536 | 12GB+ | 0.4x | 大显存卡 |

### 关键：bucket 训练

最佳实践是按图片**原始比例**缩放到 512~1024 范围，让 SDXL 多尺度 attention 学到不同构图。

当前 `train_lora.py` 用固定 `(512, 768)`，简单但有效。如果数据集是混合构图（方/竖/横），可以考虑改 `--resolution` 为可选项或实现 aspect ratio bucketing。

### ⚠️ 缓存陷阱

如果换分辨率，**必须删 latent cache**（不然会用老分辨率的 latent 训练，浪费）。
`train_lora.py` 已经把 cache 目录按分辨率分开：
```python
self.cache_dir = self.data_dir / f".latent_cache_{resolution[0]}x{resolution[1]}"
```

---

## 8. 混合精度 & 数值稳定性

### 当前策略（3060 6G）

- **模型权重 bf16**
- **AdamW 优化器 fp32**（状态）
- **loss 计算 fp32**
- **autocast bf16**（layer norm/softmax 提升精度）

### 关键避坑（loss nan）

1. **fp16 + AdamW** 在 3060 上会出现 denormal → NaN。**用 bf16 解决**。
2. **time_ids（SDXL 的 add_cond）必须 fp32**，否则数值溢出。
3. **AdamW betas** 不要用默认 (0.9, 0.999)——v_t 在小梯度时变 denormal。
   **改 (0.9, 0.99)** 是经验值。
4. **gradient checkpointing + `enable_input_require_grads()`** 是 peft LoRA 必加。

---

## 9. 注意力优化

### 当前配置

```python
# 已禁用 xformers / triton（这版本有兼容问题）
XFORMERS_DISABLED=1
USE_TRITON=0
# 用 PyTorch 2.x 的 SDP (Scaled Dot Product) attention
```

### 备选方案

| 方案 | 速度 | 兼容性 | 备注 |
|---|---|---|---|
| SDP (PyTorch 2.x) | ⭐⭐⭐ | 最好 | **当前使用** ✅ |
| xformers | ⭐⭐⭐ | 需 0.0.31+ | 当前环境有 bug 已禁 |
| Flash Attention 2 | ⭐⭐⭐⭐ | 需 Ampere+ | 3060 支持，理论更快 |
| math (fallback) | ⭐ | 永远 OK | 慢但最稳 |

如果想试 Flash Attention 2：
```bash
pip install flash-attn --no-build-isolation
```
在 `train_lora.py` 顶部加 `os.environ["USE_FLASH_ATTENTION_2"] = "1"`。

---

## 10. 训练监控

### 关键指标

#### (1) Loss 曲线
- **正常**：从 0.3+ 平滑降到 0.05~0.1
- **异常 A**：1~2 step 就 0.05 → 数据泄漏/过拟合
- **异常 B**：一直不降 → LR 太小 / 数据问题
- **异常 C**：降着降着 NaN → 数值问题（已用 bf16 修）

#### (2) 显存占用
- 用 `nvidia-smi -l 1` 1 秒刷新看峰值
- 接近 6G 时会 OOM 死
- 预留 200~500MB buffer 安全

#### (3) 训练速度
- 当前 512x768 大约 1.8~2.5s/step
- 如果突然变慢 → dataloader 瓶颈 / 磁盘 I/O
- 监控：`--log_every_n_steps 10` 即可（之前每步输出过密）

---

## 11. 过拟合检测（最实用）

### 现场检查法

1. **训练时**：每隔 200 step 用同一 seed + 触发词出 4 张图对比
   - 早期：风格若有若无
   - 甜点：风格明显 + 构图自由
   - 过拟合：构图/姿态开始"克隆"训练图

2. **checkpoint 对比**：把 `checkpoint-200` / `400` / `600` / `800` / `1000` 都用同样的 prompt 跑一次
   - 找风格**最稳定且不僵化**的那个

3. **test prompt 套件**（建议固定保留几个）：
```text
test1: masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 1girl, solo, white background
test2: masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 1boy, solo, sitting
test3: masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 2girls, fighting
test4: masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, 1girl, cat, indoors
test5: masterpiece, best quality, anime style, thick paint, impasto, painterly, clean lines, soft shading, landscape, no humans
```

### 数据集自检

- **29 张全是单人**：推理换"2girl"会糊 → 加多人图
- **29 张全是半身**：推理"全身"会糊 → 加全身图
- **29 张全是某种构图**：推理反构图会崩 → 加多种构图

---

## 12. 常见误区

| 误区 | 真相 |
|---|---|
| ❌ step 越多越好 | step 太多 = 过拟合 |
| ❌ 数据越多越好 | 风格 LoRA 30~150 张甜点，更多反而稀释 |
| ❌ rank 越大越好 | rank 4~16 风格 LoRA 足够，大 rank 易过拟合 |
| ❌ LR 越大越快 | LR 太大直接 loss 起飞/NaN |
| ❌ constant caption 太偷懒 | 对**风格** LoRA 是**正确**做法 |
| ❌ 全用同一段触发词 | 不同层风格（背景 / 人物 / 服饰）要分开触发 |
| ❌ ComfyUI strength 1.0 永远合适 | LoRA 弱时试 1.3~1.5，强时降到 0.6~0.8 |

---

## 13. 项目专属配置（ill_style_train）

### 你的当前最佳配置

```bash
--resolution 512,768
--gradient_accumulation_steps 2
--learning_rate 3.5e-4
--max_train_steps 1000-1500
--save_every_n_steps 200-300
--log_every_n_steps 10
--lr_warmup_steps 100
--network_rank 4-8
--network_alpha 8-16
--seed 42
```

### 提升建议（按性价比排序）

1. **加数据到 80~100 张**（最大提升，性价比最高）
2. **rank 4 → 8**（风格细节更准）
3. **caption 微调**：从 constant 改为 trigger + 主体词
4. **多 step 对比**挑最优 checkpoint
5. **试 LR 1e-4 vs 5e-4** 找最佳点

### 未来想试的（需评估）

- LoCon (Conv layer LoRA)：适合风格，显存稍高
- LoHa (Hadamard product)：参数更省，效果相近
- 高分辨率先训低分辨率再 fine-tune 高分辨率

---

## 14. 速查表

| 任务 | 关键参数 |
|---|---|
| 找甜点 step | 训 200/400/600/800/1000 多 ckpt，对比出图 |
| 防过拟合 | 加图 > 减 step > 降 LR > 降 rank |
| 加速训练 | 降分辨率 > 关 grad checkpoint > 加 grad_accum |
| 显存不够 | 降分辨率 > 开 grad checkpoint > 降 rank |
| 风格不显 | 强化触发词 > 加 LR > 加 rank |
| 风格太死 | 加图多样性 > 减 step > 触发词拆细 |

---

## 15. 参考资源

- [kohya_ss 官方脚本](https://github.com/bmaltais/kohya_ss)：LoRA 训练参考
- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)：peft 库用法
- [diffusers 文档](https://huggingface.co/docs/diffusers)：UNet/LoRA API
- [CivitAI 风格 LoRA 作品](https://civitai.com/models?types=LORA&sortBy=models_v9)：参考别人的训练参数

---

*最后更新：2026-06-08*
