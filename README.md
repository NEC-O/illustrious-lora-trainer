# Illustrious-XL LoRA 风格训练

> 针对 **RTX 3060 6GB** 显存优化的 SDXL / Illustrious-XL 风格 LoRA 训练工程。
> 直接使用 **diffusers + peft**，无需 kohya-ss / sd-scripts，在 VAE Latent 空间训练，出 LoRA 可直接拖进 ComfyUI。

---

## 1. 特性

- **低显存友好**：6GB 显存跑 768×1024，bf16 + 梯度累积 + VAE latent 预缓存
- **全 diffusers 栈**：训练、加载、保存一条龙，无需外部 kohya-ss
- **VAE Latent 训练**：避免每步重复 encode，训练循环更稳、更快
- **断点续训**：自动检测 `output_dir` 中 checkpoint 恢复训练
- **格式自动转换**：单文件 `.safetensors` checkpoint 自动转 diffusers 目录
- **ComfyUI 兼容**：训练产物一键转换为 kohya / ComfyUI 标准格式
- **多类型 LoRA 配方**：内置风景 / 人物风格 / 角色 / 姿态 / 场景 5 类训练手册

---

## 2. 目录结构

```
ill_style_train/
├── train_lora.py             # 核心训练脚本（diffusers + peft）
├── gen_captions.py           # 批量为图片生成 .txt caption
├── composite_bg.py           # 透明背景图合成（避免 LoRA 学到"贴纸"）
├── lora_to_comfyui.py        # diffusers LoRA → kohya/ComfyUI 格式（遗留工具，新 checkpoint 无需运行）
├── parse_debug.py            # 训练日志 / NaN 调试工具
├── run_train.bat             # Windows 训练启动脚本
├── extract_lora.bat          # Windows LoRA 转换脚本（遗留工具，同上）
├── pyproject.toml            # uv 依赖
├── requirements.txt          # pip 依赖
│
├── model/                    # 基模目录
│   └── Illustrious-XL-v2.0-FP16-Diffusers/
│       ├── unet/             # UNet（FP16）
│       ├── vae/              # VAE
│       ├── text_encoder/     # CLIP-L
│       ├── text_encoder_2/   # CLIP-G
│       ├── tokenizer/        # CLIP-L tokenizer
│       ├── tokenizer_2/      # CLIP-G tokenizer
│       ├── scheduler/        # 调度器
│       └── model_index.json
│
├── train_data/               # 训练数据
│   └── 12_style/             # 示例：12 风格数据集
│       ├── image_27.jpg
│       ├── image_27.txt      # caption
│       └── ...
│
├── output_lora/              # 训练输出（自动生成）
│   ├── checkpoint-400/       # 中间检查点
│   ├── checkpoint-final/     # 最终检查点
│   └── ill_style_final.safetensors   # 转换后的 ComfyUI LoRA
│
└── docs/                     # 详细文档
    ├── TRAINING_GUIDE.md     # 训练总指南
    ├── LORA_TIPS.md          # LoRA 微调技巧
    ├── LORA_TYPES.md         # 不同类型 LoRA 配方
    └── debug-lora-nan-loss-step2.md  # NaN loss 排错
```

---

## 3. 环境要求

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.7.1 (CUDA 12.6) |
| diffusers | ≥ 0.37.1 |
| peft | ≥ 0.15.0 |
| transformers | ≥ 4.56.1 |
| accelerate | ≥ 1.10.1 |
| bitsandbytes | ≥ 0.47.0 |
| GPU | RTX 3060 6GB 起（其他 6G+ 显卡亦可） |

> **重要**：xformers 在 SDXL+LoRA+fp16 上数值不稳会吐 NaN，本工程**禁用 xformers**（已在脚本中 `os.environ["XFORMERS_DISABLED"] = "1"`）。

### 安装依赖

```bash
# 推荐 uv
uv sync

# 或 pip
pip install -r requirements.txt
```

---

## 4. 快速开始

### 4.1 准备基模

将 Illustrious-XL 或其他 SDXL 基模（diffusers 目录格式）放到 `./model/` 下。  
已默认包含：`model/Illustrious-XL-v2.0-FP16-Diffusers/`

> 单文件 `.safetensors` checkpoint 训练脚本会**自动转换**为 diffusers 目录。

### 4.2 准备数据

```bash
mkdir -p train_data/my_style
```

把图片放进 `train_data/my_style/`，运行脚本批量生成 caption：

```bash
python gen_captions.py
```

> 默认 caption 写在 `gen_captions.py` 顶部 `DEFAULT_CAPTION`，按需修改。

### 4.3 修改训练配置

编辑 [train_lora.py](file:///d:/project/OTHER/ill_style_train/train_lora.py) 顶部配置区：

```python
PRETRAINED_MODEL = "./model/Illustrious-XL-v2.0-FP16-Diffusers"
TRAIN_DATA_DIR   = "./train_data/12_style"
OUTPUT_DIR       = "./output_lora"

RESOLUTION    = (768, 1024)  # 高, 宽
BATCH_SIZE    = 1
GRAD_ACC      = 8
LR_UNET       = 2.8e-4
RANK          = 4
ALPHA         = 8
MAX_STEPS     = 1000
SAVE_STEP     = 400
WARMUP_STEPS  = 100
SEED          = 42
```

### 4.4 启动训练

**方式一：直接跑 Python**

```bash
python train_lora.py
```

**方式二：Windows 批处理**（参数已针对 3060 6G 调好）

```bash
run_train.bat
```

### 4.5 训练产物（已直接输出 ComfyUI 格式）

`train_lora.py` 的 `save_checkpoint` 在保存时**已自动转换为 kohya / ComfyUI 通用格式**，无需额外转换脚本。

每个 `checkpoint-<step>/` 目录结构：

```
output_lora/checkpoint-final/
├── lora.safetensors        # ComfyUI 通用 LoRA (~12MB，可直接拖入 Load LoRA 节点)
├── adapter_model.safetensors  # peft 格式 LoRA（用于断点续训）
├── adapter_config.json       # peft adapter 配置
├── optimizer.pt            # AdamW 状态（~50MB，仅 LoRA param，续训用）
├── scheduler.pt            # LR 调度器状态（续训用）
├── tokenizer/              # CLIP-L tokenizer（续训用）
└── tokenizer_2/            # CLIP-G tokenizer（续训用）
```

**单 checkpoint ≈ 70MB**（相比早期 4.79GB 的全量 UNet 方案降 70x）。

**直接使用**：

```bash
# 训练结束后，ComfyUI Load LoRA 节点直接选这个文件
output_lora/checkpoint-final/lora.safetensors
```

> **关于 `lora_to_comfyui.py` / `extract_lora.bat`**：早期版本（旧脚本）需要训练后再单独跑转换，已被 `save_checkpoint` 内置取代。  
> 如果拿到的是**老格式** checkpoint（只有全量 UNet + 嵌入的 LoRA 权重，路径形如 `diffusion_pytorch_model.safetensors`），可使用 [lora_to_comfyui.py](file:///d:/project/OTHER/ill_style_train/lora_to_comfyui.py) 转换：
>
> ```bash
> python lora_to_comfyui.py \
>     -i output_lora/checkpoint-final/diffusion_pytorch_model.safetensors \
>     -o output_lora/ill_style_final.safetensors \
>     --alpha 8 --rank 4
> ```

---

## 5. 训练参数参考

### 5.1 6GB 显存（RTX 3060）默认配方

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| resolution | `(768, 1024)` 或 `(512, 768)` | 高 H，宽 W |
| batch_size | 1 | 显存吃紧时只能 1 |
| gradient_accumulation | 4~8 | 凑出等效 batch |
| learning_rate (UNet) | 2.5e-4 ~ 3.5e-4 | 风格 LoRA 推荐区间 |
| rank | 4~8 | 风格 LoRA 不需要太高 |
| alpha | rank × 2 | 常用 8 / 16 |
| max_train_steps | 1000~2000 | 见 epoch 估算 |
| warmup | 100 | 必备，避免早期 loss 飞 |
| dtype | bfloat16 | 替代 fp16 防 NaN |

### 5.2 Epoch / Step 速算

```
总 step × gradient_accumulation = 有效图像前向次数
epoch/图 = 有效图像前向次数 / 图片数
```

| 数据规模 | 推荐 epoch/图 |
|---|---|
| < 30 张 | 30~50（风险高） |
| 50~80 张 | 20~30（甜点） |
| 80~150 张 | 15~25（最佳） |
| 150+ 张 | 10~20 |
| 300+ 张 | 5~15 |

> **原则**：加图片永远比加 step 好。80 张图 + 1200 step > 30 张 + 2400 step。

---

## 6. 数据准备规范

### 6.1 图片标准

- 格式：`.jpg / .png / .webp`
- 分辨率：长边 ≥ 512px，宽高比 3:4 最佳（自动中心裁剪）
- 数量：20~100 张风格图效果较好
- 清晰度：高分辨率、无水印、无 UI 元素

### 6.2 Caption 写法（两种模式）

**方式 A：图片 + 同名 .txt**

```
image_001.jpg
image_001.txt   ← 内容: masterpiece, best quality, anime style, ...
```

**方式 B：metadata.jsonl**（推荐）

```jsonl
{"image": "train_data/12_style/image_001.jpg", "caption": "masterpiece, best quality, anime style, 1girl, solo"}
{"image": "train_data/12_style/image_002.jpg", "caption": "..."}
```

### 6.3 Caption 三种风格（详见 [docs/LORA_TIPS.md](file:///d:/project/OTHER/ill_style_train/docs/LORA_TIPS.md)）

| 模式 | 用法 | 适用 |
|------|------|------|
| **Constant caption** | 所有图共用同一段 | 纯风格 LoRA（新手首选） |
| **Trigger + 主体词** | 风格 + `1girl`/`1boy`/`landscape` 等 | 风格 + 主体多样性 |
| **Trigger + 描述** | 风格 + 主体 + 少量属性 | 风格同时保留主体特征 |

**推荐**：

```text
masterpiece, best quality, high quality, ultra detailed, anime, [风格触发词], [次要风格词]
```

---

## 7. 不同类型 LoRA 配方速查

详细见 [docs/LORA_TYPES.md](file:///d:/project/OTHER/ill_style_train/docs/LORA_TYPES.md)。

| LoRA 类型 | 数据规模 | rank | LR | epoch/图 |
|---|---|---|---|---|
| 风景 | 50~200 | 8~16 | 3e-4 | 20~30 |
| 人物风格 | 80~300 | 16 | 3e-4 | 15~25 |
| 角色 ID | 15~50 | 32 | 1e-4 | 30~50 |
| 动作/姿态 | 30~100 | 8 | 3e-4 | 25~35 |
| 场景构图 | 50~150 | 8 | 3e-4 | 20~30 |

---

## 8. 辅助工具

### 8.1 透明背景合成 [composite_bg.py](file:///d:/project/OTHER/ill_style_train/composite_bg.py)

透明 PNG 人物图批量合成到渐变背景，避免 LoRA 把"无背景"也学成风格。

```bash
python composite_bg.py
python composite_bg.py --variants 5
```

### 8.2 Caption 批量生成 [gen_captions.py](file:///d:/project/OTHER/ill_style_train/gen_captions.py)

```bash
python gen_captions.py
```

只对还没有 `.txt` 的图片生效，已有 caption 不会被覆盖。

### 8.3 LoRA 格式转换 [lora_to_comfyui.py](file:///d:/project/OTHER/ill_style_train/lora_to_comfyui.py) — 遗留工具

> ⚠️ **当前训练脚本 (`save_checkpoint`) 已内置转换**，新训练的 checkpoint 直接就有 `lora.safetensors`。  
> 此工具仅在需要转换**老格式** checkpoint（路径形如 `diffusion_pytorch_model.safetensors`、含全量 UNet + 嵌入 LoRA）时使用。

diffusers 训练产物 → kohya / ComfyUI 标准格式，并把 `alpha/rank` 的 scaling 烤进权重。

### 8.4 NaN loss 排查 [parse_debug.py](file:///d:/project/OTHER/ill_style_train/parse_debug.py)

读取 `debug-lora-nan-loss-step2.ndjson` 排查训练中 NaN 来源。  
详细见 [docs/debug-lora-nan-loss-step2.md](file:///d:/project/OTHER/ill_style_train/docs/debug-lora-nan-loss-step2.md)。

---

## 9. 常见问题

### Q1: 训练中出 NaN loss

1. 确认 xformers 已禁用（脚本已自动禁用）
2. 改用 **bfloat16** 而非 fp16
3. 加 `lr_warmup_steps`（如 100）
4. 降低 learning_rate
5. 检查训练图是否有全黑/全白异常图
6. 参考 `docs/debug-lora-nan-loss-step2.md`

### Q2: 出图风格不显著

- 增加训练步数（仍在甜点区间）
- 提高 `network_rank` 到 8~16
- 降低数据量到 50~80 张
- 检查 caption 触发词是否清晰

### Q3: 出图只会画训练图（过拟合）

- 降低训练步数
- 增加数据集规模
- 把 caption 改成 trigger + 主体词模式
- 用 `constant caption` 让模型只学风格

### Q4: ComfyUI 加载报错

- 当前训练脚本**已直接输出 ComfyUI 格式 `lora.safetensors`**，直接拖入 Load LoRA 节点即可
- 如果拿到的是老格式 checkpoint（`diffusion_pytorch_model.safetensors`），才需要用 [lora_to_comfyui.py](file:///d:/project/OTHER/ill_style_train/lora_to_comfyui.py) 转换
- 检查 `alpha`/`rank` 与训练一致
- ComfyUI LoRA loader 强度设 0.7~1.0 测试

### Q5: 显存 OOM

- 把 `RESOLUTION` 降到 `(512, 768)` 或更小
- 关闭其他 GPU 进程
- 检查 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（脚本已设）

---

## 10. 参考文档

- [训练总指南](file:///d:/project/OTHER/ill_style_train/docs/TRAINING_GUIDE.md) — 数据 / caption / 配置详解
- [LoRA 微调技巧](file:///d:/project/OTHER/ill_style_train/docs/LORA_TIPS.md) — 甜点区间 / epoch 估算 / caption 写法
- [LoRA 类型配方](file:///d:/project/OTHER/ill_style_train/docs/LORA_TYPES.md) — 风景 / 人物 / 角色 / 动作 / 场景
- [NaN Loss 排错](file:///d:/project/OTHER/ill_style_train/docs/debug-lora-nan-loss-step2.md) — 训练调试实战

---

## 11. 致谢

- 基模：[Illustrious-XL v2.0 FP16 Diffusers](https://huggingface.co/Bercraft/Illustrious-XL-v2.0-FP16-Diffusers)
- 训练栈：[diffusers](https://github.com/huggingface/diffusers) + [peft](https://github.com/huggingface/peft)
- LoRA 概念：Microsoft LoRA (arXiv:2106.09685)
