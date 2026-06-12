# Ill Style LoRA 训练指南

## 目录结构

```
ill_style_train/
├── train_lora.py          # 训练脚本
└── train_data/
    └── 12_style/          # 训练数据目录
        ├── metadata.jsonl  # 训练数据清单 (可选)
        ├── image1.jpg
        ├── image1.txt     # 图片对应的 tag/caption
        ├── image2.png
        ├── image2.txt
        └── ...
```

---

## 图片标准

### 支持格式
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| JPEG | `.jpg`, `.jpeg` | 最常用 |
| PNG | `.png` | 支持透明通道 |
| WebP | `.webp` | 较好压缩率 |

### 分辨率要求
| 参数 | 默认值 | 说明 |
|------|--------|------|
| 默认分辨率 | 768x1024 (HxW) | 可在配置区修改 |
| 长边 | 建议 ≥ 512px | 太低影响质量 |
| 宽高比 | 3:4 最佳 | 会自动裁剪 |

### 预处理建议

1. **清晰度**：使用高质量图片，避免模糊、低分辨率
2. **内容纯净**：背景尽量简洁，主体突出
3. **风格一致**：同一主题的图片放同一目录
4. **数量建议**：20-100 张风格图片效果较好

---

## Tag / Caption 格式

### 方式一：metadata.jsonl (推荐)

每行一个 JSON，包含图片路径和描述：

```jsonl
{"image": "train_data/12_style/artwork_001.jpg", "caption": "masterpiece, best quality, artistic style, watercolor painting"}
{"image": "train_data/12_style/artwork_002.jpg", "caption": "illustration, style of anime, vibrant colors"}
{"image": "train_data/12_style/sketch_003.png", "caption": "sketch, line art, black and white"}
```

**格式说明**：
- `image`: 图片相对路径（相对于 `train_data_dir`）
- `caption`: 图片描述，逗号分隔的 tag 列表

### 方式二：图片 + 同名 .txt 文件

图片文件和同名文本文件配对：

```
train_data/12_style/artwork_001.jpg
train_data/12_style/artwork_001.txt  ← 包含 caption
```

`artwork_001.txt` 内容示例：
```
masterpiece, best quality, artistic style, watercolor painting, soft colors
```

---

## Tag 编写规范

### 风格训练 Tag 结构

```
质量标签 + 风格标签 + 内容描述 + 艺术特征
```

| 类型 | 示例 | 说明 |
|------|------|------|
| 质量 | `masterpiece, best quality, high quality` | 提升生成质量 |
| 风格 | `watercolor, oil painting, sketch, anime style` | 定义画风 |
| 内容 | `1girl, portrait, landscape, still life` | 主体内容 |
| 特征 | `soft colors, bold strokes, detailed, minimalist` | 艺术特征 |

### 常用质量 Tag

| Tag | 用途 |
|-----|------|
| `masterpiece` | 标志图片为高质量作品 |
| `best quality` | 确保生成最佳质量 |
| `high quality` | 高质量标记 |
| `ultra detailed` | 超级细节 |

### 风格 Tag 参考

| 风格 | Tag 示例 |
|------|----------|
| 动漫 | `anime, anime style, cel shading` |
| 水彩 | `watercolor, watercolor painting, soft edges` |
| 油画 | `oil painting, canvas texture, impasto` |
| 素描 | `sketch, line art, pencil drawing` |
| 厚涂 | `thick paint, impasto, painterly` |
| 扁平 | `flat illustration, vector style, minimalist` |
| 像素 | `pixel art, 8-bit, retro game` |
| 赛璐璐 | `cel shading, anime cel, flat shading` |

### 避免的 Tag

- 避免使用人物名字、版权角色名
- 避免过多细节描述，保持 tag 简洁
- 避免矛盾的风格描述（如同时写 `watercolor` 和 `oil painting`）

---

## 配置修改

在 `train_lora.py` 顶部配置区修改：

```python
# ===================== 配置区 =====================
PRETRAINED_MODEL = "./model/Illustrious_fp16.safetensors"  # 模型路径
TRAIN_DATA_DIR = "./train_data/12_style"                   # 训练数据目录
OUTPUT_DIR = "./output_lora"                               # 输出目录

RESOLUTION = (768, 1024)   # 分辨率 (高度, 宽度)
BATCH_SIZE = 1             # batch size
GRAD_ACC = 8               # 梯度累积步数
LR_UNET = 2.8e-4           # UNet 学习率
RANK = 4                   # LoRA rank
ALPHA = 8                  # LoRA alpha
MAX_STEPS = 3600           # 最大训练步数
SAVE_STEP = 800            # 保存间隔
# ================================================
```

---

## 快速开始

### 1. 准备数据

```bash
# 创建数据目录
mkdir -p train_data/my_style
```

### 2. 放入图片并编写 tag

```
train_data/my_style/
├── image_001.jpg
├── image_001.txt  (包含: masterpiece, best quality, anime style, colorful)
├── image_002.jpg
├── image_002.txt  (包含: illustration, anime, detailed, vibrant)
└── ...
```

### 3. 修改配置

编辑 `train_lora.py`：
```python
PRETRAINED_MODEL = "./model/Illustrious_fp16.safetensors"
TRAIN_DATA_DIR = "./train_data/my_style"
```

### 4. 运行训练

```bash
cd ill_style_train
python train_lora.py
```

### 5. 断点续训

训练会自动保存进度，中断后可继续：

```bash
# 默认会自动从最新 checkpoint 恢复
python train_lora.py

# 强制从头开始（不恢复）
python train_lora.py --no-resume
```

**保存内容**：
- LoRA 权重 (`unet/`)
- 优化器状态 (`optimizer.pt`)
- 调度器状态 (`scheduler.pt`)
- Tokenizer 配置

**恢复时机**：启动时自动检测 `output_dir` 下是否有 checkpoint

---

## 常见问题

**Q: 图片数量多少合适？**
A: 风格训练建议 20-100 张，太多反而容易过拟合。

**Q: tag 要不要用中文？**
A: 建议使用英文，CLIP 模型对英文理解更好。

**Q: 分辨率如何选择？**
A: 根据原图比例选择，训练脚本会自动裁剪到目标分辨率。

**Q: metadata.jsonl 和 .txt 文件哪个更好？**
A: `metadata.jsonl` 更适合大规模数据集，`.txt` 更直观易编辑。
