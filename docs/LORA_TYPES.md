# LoRA 类型与数据准备手册

> 不同类型的 LoRA 训练策略**完全不同**。风格 LoRA 的最佳实践用在角色 LoRA 上会失败，反之亦然。
> 本文按 LoRA 类型给出**数据集准备 + caption 写法 + 参数建议**的完整配方。

---

## 0. 速查表

| LoRA 类型 | 核心目标 | 数据集规模 | 关键参数 | 难度 |
|---|---|---|---|---|
| **风景** | 还原特定场景的画风/光影 | 50~200 张 | rank 8~16, LR 3e-4, 20~30 epoch | ⭐⭐ |
| **人物风格** | 还原某画师/作品的角色画风 | 80~300 张 | rank 16, LR 3e-4, 15~25 epoch | ⭐⭐ |
| **人物迁移/角色** | 还原特定角色的 ID（长相/服饰） | 15~50 张 | rank 32, LR 1e-4, 30~50 epoch | ⭐⭐⭐⭐ |
| **动作/姿态** | 学特定动作/姿势 | 30~100 张 | rank 8, LR 3e-4, 25~35 epoch | ⭐⭐⭐ |
| **场景构图** | 学特定镜头/视角/构图 | 50~150 张 | rank 8, LR 3e-4, 20~30 epoch | ⭐⭐ |

**通用原则**：任务越"具体"（如角色 ID），数据要越**精**；任务越"风格化"（如画风），数据要越**广**。

---

## 1. 风景 LoRA

### 目标

让模型学会**特定画风下的风景**（水彩山水 / 赛博朋克城市 / 油画森林 / 概念场景等）。

### 数据集要求

| 维度 | 建议 |
|---|---|
| **数量** | 50~200 张（少而精，构图和风格要**统一**） |
| **来源** | 自己拍的 / 找的高质量参考 / AI 生成（推荐） |
| **构图** | 多样化（远景/中景/特写/鸟瞰/平视） |
| **主题** | 集中在 1~2 个主题（全是"森林"或全是"城市"），避免混杂 |
| **质量** | 必须是高分辨率 + 清晰，无水印/UI 元素/文字 |
| **尺寸** | 至少 1024×1024 起，风景对分辨率敏感 |

### Caption 模板

**推荐**：constant caption（所有图共用）

```
masterpiece, best quality, high quality, ultra detailed,
anime, [风格触发词], [次要风格词], [基础主体]
```

**示例（赛博朋克城市风）**：
```
masterpiece, best quality, high quality, ultra detailed, anime, cyberpunk city style, neon lights, futuristic, no humans, building, scenery
```

每张图**共用同一 caption**——因为风景 LoRA 是学**风格**，不学具体某栋楼。

如果一定要区分：
```
masterpiece, best quality, anime, watercolor, [主体类别], [主体描述]

例如: masterpiece, best quality, anime, watercolor, landscape, mountain
```

**不要**写"夜间/白天/雨雪"——这些是图本身属性，会让 LoRA 偏向特定时间。

### 参数建议

```bash
--network_rank 8~16
--network_alpha 16~32
--learning_rate 3e-4
--max_train_steps 总图数 × 25 / 2
--resolution 768,768 或 1024,768
```

### 常见坑

- ❌ 数据集是混合主题（5 张山、5 张海、5 张城市、5 张建筑）→ LoRA 学到的是"杂乱"
- ✅ 集中一个主题（30 张全是"森林"或全是"城市夜景"）
- ❌ 用了低质量/有水印的图 → 模型学到噪声
- ❌ caption 写得太具体（"red sunset, rainy day"） → 风格被场景属性污染

### 推理建议

```text
masterpiece, best quality, anime, [你的触发词], [新主体类别]
例如: masterpiece, best quality, anime, watercolor, mountain, sunset
```

新主体用 LoRA 没见过的（比如训练时只有"森林"，推理时给"沙漠"）—— 触发词 + 新主体一起出现，LoRA 把"沙漠"染上 watercolor 风格。

---

## 2. 人物风格 LoRA

### 目标

还原**某画师/作品的角色画风**（画风 + 角色气质），但**不绑定特定角色**。

例如：想练出"宫崎骏风格"、"新海诚风格"、"鬼灭之刃风格"——风格可复用到任何角色。

### 数据集要求

| 维度 | 建议 |
|---|---|
| **数量** | 80~300 张（要够多样才能学"风格"而非"角色"） |
| **构图** | 极度多样化（半身/全身/多角度/不同表情/不同服装） |
| **角色** | 多角色（5+ 不同角色），**避免学到某具体角色** |
| **质量** | 高质量原作/同人，禁止 AI 二次生成（会污染风格） |
| **主题** | 集中在同一部作品/同一画师 |

### 关键：避免"角色污染"

**风景 LoRA 可以 constant caption，但人物风格 LoRA 一定要区分主体**。

```
错误: 所有图都用 "1girl, solo"   → LoRA 绑死到"1girl"
正确: 1girl, 1boy, 2girls, multiple girls 等混用
```

### Caption 模板

**推荐**：Trigger + 主体词（每张图按主体写）

```
masterpiece, best quality, high quality, ultra detailed,
anime, [风格触发词], [次要风格词], <主体>

<主体> 选项:
  1girl, solo
  1boy, solo
  2girls
  1girl, 1boy
  multiple girls
```

**示例（吉卜力风格）**：
- 图 A：1girl + 长发 + 草原 → `masterpiece, best quality, anime, ghibli style, soft watercolor, 1girl, solo, long hair, grass`
- 图 B：1boy + 短发 + 城市 → `masterpiece, best quality, anime, ghibli style, soft watercolor, 1boy, solo, short hair, city`
- 图 C：2girls + 室内 → `masterpiece, best quality, anime, ghibli style, soft watercolor, 2girls, indoor`

**主体词要每次换**，让 LoRA 学到"风格"而非"角色"。

### 参数建议

```bash
--network_rank 16          # 风格需要中等容量
--network_alpha 32         # alpha = 2 × rank
--learning_rate 3e-4
--max_train_steps 总图数 × 20 / 2
--resolution 512,768
```

### 常见坑

- ❌ 50 张图全是同一个角色 → 训出"角色 LoRA"而非"风格 LoRA"
- ❌ caption 写得太具体（"red hair, school uniform"） → 风格被属性污染
- ❌ 数据集混杂多个画师 → 风格混乱
- ❌ 图片分辨率低（512 以下）→ 风格细节学不到位

### 推理建议

```text
masterpiece, best quality, anime, [你的触发词], [新角色描述]
```

**新角色 + 触发词 = 风格化的新角色**。

---

## 3. 人物迁移 / 角色 LoRA

### 目标

还原**特定角色**（长相、发型、服饰、气质）—— 这是最难但最受欢迎的 LoRA 类型。

例如：还原"原神里纳西妲"、"鬼灭祢豆子"、"真人 cosplay 角色"。

### 数据集要求（关键中的关键）

| 维度 | 建议 |
|---|---|
| **数量** | 15~50 张（**少而精**，图多反而稀释 ID） |
| **构图** | **多角度多表情**（正面/侧面/背面/微笑/严肃/特写/全身） |
| **背景** | 多样化（让 LoRA 不绑死 BG） |
| **质量** | 高清 + 角色特征清晰 |
| **来源** | 官方立绘/原作截图/高质量同人/自拍 |
| **避免** | 同一姿势/同一表情/同一构图（数据无意义） |

### 黄金法则：多角度 + 多表情 + 一致 ID

```
✅  30 张: 5 角度 × 3 表情 × 2 距离 = 30 张有意义数据
❌  30 张: 30 张同角度/同姿势/同表情 = 1 张有意义数据
```

### Caption 模板

**推荐**：Trigger + 详细主体描述（Danbooru 自然 tag 风格）

```
masterpiece, best quality, high quality, ultra detailed,
anime, [角色触发词: e.g. "nahida"], [主体类型: 1girl],
[发型: long hair, twintails, green hair],
[眼睛: green eyes],
[服饰: white dress, gold ornaments],
[其他: solo, looking at viewer, ...]
```

**示例（角色 LoRA）**：
```
masterpiece, best quality, high quality, ultra detailed,
anime, character_nahida, 1girl, solo, long hair, twintails,
green hair, green eyes, white dress, gold headdress,
looking at viewer, blush, smile
```

**关键**：
- **触发词要独特**（`character_nahida` 比 `1girl` 强 100 倍）
- **角色 ID 特征要写全**（发型、发色、眼睛、服饰、配件）
- **背景/构图/动作不写**（让 LoRA 自由组合）

### 参数建议（这个类型很敏感）

```bash
--network_rank 32          # 角色 ID 需要大容量
--network_alpha 32         # alpha = rank（保守）
--learning_rate 1e-4       # **比风格 LoRA 低**，防过拟合
--max_train_steps 总图数 × 40 / 2
--resolution 512,768 或 768,1024
```

**注**：角色 LoRA 的 LR 通常比风格 LoRA 低 2~3 倍，否则容易学到"死记"训练图。

### 触发词设计（重要！）

**禁用通用词**：
```
❌  触发词: "girl" / "anime" / "character"  ← 跟 SDXL 原生词冲突
❌  触发词: "red hair" / "twintails"  ← 通用 tag，污染风险
```

**推荐独特词**：
```
✅  触发词: "character_nahida"      ← 复合词，独特
✅  触发词: "suzume_style"          ← 短句+_style
✅  触发词: "mychara_01"            ← 个人命名
✅  触发词: "cos_saber_v2"          ← 项目化命名
```

### 常见坑

- ❌ 15 张图全是同角度 → LoRA 只学到一个角度，换角度就崩
- ❌ 触发词是通用词（"girl"）→ 和 SDXL 原生词冲突
- ❌ LR 太高（5e-4）→ 过拟合，推理时画不出 LoRA 之外的变化
- ❌ rank 太小（4）→ 容量不够，特征学不全
- ❌ alpha 太大（rank=8, alpha=32）→ 训练过激，泛化差
- ❌ 没分角度 → 推理换角度出图糊

### 推理建议

```text
masterpiece, best quality, [角色触发词], [动作/场景/服饰]
```

**触发词 + 新描述 = 角色在新场景**。

**强度建议**：
- 标准用法：ComfyUI strength 0.7~1.0
- 完全还原：strength 1.0 + 详细描述
- 创意融合：strength 0.4~0.6（角色 + 其他 LoRA 混合）

### 数据集准备示例

**15 张图的角色 LoRA 准备清单**：

```
img_01.png  正面, 微笑, 半身, 室内
img_02.png  侧面, 严肃, 全身, 室外
img_03.png  背面, 中性, 全身, 室外
img_04.png  3/4 角度, 大笑, 特写, 室内
img_05.png  俯视, 微笑, 半身, 室外
img_06.png  正面, 害羞, 半身, 室内
img_07.png  侧面, 闭眼, 半身, 室外
img_08.png  正面, 战斗表情, 全身, 室外
img_09.png  仰视, 惊讶, 特写, 室内
img_10.png  背面, 行走, 全身, 室外
img_11.png  正面, 微笑, 全身, 多人物背景
img_12.png  3/4 角度, 严肃, 半身, 室内
img_13.png  正面, 中性, 特写, 室内
img_14.png  侧面, 战斗, 全身, 室外
img_15.png  正面, 微笑, 半身, 室外
```

**多样化 × 15 张** > **同质化 × 50 张**。

---

## 4. 动作 / 姿态 LoRA

### 目标

让模型学会**特定动作/姿势/构图**（如"双手叉腰"、"战斗姿态"、"舞蹈动作"），不绑定角色。

### 数据集要求

| 维度 | 建议 |
|---|---|
| **数量** | 30~100 张（**动作越复杂，图越多**） |
| **构图** | **同一动作、不同角色、不同场景**（关键！） |
| **角色** | 多样化（避免学到某具体角色） |
| **质量** | 动作清晰、姿势完整 |
| **避免** | 同一动作 + 同一角色（等于 1 张数据） |

### Caption 模板

**推荐**：Trigger + 主体 + 动作描述

```
masterpiece, best quality, anime, [动作触发词], <主体>, <动作细节>
```

**示例（"战斗姿态" LoRA）**：

```
masterpiece, best quality, anime, dynamic_pose_battle, 1girl, solo, holding sword, action pose, dynamic angle
```

**注意**：动作 LoRA 的 trigger 应该是"动作描述"而非"风格描述"。

### 参数建议

```bash
--network_rank 8~16
--network_alpha 16~32
--learning_rate 3e-4
--max_train_steps 总图数 × 30 / 2
```

### 常见坑

- ❌ 30 张图都是"1girl 战斗" → 训出"1girl LoRA"而非"战斗 LoRA"
- ❌ 动作描述不具体 → 触发词起不到作用
- ❌ trigger 写得太长（"1girl with sword in dynamic battle pose"）→ 难触发

### 推理建议

```text
masterpiece, best quality, [动作触发词], [角色/场景描述]
```

**触发词 = 强制动作**。

---

## 5. 场景构图 LoRA

### 目标

让模型学会**特定构图/视角/镜头**（如"鸟瞰视角"、"对称构图"、"电影感宽屏"、"鱼眼镜头"），不绑定具体内容。

### 数据集要求

| 维度 | 建议 |
|---|---|
| **数量** | 50~150 张 |
| **构图** | **同一构图/视角，不同主体/场景**（关键） |
| **主题** | 多样化（室内/室外/人物/风景都要有） |
| **质量** | 构图特征明显、视觉冲击力强 |

### Caption 模板

**推荐**：Trigger + 通用描述（不写具体主体）

```
masterpiece, best quality, anime, [构图触发词], [构图描述]
```

**示例（"鸟瞰视角" LoRA）**：

```
masterpiece, best quality, anime, bird_eye_view, looking down, aerial perspective
```

不写"1girl / city / forest"——让 LoRA 只学"鸟瞰"。

### 参数建议

```bash
--network_rank 8
--network_alpha 16
--learning_rate 3e-4
--max_train_steps 总图数 × 25 / 2
```

### 常见坑

- ❌ 50 张图都是"鸟瞰城市" → 训出"鸟瞰城市 LoRA"而非"鸟瞰 LoRA"
- ❌ 构图描述不突出 → 触发词作用弱
- ❌ 触发词用通用词（"high angle"）→ 与 SDXL 原生词冲突

### 推理建议

```text
masterpiece, best quality, [构图触发词], [新内容]
```

**触发词 = 强制构图**。

---

## 6. 其他常见类型（简表）

### 服饰 LoRA（特定服装风格）

| 维度 | 建议 |
|---|---|
| 数据 | 30~100 张（多种角色穿同种服饰） |
| Caption | Trigger + 服饰描述（"cyberpunk_suit"） |
| 注意 | 角色要多样化，避免学到"角色 ID" |
| 触发词 | `"suit_xxx_style"`, `"dress_yyy"` |

### 物体 LoRA（特定物品/概念）

| 维度 | 建议 |
|---|---|
| 数据 | 30~80 张（同一物体多角度多场景） |
| Caption | Trigger + 物体描述（"concept_book"） |
| 注意 | 背景要多样化 |
| 触发词 | `"object_xxx"`, `"concept_yyy"` |

### 表情 LoRA

| 维度 | 建议 |
|---|---|
| 数据 | 30~80 张（同一表情多角色） |
| Caption | Trigger + 表情描述（"smiling_sadly"） |
| 注意 | 角色要多样化（避免学到某具体角色） |
| 触发词 | `"emotion_xxx"`, `"expression_yyy"` |

### 色彩 LoRA（特定色调）

| 维度 | 建议 |
|---|---|
| 数据 | 50~150 张（同一色调不同内容） |
| Caption | Trigger + 色调描述（"warm_tone"） |
| 注意 | 主体要多样化 |
| 触发词 | `"color_warm"`, `"tone_cold"` |

---

## 7. 数据集准备通用原则

### 一、数据质量 > 数量

- **10 张高质量 > 100 张低质量**
- 模糊、水印、UI 元素（界面/字幕/按钮）、低分辨率 → 全部剔除
- AI 生成的图（除非风格统一）→ 慎用，可能污染风格

### 二、多样性 vs 一致性

| 任务 | 多样性 | 一致性 |
|---|---|---|
| 风景 LoRA | 中（多构图） | 高（同一风格/主题） |
| 人物风格 LoRA | 高（多角色） | 中（同一画师） |
| 角色 LoRA | 中（多角度/表情） | 高（同一角色） |
| 动作 LoRA | 高（多角色） | 高（同一动作） |
| 构图 LoRA | 高（多主题） | 高（同一构图） |

**平衡原则**：跨数据变化的部分 → **caption 不写**（让 LoRA 自由）；固定的部分 → **caption 写**（让 LoRA 强化）。

### 三、构图多样性

| 类别 | 必备角度 |
|---|---|
| **风景** | 远/中/近/特写 + 仰/俯/平视 |
| **人物** | 正面/侧面/背面/3/4 + 仰/俯/平视 + 全身/半身/特写 |
| **角色** | 至少 4 角度 + 3 表情 + 2 距离 |

### 四、预处理

#### 推荐工具

- **裁剪/缩放**：PIL / OpenCV
- **背景去除**：`rembg`（`pip install rembg`）— 但要清楚风险（见 `composite_bg.py`）
- **批量重命名**：`Bulk Rename Utility` / PowerShell
- **数据审查**：`python composite_bg.py`（合成背景用）

#### 分辨率策略

- SDXL 原生 1024×1024
- 训练时用 512×768 / 768×1024 节省显存
- 推理时用 1024×1024 / 1024×1536
- **比例要匹配**：竖图训的就别横图推理（除非有 bucketing）

### 五、数据集自检清单

训练前**必须**检查：

```
[ ]  所有图 ≥ 512px 短边
[ ]  无水印/UI 元素/字幕
[ ]  无重复图（hash 检查）
[ ]  角色/主体在图中占比 30~70%（不能太小或太大）
[ ]  caption 与图内容匹配（手抽 5 张检查）
[ ]  caption 无拼写错误
[ ]  caption 不含 [主体/角色 ID] 之外的"主观描述"
[ ]  数据集无版权争议（自己拍的/原作/官方）
```

### 六、版权

**重要警告**：
- ❌ 不要用未授权的 AI 生成图训 LoRA 商用
- ✅ 自己拍的/画的可商用
- ✅ 同人作品仅供个人学习
- ✅ 官方立绘/原作截图仅供研究
- 商用 LoRA 时图源必须合法

---

## 8. 失败案例 & 排查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 触发词不起作用 | 触发词是通用词 | 换独特词（`character_xxx`） |
| 触发词起作用但效果混乱 | 数据集混杂 | 重新筛选，统一数据 |
| 风格/角色"死记"训练图 | 过拟合 | 减 step / 降 LR / 加图 |
| 推理时换角度就糊 | 训练图角度单一 | 加多角度图重训 |
| 推理时背景也变了 | 背景是 LoRA 一部分 | 加更多不同背景的图 |
| 强度 1.0 效果仍然弱 | rank/alpha 太小 | rank 4→8，alpha=2×rank |
| 强度 0.5 也太重 | alpha 太大 | alpha=rank 重训 |
| 出现奇怪 artifacts | 数据集有噪声 | 清洗数据集 |
| 出图有黑边/黑底 | 训练图有黑边/黑底 | 重做数据集（透明 BG 时尤为注意） |
| Loss 不下降 | LR 太小 | LR ×2 |
| Loss NaN | 数值问题 | 用 bf16 + betas=(0.9, 0.99) |

---

## 9. 完整工作流示例

### 案例：练"宫崎骏风格"LoRA

```bash
# 1. 准备数据
#    收集 100 张宫崎骏电影的高清截图
#    跨作品（《千与千寻》《龙猫》《起风了》等）
#    多角色（5+ 角色）
#    多构图（特写/中景/远景/全景）

# 2. 准备数据
mkdir -p train_data/ghibli_style
# 把 100 张图拷到 train_data/ghibli_style/
# 每张图写对应 caption（trigger + 主体词）

# 3. 生成 caption（如果还没写）
python gen_captions.py --src train_data/ghibli_style

# 4. 检查 caption（抽 10 张看是否合理）

# 5. 训
python train_lora.py \
    --train_data_dir ./train_data/ghibli_style \
    --output_dir ./output_ghibli \
    --network_rank 16 \
    --network_alpha 32 \
    --learning_rate 3e-4 \
    --max_train_steps 1500 \
    --resolution 512,768 \
    --save_every_n_steps 300

# 6. 转换 LoRA 给 ComfyUI
python lora_to_comfyui.py \
    -i output_ghibli/checkpoint-1500/adapter_model.safetensors \
    -o ghibli_style.safetensors \
    --alpha 32 --rank 16

# 7. 在 ComfyUI 测试
#    prompt: "masterpiece, best quality, anime, ghibli style, 1girl, forest"
#    试 5 个 checkpoint (300/600/900/1200/1500) 找最佳
```

### 案例：练"原神纳西妲"角色 LoRA

```bash
# 1. 准备数据：15 张高质量纳西妲
#    多角度（正/侧/背/3-4/俯仰）
#    多表情（笑/严肃/害羞/战斗）
#    多距离（特写/半身/全身）
#    不同背景（12 室内 + 8 室外 + 5 战斗）

# 2. Caption 模板（每张图按实际写）
masterpiece, best quality, anime, character_nahida, 1girl, solo,
[发型: long hair, twintails, green hair],
[眼睛: green eyes],
[服饰: white dress, gold headdress],
[其他: looking at viewer / smile / serious / ...]

# 3. 训（注意：角色 LoRA 用较低 LR）
python train_lora.py \
    --train_data_dir ./train_data/nahida \
    --output_dir ./output_nahida \
    --network_rank 32 \
    --network_alpha 32 \
    --learning_rate 1e-4 \
    --max_train_steps 600 \
    --resolution 512,768

# 4. 转换 + ComfyUI
#    prompt: "masterpiece, best quality, character_nahida, [新动作/场景/服饰]"
#    strength 0.7~1.0
```

---

## 10. 文档关联

- 通用训练参数详解：[LORA_TIPS.md](LORA_TIPS.md)
- 透明背景合成工具：[composite_bg.py](composite_bg.py)
- Caption 批量生成：[gen_captions.py](gen_captions.py)
- 训练主脚本：[train_lora.py](train_lora.py)
- LoRA 转换工具：[lora_to_comfyui.py](lora_to_comfyui.py)

---

*最后更新：2026-06-08*
