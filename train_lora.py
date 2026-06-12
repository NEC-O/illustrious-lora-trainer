"""
Illustrious/Stable Diffusion LoRA 风格训练脚本
针对 RTX 3060 6GB 显存优化
直接使用 diffusers + peft 微调，无需 kohya-ss/sd-scripts
使用 VAE Latent 空间训练，效率更高效果更好
"""
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from PIL import Image
import numpy as np

# 必须在 import diffusers/xformers 之前设置：禁用 xformers（与副本对齐）
# xformers 的 fp16 attention 在 SDXL+LoRA 上数值不稳，会吐 NaN
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["TRITON_DISABLED"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# 核心库
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict  # LoraConfig 仍需传给 unet.add_adapter；set_* 用于续训回填 LoRA 权重
from safetensors.torch import save_file, load_file
from lora_to_comfyui import diffusers_key_to_comfyui  # 复用 key 转换逻辑
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================== 配置区，只改这里 =====================
# 支持两种模式：
# 1. diffusers 目录: "./model/Illustrious_fp16"
# 2. 完整 checkpoint: "./model/Illustrious_fp16.safetensors" (自动转换)
PRETRAINED_MODEL = "./model/Illustrious-XL-v2.0-FP16-Diffusers"
TRAIN_DATA_DIR = "./train_data/12_style"
OUTPUT_DIR = "./output_lora"
SEED = 42

# 训练超参 - 3060 6G优化 (针对 <2h 完成训练)
RESOLUTION = (768, 1024)  # 高度, 宽度 (从 768x1024 降低 ~50% 像素，提速 40-50%)
BATCH_SIZE = 1
GRAD_ACC = 8
LR_UNET = 2.8e-4
RANK = 4
ALPHA = 8
MAX_STEPS = 1000   # 29 张图 × GRAD_ACC=8 = 232 步/epoch，约 4.3 个 epoch
SAVE_STEP = 400    # 1000 步存 2 个中间检查点
WARMUP_STEPS = 100
# ==========================================================


class StyleDataset:
    """风格训练数据集加载器，支持 VAE Latent 训练"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        resolution: tuple = (512, 768),
        vae: Optional[AutoencoderKL] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution
        self.vae = vae
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 预计算 latents 缓存目录（按分辨率分目录，避免不同分辨率误用老 cache）
        # 例: 512x768 -> .latent_cache_512x768/, 768x1024 -> .latent_cache_768x1024/
        self.cache_dir = self.data_dir / f".latent_cache_{resolution[0]}x{resolution[1]}"
        
        # 支持多种数据格式
        print(f"DEBUG: StyleDataset _load_samples...", flush=True)
        self.samples = self._load_samples()
        print(f"DEBUG: _load_samples returned {len(self.samples)} samples", flush=True)
        logger.info(f"Loaded {len(self.samples)} training samples")
        
        # 检查数据集是否为空
        if len(self.samples) == 0:
            raise ValueError(f"No training samples found in {data_dir}. Please add images or metadata.jsonl")

        print(f"DEBUG: Starting VAE latent preprocessing...", flush=True)
        logger.info(f"Preprocessing {len(self.samples)} images to latents...")

        # ===== 性能优化：启动时一次性预缓存全部 VAE latent (避免训练循环里重复 encode) =====
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for s in self.samples:
            p = self.cache_dir / (Path(s["image"]).stem + ".pt")
            if not p.exists():
                missing.append(s)
        if missing:
            logger.info(f"Encoding {len(missing)} missing latents to {self.cache_dir} ...")
            for s in missing:
                _ = self[self.samples.index(s)]
        else:
            logger.info(f"All {len(self.samples)} latents already cached at {self.cache_dir}")
        # 之后训练时 _encode_image_to_latent 不会再被调用 (cache 命中)
        # =============================================================================
    
    def _load_samples(self) -> List[Dict]:
        """加载训练样本"""
        samples = []
        
        # 优先找 metadata.jsonl
        metadata_file = self.data_dir / "metadata.jsonl"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
        
        # 尝试目录下所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        for ext in image_extensions:
            for img_path in self.data_dir.rglob(f"*{ext}"):
                # 跳过缓存目录
                if ".latent_cache" in str(img_path):
                    continue
                    
                # 尝试找同名 txt 文件作为 caption
                caption_path = img_path.with_suffix('.txt')
                caption = ""
                if caption_path.exists():
                    caption = caption_path.read_text(encoding='utf-8').strip()
                
                samples.append({
                    "image": str(img_path),
                    "caption": caption or "style reference"
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """将图片缩放并中心裁剪到目标分辨率"""
        w, h = image.size
        target_w, target_h = self.resolution
        
        # 计算缩放比例，保持宽高比
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 中心裁剪
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))
        
        return np.array(image)
    
    def _encode_image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """将图片编码为 VAE latent"""
        if self.vae is None:
            raise ValueError("VAE not provided for latent encoding")
        
        with torch.no_grad():
            # 归一化到 [-1, 1]
            image = image.to(dtype=torch.bfloat16, device=self.device)
            latent = self.vae.encode(image).latent_dist.sample()
            # VAE latent 缩放因子 (SD 使用 0.18215)
            latent = latent * 0.18215
        return latent
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图片
        img_path = sample["image"]
        latent_cache_path = self.cache_dir / (Path(img_path).stem + ".pt")

        # ===== 性能优化：VAE latent 预缓存到磁盘 (避免每步重复 VAE encode) =====
        if latent_cache_path.exists():
            latent = torch.load(latent_cache_path, map_location="cpu")
        else:
            image = Image.open(img_path).convert("RGB")
            # 缩放并中心裁剪到目标分辨率
            image_np = self._preprocess_image(image)
            # 转为 tensor
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 127.5 - 1.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
            # 编码到 latent 空间
            latent = self._encode_image_to_latent(image_tensor).squeeze(0)
            # 缓存到 CPU 磁盘 (bf16)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(latent.to(dtype=torch.bfloat16).cpu(), latent_cache_path)
        # ===================================================================

        # Tokenize caption - SDXL 使用两个 tokenizer
        caption = sample.get("caption", "style reference")
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        tokens_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "latent": latent,
            "input_ids": tokens.input_ids.squeeze(0).to(self.device),
            "input_ids_2": tokens_2.input_ids.squeeze(0).to(self.device),
            "attention_mask": tokens.attention_mask.squeeze(0).to(self.device),
            "caption": caption,
        }


def collate_fn(batch):
    """批处理整理函数"""
    return {
        "latent": torch.stack([x["latent"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "input_ids_2": torch.stack([x["input_ids_2"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "caption": [x["caption"] for x in batch],
    }


def resolve_model_path(model_path: str) -> str:
    """
    自动识别并转换模型格式
    支持：
    1. diffusers 目录 -> 直接返回
    2. .safetensors / .ckpt 完整文件 -> 自动转换
    """
    model_path = Path(model_path)
    
    # 已经是目录，直接返回
    if model_path.is_dir():
        logger.info(f"使用 diffusers 目录: {model_path}")
        return str(model_path)
    
    # 是完整 checkpoint 文件，需要转换
    if model_path.suffix in ['.safetensors', '.ckpt', '.pt']:
        # 转换后的目录名
        converted_path = model_path.parent / f"{model_path.stem}_diffusers"
        
        if converted_path.exists():
            logger.info(f"找到已转换的 diffusers 目录: {converted_path}")
            return str(converted_path)
        
        logger.info(f"检测到完整 checkpoint: {model_path}")
        logger.info(f"正在转换为 diffusers 格式: {converted_path}")
        
        convert_model_to_diffusers(str(model_path), str(converted_path))
        
        return str(converted_path)
    
    raise ValueError(f"无法识别的模型路径: {model_path}")


def convert_model_to_diffusers(checkpoint_path: str, output_dir: str):
    """将完整 checkpoint 转换为 diffusers 格式"""
    try:
        from diffusers.pipelines import StableDiffusionPipeline
        import torch
        from safetensors.torch import load_file
        import json
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # 加载 checkpoint
        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        logger.info(f"Loaded {len(state_dict)} keys from checkpoint")
        
        # 打印前几个 key 帮助调试
        for i, key in enumerate(list(state_dict.keys())[:5]):
            logger.info(f"  Key {i}: {key}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分离各组件 - 使用更宽松的匹配规则
        text_encoder_state = {}
        vae_state = {}
        unet_state = {}
        
        for key, value in state_dict.items():
            key_lower = key.lower()
            if "text_encoder" in key_lower or "text_model" in key_lower:
                new_key = key.replace("text_encoder.", "").replace("text_model.", "").replace("text_encoder/", "")
                text_encoder_state[new_key] = value
            elif "vae" in key_lower or "decoder" in key_lower or "encoder" in key_lower:
                new_key = key.replace("vae.", "").replace("vae/", "")
                vae_state[new_key] = value
            elif "unet" in key_lower or "diffusion" in key_lower:
                new_key = key.replace("unet.", "").replace("unet/", "")
                unet_state[new_key] = value
        
        # 打印分离结果
        logger.info(f"Text encoder keys: {len(text_encoder_state)}")
        logger.info(f"VAE keys: {len(vae_state)}")
        logger.info(f"UNet keys: {len(unet_state)}")
        
        # 保存为 safetensors
        if text_encoder_state:
            save_path = os.path.join(output_dir, "text_encoder", "model.safetensors")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            from safetensors.torch import save_file
            save_file(text_encoder_state, save_path)
            logger.info(f"Saved text_encoder to {save_path}")
        else:
            logger.warning("No text encoder keys found!")
        
        if vae_state:
            save_path = os.path.join(output_dir, "vae", "diffusion_pytorch_model.safetensors")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_file(vae_state, save_path)
            logger.info(f"Saved vae to {save_path}")
        else:
            logger.warning("No VAE keys found!")
        
        if unet_state:
            save_path = os.path.join(output_dir, "unet", "diffusion_pytorch_model.safetensors")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_file(unet_state, save_path)
            logger.info(f"Saved unet to {save_path}")
        else:
            logger.warning("No UNet keys found!")
        
        # 检查是否成功转换
        if not unet_state:
            logger.error("转换失败：无法提取 UNet 权重！")
            logger.error("请检查 checkpoint 文件是否完整，或手动转换为 diffusers 格式")
            raise ValueError("No UNet keys extracted from checkpoint")
        
        # 创建必要的配置文件
        # 1. model_index.json
        model_index = {
            "_class_name": "StableDiffusionPipeline",
            "_diffusers_version": "0.26.0",
            "text_encoder": ["text_encoder"],
            "vae": ["vae"],
            "unet": ["unet"],
        }
        
        with open(os.path.join(output_dir, "model_index.json"), "w") as f:
            json.dump(model_index, f, indent=2)
        
        # 2. unet/config.json
        unet_config = {
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.26.0",
            "in_channels": 4,
            "out_channels": 4,
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "block_in_channels": [320, 640, 1280, 1280],
            "block_out_channels": [320, 640, 1280, 1280],
            "layers_per_block": 2,
            "cross_attention_dim": 768,
            "attention_head_dim": 8,
        }
        os.makedirs(os.path.join(output_dir, "unet"), exist_ok=True)
        with open(os.path.join(output_dir, "unet", "config.json"), "w") as f:
            json.dump(unet_config, f, indent=2)
        
        # 3. vae/config.json
        vae_config = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.26.0",
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "block_out_channels": [128, 256, 512, 512],
            "latent_channels": 4,
            "layers_per_block": 2,
        }
        os.makedirs(os.path.join(output_dir, "vae"), exist_ok=True)
        with open(os.path.join(output_dir, "vae", "config.json"), "w") as f:
            json.dump(vae_config, f, indent=2)
        
        # 4. text_encoder/config.json
        text_encoder_config = {
            "_class_name": "CLIPTextModel",
            "_diffusers_version": "0.26.0",
            "vocab_size": 49408,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 77,
        }
        os.makedirs(os.path.join(output_dir, "text_encoder"), exist_ok=True)
        with open(os.path.join(output_dir, "text_encoder", "config.json"), "w") as f:
            json.dump(text_encoder_config, f, indent=2)
        
        # 5. scheduler/scheduler_config.json
        scheduler_config = {
            "_class_name": "DDPMScheduler",
            "_diffusers_version": "0.26.0",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 1,
        }
        os.makedirs(os.path.join(output_dir, "scheduler"), exist_ok=True)
        with open(os.path.join(output_dir, "scheduler", "scheduler_config.json"), "w") as f:
            json.dump(scheduler_config, f, indent=2)
        
        logger.info(f"转换完成: {output_dir}")
        
    except Exception as e:
        logger.error(f"转换失败: {e}", exc_info=True)
        raise


def load_model_and_tokenizer(model_path: str):
    """加载 Stable Diffusion 模型和 tokenizer"""
    print(f"DEBUG: load_model_and_tokenizer starting...", flush=True)
    logger.info(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"DEBUG: Loading tokenizer...", flush=True)
    
    # 加载 tokenizer (CLIP)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )
    
    # 加载 text_encoder (CLIP)
    print(f"DEBUG: Loading text_encoder...", flush=True)
    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"DEBUG: text_encoder loaded", flush=True)
    
    # SDXL 需要获取 text_projection 权重用于 pooled output
    # CLIPTextModel 的 config.text_embed_dim 应该是 768，但 text_projection 会投影到 2048
    text_embed_dim = text_encoder.config.hidden_size  # 768
    text_projection_dim = getattr(text_encoder.config, 'projection_dim', text_embed_dim)
    
    # 直接从 text_encoder 获取投影层
    if hasattr(text_encoder, 'text_projection'):
        text_embed_proj = text_encoder.text_projection
        text_embed_proj = text_embed_proj.to(device, dtype=torch.bfloat16)
        print(f"DEBUG: text_projection shape: {text_embed_proj.shape}")
    else:
        # Fallback: 创建线性投影层
        text_embed_proj = torch.nn.Linear(768, 2048).to(device, dtype=torch.bfloat16)
        print(f"DEBUG: Created random projection layer")
    
    # SDXL 有第二个 text_encoder (OpenCLIP)
    print(f"DEBUG: Loading tokenizer_2 and text_encoder_2...", flush=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer_2",
    )
    try:
        # SDXL 的 text_encoder_2 必须是 CLIPTextModelWithProjection
        # 它会从 hidden state (1280D) 投影到 pooled output (1280D)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"WARNING: text_encoder_2 loading error: {e}", flush=True)
        # Try loading with mismatched sizes ignored
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )
    print(f"DEBUG: text_encoder_2 loaded", flush=True)
    
    # 加载 VAE
    print(f"DEBUG: Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    print(f"DEBUG: VAE loaded", flush=True)

    # 加载 UNet
    print(f"DEBUG: Loading UNet...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"DEBUG: UNet loaded", flush=True)
    
    # 冻结不需要训练的模块
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # 移动到 GPU（bf16 而非 fp16）
    # 原因：fp16 优化器状态 v_t = (1-β₂)·g² 在小梯度时是 denormal → 数值崩
    # bf16 指数位 = fp32 (8 bits)，范围跟 fp32 一样，不会 denormal
    # Ampere (3060) 原生支持 bf16，显存跟 fp16 一样，速度接近 fp16
    text_encoder = text_encoder.to(device, dtype=torch.bfloat16)
    text_encoder_2 = text_encoder_2.to(device, dtype=torch.bfloat16)
    vae = vae.to(device, dtype=torch.bfloat16)
    unet = unet.to(device, dtype=torch.bfloat16)
    
    # 冻结 text_encoder_2
    text_encoder_2.requires_grad_(False)
    
    # VAE 不训练
    vae.eval()
    
    # 为 UNet 添加 LoRA
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
        init_lora_weights="gaussian",  # 副本对齐：rank=4 时 gaussian 比默认 kaiming 更稳
    )
    
    # 注入 LoRA 到 UNet
    # 用 diffusers 原生 add_adapter（不是 peft 的 inject_adapter_in_model），
    # 这样 gradient checkpointing 才能正确处理 LoRA 层的激活重算
    # 否则 peft wrap 在 Linear 外面，checkpointing 重算时丢上下文，反向梯度会出 NaN
    unet.add_adapter(lora_config)

    # 只启用 LoRA 参数的梯度
    for name, param in unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # ===== 整体 bf16 方案 =====
    # 整个 UNet（包括 LoRA 和 base 权重）已统一在 bf16 加载（model loading 阶段）。
    # 选择 bf16 的原因：
    #   - fp16 优化器状态 v_t = (1-β₂)·g² 在小梯度时是 denormal → 3060 上行为不一致 → NaN
    #   - bf16 指数位 = fp32 (8 bits)，范围跟 fp32 一样，不会 denormal
    #   - Ampere (3060) 原生支持 bf16 加速，显存跟 fp16 一样，速度接近 fp16
    # 这里做一次 dtype 一致性检查（理论应该全是 bf16，verify 一下）以便排查回归
    dtypes = set()
    for n, p in unet.named_parameters():
        if p.requires_grad:
            dtypes.add(str(p.dtype))
    print(f"DEBUG: Trainable param dtypes (should be bfloat16): {dtypes}")
    # ==============================================================

    # ===== 性能优化：3060 6G 提速关键 =====
    # 1) Gradient checkpointing: 显存换时间，省 ~30% 显存
    # ⚠️ 在 diffusers 0.37 + peft LoRA 组合上开启 checkpointing 会导致 Step 2 起 NaN！
    #    原因：unet.add_adapter() 内部走 peft.inject_adapter_in_model()，LoRA 套在 Linear 外面。
    #    checkpointing 重算激活时，peft 包装层输入的 requires_grad 是 False，
    #    反向时 autograd 沿着一条 require_grad=False 的路径走到 LoRA 参数 → grad 链断裂 → NaN
    #    修复：开启 checkpointing 后调 enable_input_require_grads()，让 UNet 输入保留 grad
    #    （这是 peft 官方 README 推荐的与 gradient checkpointing 共存方法）
    try:
        unet.enable_gradient_checkpointing()
        # 关键修复：让 UNet 输入 require_grad，peft LoRA 才能正常反向
        if hasattr(unet, "enable_input_require_grads"):
            unet.enable_input_require_grads()
        else:
            # 兼容老 API
            unet.conv_in.requires_grad_(True)
        print("DEBUG: Gradient checkpointing enabled + input_require_grads fixed (peft LoRA compatible)")
    except Exception as e:
        print(f"WARNING: enable_gradient_checkpointing failed: {e}")

    # 2) Attention 加速: 用 PyTorch 2.0+ SDP（无需额外依赖，避免 xformers fp16 数值不稳）
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet.set_attn_processor(AttnProcessor2_0())
        print("DEBUG: SDP (AttnProcessor2_0) enabled")
    except Exception as e2:
        print(f"WARNING: SDP fallback failed: {e2}")
    # 注意：xformers 在 SDXL+LoRA+fp16 组合上会吐 NaN（不兼容 autocast 提升），
    # 副本文件也不启用，这里直接禁用
    # 速度不损失：SDP (AttnProcessor2_0) 用的是 PyTorch 2.0+ 的 scaled_dot_product_attention 底层走 Flash/Memory-Efficient 后端

    # 3) VAE tiling: 进一步省 VAE 显存（预处理时单图能跑到大图）
    try:
        vae.enable_tiling()
        print("DEBUG: VAE tiling enabled")
    except Exception as e:
        print(f"WARNING: VAE tiling failed: {e}")
    # ===================================
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info("Model loading completed successfully")
    
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
        "text_embed_proj": text_embed_proj,
        "vae": vae,
        "unet": unet,
    }


def find_latest_checkpoint(output_dir: str) -> Optional[tuple]:
    """查找最新的 checkpoint，返回 (checkpoint_path, global_step)"""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # 解析 step 数字
    valid_checkpoints = []
    for ckpt in checkpoints:
        name = ckpt.name
        if name == "checkpoint-final":
            continue
        try:
            step_str = name.replace("checkpoint-", "")
            step = int(step_str)
            valid_checkpoints.append((ckpt, step))
        except ValueError:
            continue
    
    if not valid_checkpoints:
        return None
    
    # 返回最新的
    latest = sorted(valid_checkpoints, key=lambda x: x[1])[-1]
    return latest


def load_checkpoint(unet, optimizer, scheduler, checkpoint_path: str, device):
    """加载 checkpoint 权重和优化器状态"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # 加载 peft 格式的 LoRA 权重（save_checkpoint 时用 get_peft_model_state_dict 导出到 checkpoint_dir 根目录的 adapter_model.safetensors）
    # 不用 unet.load_adapter：diffusers 0.37.1 的 UNet2DConditionModel 没有该方法。
    # 改用 peft 的 set_peft_model_state_dict，把权重灌回到已经在初始化时 add_adapter 注入的 LoRA 层。
    adapter_weights_path = Path(checkpoint_path) / "adapter_model.safetensors"
    if adapter_weights_path.exists():
        peft_state = load_file(str(adapter_weights_path))
        set_peft_model_state_dict(unet, peft_state, adapter_name="default")
        logger.info(f"Loaded peft adapter weights from {adapter_weights_path}")
    else:
        raise FileNotFoundError(
            f"未找到 adapter_model.safetensors: {adapter_weights_path}"
        )
    
    # 尝试加载优化器状态
    # ⚠️ torch 2.6+ 默认 weights_only=True，但 AdamW 状态里有 numpy._core 多维数组标量
    #    （exp_avg / exp_avg_sq 用 numpy.float32 等存）。optimizer.pt / scheduler.pt 都是
    #    本训练脚本自己 dump 的，来源可信，显式关掉 weights_only
    optimizer_path = Path(checkpoint_path) / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=False))
        logger.info("Loaded optimizer state")

    # 尝试加载 scheduler 状态
    scheduler_path = Path(checkpoint_path) / "scheduler.pt"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
        logger.info("Loaded scheduler state")
    
    # 解析 global_step
    step_str = Path(checkpoint_path).name.replace("checkpoint-", "")
    global_step = int(step_str)
    
    return global_step


def train(
    model_path: str,
    train_data_dir: str,
    output_dir: str,
    resolution: tuple = (512, 768),
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2.8e-4,
    max_steps: int = 3600,
    save_steps: int = 800,
    log_every_n_steps: int = 10,
    warmup_steps: int = 300,
    seed: int = 42,
    rank: int = 4,
    alpha: int = 8,
    resume: bool = True,
):
    """训练函数"""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print(f"DEBUG: About to call load_model_and_tokenizer...", flush=True)
    models = load_model_and_tokenizer(model_path)
    print(f"DEBUG: load_model_and_tokenizer returned", flush=True)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    tokenizer_2 = models["tokenizer_2"]
    text_encoder_2 = models["text_encoder_2"]
    text_embed_proj = models["text_embed_proj"]
    vae = models["vae"]
    unet = models["unet"]
    
    # 加载数据集
    print(f"DEBUG: Creating StyleDataset...", flush=True)
    dataset = StyleDataset(train_data_dir, tokenizer, tokenizer_2, resolution, vae)
    print(f"DEBUG: StyleDataset created", flush=True)
    logger.info(f"Dataset size: {len(dataset)}")

    # ===== 性能优化：预缓存 text encoder 输出 (CPU 上一次性算完) =====
    # 训练循环中不再调用 text_encoder / text_encoder_2，节省大量前向时间
    print(f"DEBUG: Pre-encoding all captions (text encoders)...", flush=True)
    cache = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        caption = sample.get("caption", "style reference")
        with torch.no_grad():
            t1 = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            t2 = tokenizer_2(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            out1 = text_encoder(t1, output_hidden_states=True, return_dict=True)
            out2 = text_encoder_2(t2, output_hidden_states=True, return_dict=True)
            clip_hidden = out1.hidden_states[-2]                  # 1x77x768
            openclip_hidden = out2.hidden_states[-2]               # 1x77x1280
            pooled = out2.text_embeds                             # 1x1280
            encoder_hidden_states = torch.cat([clip_hidden, openclip_hidden], dim=-1)  # 1x77x2048
        # 转 bf16 并搬到 CPU 节省显存（与 UNet dtype 对齐，避免后续前向类型提升）
        cache.append({
            "encoder_hidden_states": encoder_hidden_states.squeeze(0).to(dtype=torch.bfloat16).cpu(),
            "pooled_text_embeds": pooled.squeeze(0).to(dtype=torch.bfloat16).cpu(),
        })
    logger.info(f"Pre-encoded {len(cache)} caption embeddings cached on CPU")
    # text_encoder 用完可以丢回 CPU 进一步省显存
    text_encoder = text_encoder.to("cpu")
    text_encoder_2 = text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    # ==============================================================
    
    # 优化器 - 只优化 UNet LoRA 参数
    # betas=(0.9, 0.99) 而不是 (0.9, 0.999)：与 kohya_ss SDXL LoRA 训练对齐
    # β₂=0.99 时 v_t = 0.01·g²，更新量在合理范围；β₂=0.999 时 v_t 太小，在 fp16 下是 denormal
    # 详见 debug-lora-nan-loss-step2.md
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # 学习率调度器 - Cosine with restarts
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # 余弦退火
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 噪声调度器 - DDPM
    # 不能用 from_pretrained：模型自带的是 EulerAncestralDiscreteScheduler 配置（含 steps_offset=1），
    # 硬塞给 DDPMScheduler 会导致 add_noise 算出的 noisy_latent 异常，t=999 时直接爆 NaN
    # 直接 new 一个干净的 DDPMScheduler（与副本对齐）
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )
    
    # 检查断点续训
    global_step = 0
    if resume:
        latest = find_latest_checkpoint(output_dir)
        if latest:
            checkpoint_path, global_step = latest
            load_checkpoint(unet, optimizer, scheduler, str(checkpoint_path), device)
            logger.info(f"Resuming from step {global_step}")
    
    # 训练循环
    unet.train()

    logger.info(f"Starting training from step {global_step}...")
    # region debug-point nan-trace
    _dbg_step = 0
    _dbg_log = open("debug-lora-nan-loss-step2.ndjson", "a", encoding="utf-8")
    def _dbg_emit(event, **kw):
        import json, time
        rec = {"t": time.time(), "step": _dbg_step, "event": event, **kw}
        _dbg_log.write(json.dumps(rec, default=str) + "\n")
        _dbg_log.flush()
    # endregion

    while global_step < max_steps:
        _dbg_step = global_step + 1
        epoch_loss = 0.0
        optimizer.zero_grad()
        # region debug-point pre-step
        # check LoRA param count + dtype
        lora_params = [p for n,p in unet.named_parameters() if p.requires_grad]
        _dbg_emit("pre_step", lora_params_count=len(lora_params),
                  first_param_dtype=str(lora_params[0].dtype) if lora_params else None,
                  first_param_has_nan=(torch.isnan(lora_params[0]).any().item() if lora_params else None))
        # endregion

        for _ in range(gradient_accumulation_steps):
            try:
                # 随机采样一个样本
                idx = torch.randint(0, len(dataset), (batch_size,)).item()
                batch = collate_fn([dataset[idx]])

                latent = batch["latent"].to(device, dtype=torch.bfloat16, non_blocking=True)

                # ===== 性能优化：从预缓存中取 conditioning (避免每步重新跑 text encoder) =====
                cached = cache[idx]
                encoder_hidden_states = cached["encoder_hidden_states"].unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
                pooled_text_embeds = cached["pooled_text_embeds"].unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
                # region debug-point enc-out
                _dbg_emit("enc_hs", shape=list(encoder_hidden_states.shape),
                          dtype=str(encoder_hidden_states.dtype),
                          mn=float(encoder_hidden_states.min().item()),
                          mx=float(encoder_hidden_states.max().item()),
                          has_nan=torch.isnan(encoder_hidden_states).any().item(),
                          has_inf=torch.isinf(encoder_hidden_states).any().item())
                _dbg_emit("enc_pool", shape=list(pooled_text_embeds.shape),
                          mn=float(pooled_text_embeds.min().item()),
                          mx=float(pooled_text_embeds.max().item()),
                          has_nan=torch.isnan(pooled_text_embeds).any().item())
                # endregion
                # =============================================================================

                # SDXL 需要正确的 time_ids 格式
                # 格式: [original_height, original_width, crop_top, crop_left, target_height, target_width] = 6 维
                # 分辨率 (512, 768) 表示高度=512, 宽度=768
                original_size = target_size = (resolution[0], resolution[1])  # (height, width)
                crops_coords_top_left = (0, 0)
                time_ids = torch.tensor(
                    [[
                        original_size[0],  # original_height
                        original_size[1],  # original_width
                        crops_coords_top_left[0],  # crop_top
                        crops_coords_top_left[1],  # crop_left
                        target_size[0],   # target_height
                        target_size[1],   # target_width
                    ]] * batch_size,
                    device=device,
                    dtype=torch.float32,  # fp32 防止 SDXL UNet 内部 add 时溢出
                )

                # 生成噪声
                noise = torch.randn_like(latent)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,)
                ).to(device)

                # 添加噪声到 latent
                noisy_latent = noise_scheduler.add_noise(
                    latent, noise, timesteps
                )

                # 预测噪声 - SDXL 需要 added_cond_kwargs
                # autocast(bf16) 配合 UNet bf16 主体，autocast 此时不改变 matmul dtype，
                # 仅保留 LayerNorm/Softmax 的 fp32 提升（bf16 数值范围 = fp32，无 denormal 问题）
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    noise_pred = unet(
                        noisy_latent,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids},
                    ).sample

                # region debug-point unet-out
                _dbg_emit("noise_pred", shape=list(noise_pred.shape),
                          dtype=str(noise_pred.dtype),
                          mn=float(noise_pred.min().item()),
                          mx=float(noise_pred.max().item()),
                          has_nan=torch.isnan(noise_pred).any().item(),
                          has_inf=torch.isinf(noise_pred).any().item(),
                          t=int(timesteps.item()))
                # endregion

                # 计算损失 (loss 必须 fp32：避免 (noise_pred - noise)^2 在 fp16 下溢出 -> NaN)
                loss = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )
                # region debug-point loss
                _dbg_emit("loss", value=float(loss.item()),
                          has_nan=bool(torch.isnan(loss).item()))
                # endregion

                # 梯度累积
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                # region debug-point post-bwd
                first_p = lora_params[0]
                grad_nan = torch.isnan(first_p.grad).any().item() if first_p.grad is not None else None
                grad_inf = torch.isinf(first_p.grad).any().item() if first_p.grad is not None else None
                grad_mx = float(first_p.grad.abs().max().item()) if first_p.grad is not None else None
                _dbg_emit("grad_after_bwd", first_param_nan=grad_nan,
                          first_param_inf=grad_inf, first_param_max_abs=grad_mx)
                # endregion

                epoch_loss += loss.item()

            except Exception as e:
                logger.error(f"Error in training step: {e}", exc_info=True)
                continue

        # 反向传播
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, unet.parameters()),
            1.0
        )
        # region debug-point post-clip
        first_p = lora_params[0]
        grad_nan = torch.isnan(first_p.grad).any().item() if first_p.grad is not None else None
        grad_mx = float(first_p.grad.abs().max().item()) if first_p.grad is not None else None
        _dbg_emit("grad_after_clip", first_param_nan=grad_nan, first_param_max_abs=grad_mx)
        # endregion
        optimizer.step()
        # region debug-point post-step
        first_p = lora_params[0]
        param_nan = torch.isnan(first_p).any().item()
        param_mx = float(first_p.abs().max().item())
        _dbg_emit("param_after_step", first_param_nan=param_nan, first_param_max_abs=param_mx)
        # check optimizer state dtype
        opt_state = optimizer.state.get(0, {}) if 0 in optimizer.state else (next(iter(optimizer.state.values())) if optimizer.state else {})
        exp_avg_sq = opt_state.get("exp_avg_sq", None)
        _dbg_emit("opt_state",
                  state_keys=list(opt_state.keys()) if opt_state else [],
                  exp_avg_sq_dtype=str(exp_avg_sq.dtype) if exp_avg_sq is not None else None,
                  exp_avg_sq_has_nan=(torch.isnan(exp_avg_sq).any().item() if exp_avg_sq is not None else None),
                  exp_avg_has_nan=(torch.isnan(opt_state.get("exp_avg", torch.tensor(0))).any().item() if "exp_avg" in opt_state else None))
        # endregion
        scheduler.step()
        global_step += 1
        
        # 日志（每 log_every_n_steps 输出一次，NaN 排查时仍可临时改成 1）
        avg_loss = epoch_loss / max(gradient_accumulation_steps, 1)
        lr = scheduler.get_last_lr()[0]
        if global_step % log_every_n_steps == 0 or global_step == 1:
            logger.info(
                f"Step {global_step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
        
        # 保存检查点（ComfyUI LoRA + 续训状态，总计 ~70MB）
        if global_step % save_steps == 0:
            save_checkpoint(unet, optimizer, scheduler, output_dir, global_step, tokenizer, alpha=alpha, rank=rank)

    # 保存最终模型
    save_checkpoint(unet, optimizer, scheduler, output_dir, "final", tokenizer, alpha=alpha, rank=rank)
    logger.info("Training completed!")


def save_checkpoint(unet, optimizer, scheduler, output_dir: str, step: str, tokenizer: CLIPTokenizer, alpha: float = 8.0, rank: int = 4, save_resume_state: bool = True):
    """
    保存训练 checkpoint:
      1. lora.safetensors      ComfyUI 通用格式 LoRA（~12MB，可直接拖入 ComfyUI）
      2. optimizer.pt          AdamW 状态（~50MB，仅 LoRA param）
      3. scheduler.pt          LR 调度器状态（<1KB）
      4. tokenizer/, tokenizer_2/  CLIP tokenizer（~5MB）
    单个 checkpoint 总大小: ~70MB（之前 4.79GB，降 70x）
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    lora_path = os.path.join(checkpoint_dir, "lora.safetensors")

    # 1. 导出 ComfyUI LoRA（scaling = alpha/rank 烤进 lora_up）
    scaling = (alpha / rank) if (alpha != 1.0 and rank > 0) else 1.0

    lora_tensors: dict = {}
    n_a = n_b = 0
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            continue
        new_key = diffusers_key_to_comfyui(name)
        if new_key is None:
            continue
        tensor = param.detach().cpu().contiguous()
        if new_key.endswith(".lora_up.weight"):
            tensor = tensor * scaling
            n_b += 1
        else:
            n_a += 1
        lora_tensors[new_key] = tensor

    save_file(lora_tensors, lora_path)

    # 1b. 同步存一份 peft 格式的 adapter_config.json + adapter_model.safetensors
    #     （用于 load_checkpoint 走 unet.load_adapter 续训）
    #     ⚠️ 不要用 unet.save_pretrained(... , subfolder=...)——diffusers 的
    #     save_pretrained 不接受 subfolder，会忽略后把全量 UNet（4.79GB）存到 checkpoint_dir
    #     用 peft 的 get_peft_model_state_dict 只导出 LoRA（~12MB）
    try:
        from peft import get_peft_model_state_dict
        peft_state = {
            k: v.detach().cpu().contiguous()
            for k, v in get_peft_model_state_dict(unet, adapter_name="default").items()
        }
        save_file(peft_state, os.path.join(checkpoint_dir, "adapter_model.safetensors"))
        # peft 0.19 的 LoraConfig.to_dict() 里有 set 字段 (如 target_modules)，需转 list 才能 json.dump
        adapter_cfg = unet.peft_config["default"].to_dict()
        adapter_cfg = {
            k: (sorted(v) if isinstance(v, set) else v)
            for k, v in adapter_cfg.items()
        }
        with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_cfg, f, indent=2)
    except Exception as e:
        logger.warning(f"[Checkpoint] peft adapter save skipped: {e}")

    # 2. 保存续训状态（resumable state，~50MB）
    extra_mb = 0.0
    if save_resume_state:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), opt_path)
        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), sched_path)
        try:
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer_2"))
        except Exception as e:
            logger.warning(f"[Checkpoint] tokenizer save skipped: {e}")
        extra_mb = (
            os.path.getsize(opt_path) / 1024**2
            + os.path.getsize(sched_path) / 1024**2
        )

    lora_mb = os.path.getsize(lora_path) / 1024**2
    total_mb = lora_mb + extra_mb
    logger.info(
        f"[Checkpoint] step={step} | "
        f"lora={lora_mb:.2f}MB (down={n_a}, up={n_b}, scaling={scaling:.2f}) | "
        f"resume={extra_mb:.2f}MB | "
        f"total≈{total_mb:.0f}MB | "
        f"→ {checkpoint_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Illustrious LoRA 训练 (3060 6G优化)")
    
    # 模型与数据
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=PRETRAINED_MODEL)
    parser.add_argument("--train_data_dir", type=str, default=TRAIN_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    
    # 图像配置
    parser.add_argument("--resolution", type=str, default="768,1024")
    
    # 训练超参
    parser.add_argument("--train_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACC)
    parser.add_argument("--learning_rate", type=float, default=LR_UNET)
    parser.add_argument("--max_train_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save_every_n_steps", type=int, default=SAVE_STEP)
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                        help="每多少 step 输出一次训练日志（默认 10，排查 NaN 时可设 1）")
    parser.add_argument("--lr_warmup_steps", type=int, default=WARMUP_STEPS)
    
    # LoRA配置
    parser.add_argument("--network_rank", type=int, default=RANK)
    parser.add_argument("--network_alpha", type=int, default=ALPHA)
    
    # 其他
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="从最新checkpoint恢复训练")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="不恢复训练，从头开始")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 自动识别并转换模型格式
    model_path = resolve_model_path(args.pretrained_model_name_or_path)
    
    # 解析分辨率 (高度, 宽度)
    height, width = map(int, args.resolution.split(','))
    resolution = (height, width)
    
    logger.info("=" * 60)
    logger.info("Illustrious LoRA 训练配置 (3060 6G优化)")
    logger.info("=" * 60)
    logger.info(f"模型路径: {model_path}")
    logger.info(f"数据目录: {args.train_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"分辨率 (HxW): {resolution[0]}x{resolution[1]}")
    logger.info(f"Batch Size: {args.train_batch_size} x {args.gradient_accumulation_steps} (grad accum)")
    logger.info(f"LoRA Rank: {args.network_rank}, Alpha: {args.network_alpha}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"最大步数: {args.max_train_steps}")
    logger.info("=" * 60)
    
    train(
        model_path=model_path,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        resolution=resolution,
        batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_train_steps,
        save_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        rank=args.network_rank,
        alpha=args.network_alpha,
        resume=args.resume,
    )


if __name__ == "__main__":
    import sys
    print("Script starting...", flush=True)
    sys.stdout.flush()
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Fatal error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        input("Press Enter to exit...")
