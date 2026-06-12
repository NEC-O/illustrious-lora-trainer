#!/usr/bin/env python3
"""
将 diffusers 格式的 LoRA（随全量 UNet state_dict 一起存的）转换为 ComfyUI 能直接
使用的 kohya_ss 格式（也是 ComfyUI Load LoRA 节点期望的标准格式）。

输入:   diffusers 风格 safetensors（包含全量 UNet + 嵌入的 LoRA 权重）
        例如:  .../checkpoint-400/diffusion_pytorch_model.safetensors
        key 样例: down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight
                  down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight

输出:   kohya / ComfyUI 通用 safetensors（仅含 LoRA 权重）
        key 样例: lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
                  lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight

用法:
    python lora_to_comfyui.py -i input.safetensors -o output.safetensors --alpha 8 --rank 4
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def diffusers_key_to_comfyui(key: str) -> str | None:
    """
    把一个 diffusers 风格的 LoRA key 转换为 kohya/ComfyUI 风格。
    非 LoRA key 返回 None（被调用方跳过）。
    """
    if not key.endswith(".weight"):
        return None

    # 形如: <module_path>.lora_A.default.weight
    # 形如: <module_path>.lora_B.default.weight
    if ".lora_A.default" in key:
        module_path = key[: -len(".weight")]
        module_path = module_path.replace(".lora_A.default", "")
        module_path = module_path.replace(".", "_")
        return f"lora_unet_{module_path}.lora_down.weight"
    elif ".lora_B.default" in key:
        module_path = key[: -len(".weight")]
        module_path = module_path.replace(".lora_B.default", "")
        module_path = module_path.replace(".", "_")
        return f"lora_unet_{module_path}.lora_up.weight"
    return None


def convert_checkpoint(
    input_path: str,
    output_path: str,
    alpha: float = 8.0,
    rank: int = 4,
    output_dtype: str = "float16",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # LoRA scaling = alpha / rank。前向时 lora_B @ lora_A 整体乘 scaling。
    # 训练代码里 alpha=8, rank=4, 所以 scaling=2.0。把 scaling 烤进 lora_B 里，
    # 输出 LoRA 在 ComfyUI 加载时按 alpha=1 隐式处理，数值上完全等价。
    scaling = (alpha / rank) if (alpha != 1.0 and rank > 0) else 1.0

    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[output_dtype]

    lora_tensors: dict[str, torch.Tensor] = {}
    skipped = 0
    example_old: str | None = None

    with safe_open(str(input_path), framework="pt") as f:
        all_keys = list(f.keys())
        for key in all_keys:
            new_key = diffusers_key_to_comfyui(key)
            if new_key is None:
                skipped += 1
                continue
            tensor = f.get_tensor(key)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            lora_tensors[new_key] = tensor
            if example_old is None and "lora_A" in key:
                example_old = key

    # 把 scaling 烤进 lora_B (即 lora_up)：
    #   原前向:  y = (B @ A) * scaling
    #   等价于:  y = (B_scaled @ A),  其中 B_scaled = B * scaling
    if scaling != 1.0:
        for k in list(lora_tensors.keys()):
            if k.endswith(".lora_up.weight"):
                lora_tensors[k] = lora_tensors[k] * scaling

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_tensors, str(output_path))

    # 报告
    print("=" * 60)
    print("LoRA → ComfyUI 转换报告")
    print("=" * 60)
    print(f"输入文件:       {input_path}")
    print(f"输出文件:       {output_path}")
    print(f"总 key 数:      {len(all_keys)}")
    print(f"LoRA key 数:    {len(lora_tensors)} (lora_A + lora_B)")
    print(f"跳过 key 数:    {skipped} (base UNet / 其它非 LoRA)")
    print(f"alpha / rank:   {alpha} / {rank} = scaling {scaling}")
    print(f"scaling 烤进:   lora_up (lora_B)，前向数值完全等价")
    print(f"输出 dtype:     {output_dtype}")
    if example_old:
        print(f"映射示例:       {example_old}")
        print(f"             → {diffusers_key_to_comfyui(example_old)}")
    in_gb = input_path.stat().st_size / 1024**3
    out_mb = output_path.stat().st_size / 1024**2
    print(f"输入大小:       {in_gb:.2f} GB")
    print(f"输出大小:       {out_mb:.2f} MB")
    print(f"大小压缩:       {in_gb * 1024 / max(out_mb, 0.01):.0f}x")
    print("=" * 60)
    print("转换完成！把输出 .safetensors 丢进 ComfyUI 的 models/loras/ 目录即可。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 diffusers 风格 LoRA 转换为 ComfyUI kohya 通用格式"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入 safetensors 路径（含全量 UNet+LoRA，如 checkpoint-XXX/diffusion_pytorch_model.safetensors）",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出 safetensors 路径（仅 LoRA，可直接拖入 ComfyUI）",
    )
    parser.add_argument(
        "--alpha", type=float, default=8.0,
        help="LoRA alpha（默认 8，对齐 train_lora.py 的 LoraConfig.lora_alpha=8）",
    )
    parser.add_argument(
        "--rank", type=int, default=4,
        help="LoRA rank（默认 4，对齐 train_lora.py 的 LoraConfig.r=4）",
    )
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="输出 dtype（默认 float16，ComfyUI 兼容性最好）",
    )
    args = parser.parse_args()

    convert_checkpoint(
        input_path=args.input,
        output_path=args.output,
        alpha=args.alpha,
        rank=args.rank,
        output_dtype=args.dtype,
    )
