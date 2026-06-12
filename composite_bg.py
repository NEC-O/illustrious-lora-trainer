"""
将 train_data 里**透明背景**的人物图自动合成到各种背景上，
生成新的训练集，避免 LoRA 学到"人物 = 无背景/贴纸"。

用法:
    python composite_bg.py
    python composite_bg.py --variants 3     # 每张原图生成 3 张变体（默认 3）
    python composite_bg.py --src my_imgs --out my_imgs_bg

输出: 每个透明图 -> 多个不同背景的合成图 + 同步生成 .txt caption（含 BG tag）
"""
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image


# ============================================
# 调色板：anime/插画风常见背景色调
# ============================================
# 每组 (浅色, 深色)，用于渐变的两端
GRADIENT_PALETTES = [
    # 浅色调（适合明亮角色）
    [(255, 245, 235), (250, 225, 215)],   # 米白
    [(240, 248, 255), (215, 235, 250)],   # 淡蓝
    [(255, 240, 248), (250, 215, 235)],   # 淡粉
    [(245, 255, 240), (215, 240, 220)],   # 淡绿
    [(255, 250, 235), (240, 230, 200)],   # 暖黄
    [(245, 245, 250), (220, 220, 235)],   # 淡紫
    # 中性
    [(248, 248, 245), (215, 215, 210)],   # 米灰
    [(240, 240, 245), (210, 210, 220)],   # 灰蓝
    # 深色调（适合对比强烈的角色）
    [(80, 65, 90), (30, 20, 50)],          # 紫黑
    [(50, 75, 95), (20, 30, 50)],          # 蓝黑
    [(95, 65, 50), (50, 30, 25)],          # 棕
    [(60, 80, 65), (25, 40, 35)],          # 墨绿
]


# ============================================
# 背景生成器
# ============================================
def make_gradient(size: tuple, c1: tuple, c2: tuple, vertical: bool = True) -> np.ndarray:
    """生成两色渐变背景"""
    w, h = size
    if vertical:
        t = np.linspace(0, 1, h).reshape(-1, 1, 1)
    else:
        t = np.linspace(0, 1, w).reshape(1, -1, 1)
    t = np.broadcast_to(t, (h, w, 1))
    arr = np.array(c1).reshape(1, 1, 3) * (1 - t) + np.array(c2).reshape(1, 1, 3) * t
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_solid(color: tuple, size: tuple) -> np.ndarray:
    """生成纯色背景"""
    return (np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array(color)).astype(np.uint8)


def make_noise(base: tuple, size: tuple, level: int = 20) -> np.ndarray:
    """生成带细噪点的纯色背景（增加质感）"""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8) + np.array(base)
    noise = np.random.randint(-level, level, arr.shape, dtype=np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_radial(size: tuple, center_color: tuple, edge_color: tuple) -> np.ndarray:
    """径向渐变（中心亮、边缘暗）"""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist = dist / dist.max()
    dist = dist.reshape(h, w, 1)
    arr = np.array(center_color).reshape(1, 1, 3) * (1 - dist) + np.array(edge_color).reshape(1, 1, 3) * dist
    return np.clip(arr, 0, 255).astype(np.uint8)


# ============================================
# 主流程
# ============================================
def has_transparency(img: Image.Image) -> bool:
    """判断图片是否真的有透明像素"""
    if img.mode != "RGBA":
        return False
    alpha = np.array(img.split()[-1])
    return alpha.min() < 255


def composite(src: Image.Image, bg_arr: np.ndarray, target_size: tuple) -> Image.Image:
    """把透明 PNG 合成到 RGB 背景上"""
    bg = Image.fromarray(bg_arr, "RGB").resize(target_size, Image.LANCZOS)
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    src = src.resize(target_size, Image.LANCZOS)
    bg.paste(src, (0, 0), src)
    return bg


def pick_bg(target_size: tuple) -> tuple[np.ndarray, str]:
    """随机选一种背景，返回 (背景数组, 类型 tag)"""
    bg_type = random.choices(
        ["gradient", "solid", "noise", "radial"],
        weights=[0.50, 0.20, 0.15, 0.15],
    )[0]
    palette = random.choice(GRADIENT_PALETTES)

    if bg_type == "gradient":
        bg = make_gradient(target_size, palette[0], palette[1], vertical=random.random() > 0.3)
        tag = "gradient background"
    elif bg_type == "solid":
        bg = make_solid(random.choice(palette), target_size)
        tag = "simple background"
    elif bg_type == "noise":
        bg = make_noise(random.choice(palette), target_size, level=random.randint(8, 25))
        tag = "simple background"
    else:  # radial
        bg = make_radial(target_size, palette[0], palette[1])
        tag = "gradient background"

    return bg, tag


def parse_args():
    p = argparse.ArgumentParser(description="为透明背景人物图合成多样背景")
    p.add_argument("--src", type=Path, default=Path(__file__).parent / "train_data" / "12_style_no_back",
                   help="源目录（含透明 PNG）")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "train_data" / "12_style_bg",
                   help="输出目录（合成后的图+caption）")
    p.add_argument("--variants", type=int, default=3, help="每张原图生成的变体数（默认 3）")
    p.add_argument("--size", type=int, nargs=2, default=[512, 768],
                   help="目标分辨率 (宽 高)，与 train_lora.py 的 --resolution 对齐")
    p.add_argument("--base-caption", type=str,
                   default="masterpiece, best quality, high quality, ultra detailed, "
                           "anime, anime style, 1girl, solo",
                   help="基础 caption（不含 BG tag，会按背景类型自动追加）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（便于复现）")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    src_dir: Path = args.src
    out_dir: Path = args.out
    target_size: tuple = tuple(args.size)  # (W, H)
    variants: int = args.variants

    if not src_dir.exists():
        print(f"[ERROR] 源目录不存在: {src_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src_dir.glob("*.png"))
    if not pngs:
        print(f"[ERROR] {src_dir} 下没找到 PNG")
        return

    print(f"源目录: {src_dir}")
    print(f"输出目录: {out_dir}")
    print(f"目标分辨率: {target_size[0]}x{target_size[1]}")
    print(f"每张原图变体数: {variants}")
    print()

    total_made = 0
    skipped = 0
    for png in pngs:
        img = Image.open(png)
        if not has_transparency(img):
            skipped += 1
            continue

        # 读已有 caption 作为基础（如果有）
        existing_txt = png.with_suffix(".txt")
        base_caption = args.base_caption
        if existing_txt.exists():
            base_caption = existing_txt.read_text(encoding="utf-8").strip()

        for v in range(1, variants + 1):
            bg_arr, bg_tag = pick_bg(target_size)
            result = composite(img, bg_arr, target_size)

            out_name = f"{png.stem}_bg{v}.png"
            out_path = out_dir / out_name
            result.save(out_path, "PNG")

            # 同步写 caption
            out_caption = out_dir / f"{png.stem}_bg{v}.txt"
            caption = f"{base_caption}, {bg_tag}"
            out_caption.write_text(caption, encoding="utf-8")

            print(f"  + {out_name:30s}  [{bg_tag}]")
            total_made += 1

    print()
    print("=" * 50)
    print(f"原图（透明）: {len(pngs)} 张, 跳过（不透明）: {skipped} 张")
    print(f"生成合成图:   {total_made} 张")
    print(f"输出目录:     {out_dir}")
    print("=" * 50)
    print()
    print("下一步:")
    print(f"  1. 看 {out_dir.name}/ 里合成效果")
    print(f"  2. 把 run_train.bat 的 --train_data_dir 改成这个新目录")
    print(f"  3. 跑 run_train.bat 开始训练")


if __name__ == "__main__":
    main()
