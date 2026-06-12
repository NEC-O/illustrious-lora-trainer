"""
为 train_data 目录里所有图片批量生成同名 .txt caption
（只对还没有 .txt 的图片生效；已有 .txt 的图片不会覆盖）
用法:
    python gen_captions.py

============================================================
Caption 写法参考（按需选用 / 改写 DEFAULT_CAPTION）
============================================================
1) Constant caption（纯风格 LoRA，所有图共用同一段）
   masterpiece, best quality, high quality, ultra detailed,
   anime, anime style, thick paint, impasto, painterly,
   clean lines, soft shading, 1girl, solo

2) Trigger + 主体词（风格 + 适应不同主体）
   masterpiece, best quality, anime style, thick paint,
   impasto, painterly, clean lines, soft shading, <主体>
   例:
     1girl, solo
     1boy, solo
     2girls
     1girl, cat
     landscape
   ——把"1girl, solo"换成与图匹配的主体词即可。

3) Trigger + 简单描述（风格 + 部分主体特征）
   masterpiece, best quality, anime style, thick paint,
   impasto, painterly, clean lines, soft shading,
   1girl, red hair, white dress
   ——在触发词后加少量"主体+属性"描述，让 LoRA 记住
     风格同时不与具体主体强绑定。

建议流程：先用 (1) 训出基线，再实验 (2)/(3) 看效果。
============================================================
"""
from pathlib import Path
import sys

# ====== 在这里改默认 caption ======
# 风格触发词（与当前 train_data/12_style/image_27.txt 一致，可按需改）
DEFAULT_CAPTION = (
    """masterpiece, best quality, ultra detailed, 1girl, solo, shiny skin, soft shading, clean lineart, realistic anime, curvy"""
)
# ===================================

# 训练数据目录
TRAIN_DIR = Path(__file__).parent / "train_data" / "12_style"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    if not TRAIN_DIR.exists():
        print(f"[ERROR] 训练目录不存在: {TRAIN_DIR}")
        sys.exit(1)

    # 递归找所有图片
    images = sorted(p for p in TRAIN_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[ERROR] {TRAIN_DIR} 下没找到图片")
        sys.exit(1)

    created = skipped = overwritten = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists():
            skipped += 1
            continue
        txt.write_text(DEFAULT_CAPTION, encoding="utf-8")
        created += 1
        print(f"  + {txt.relative_to(TRAIN_DIR.parent)}")

    print()
    print(f"图片总数: {len(images)}")
    print(f"新建 .txt: {created}")
    print(f"已有 .txt (跳过): {skipped}")
    print()
    print("已生成的 caption (与现有 image_27.txt 风格一致):")
    print(f"  {DEFAULT_CAPTION}")
    print()
    print("提示: 如果某些图是 boy/2girls 等不同主体，")
    print("      可以手动编辑对应的 .txt，把 '1girl, solo' 改成 '1boy' / '2girls' 等。")


if __name__ == "__main__":
    main()
