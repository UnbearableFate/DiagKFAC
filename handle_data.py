import os
import shutil
from pathlib import Path
import scipy.io as sio

# 配置路径（按你的实际路径改）
ROOT = Path("/work/xg24i002/share/datasets/imagenet1k")
VAL_DIR = ROOT / "val"
DEVKIT = ROOT / "ILSVRC2012_devkit_t12" / "data"
GT_TXT = DEVKIT / "ILSVRC2012_validation_ground_truth.txt"
META_MAT = DEVKIT / "meta.mat"

assert VAL_DIR.exists(), f"Val folder not found: {VAL_DIR}"
assert GT_TXT.exists(), f"Ground-truth not found: {GT_TXT}"
assert META_MAT.exists(), f"meta.mat not found: {META_MAT}"

print("Loading meta.mat ...")
mat = sio.loadmat(str(META_MAT), squeeze_me=True)
synsets = mat["synsets"]  # structured array

# 构建 index(1..1000) -> wnid 映射（按 ILSVRC2012_ID 排序）
wnids = [None] * 1000
for entry in synsets:
    # 兼容不同字段访问（有的版本字段名稍有不同）
    ilsvrc_id = int(entry["ILSVRC2012_ID"])
    if 1 <= ilsvrc_id <= 1000:
        wnid = entry["WNID"]
        # entry["WNID"] 可能是形如 'n01440764' 或 array(['n01440764'], dtype='<U8')
        if isinstance(wnid, (list, tuple)):
            wnid = wnid[0]
        if hasattr(wnid, "item"):
            wnid = wnid.item()
        wnids[ilsvrc_id - 1] = str(wnid)

assert all(wnids), "Some wnids are missing—check meta.mat parsing."

print("Loading validation ground-truth ...")
with open(GT_TXT, "r") as f:
    gt_labels = [int(x.strip()) for x in f if x.strip()]

assert len(gt_labels) == 50000, f"Expect 50000 labels, got {len(gt_labels)}"

# 依次移动 ILSVRC2012_val_00000001.JPEG ... -> 对应 wnid 子目录
print("Reorganizing val/ into class subfolders ...")
moved, missing = 0, 0
for i, lab in enumerate(gt_labels, start=1):
    wnid = wnids[lab - 1]  # label 是 1-based
    src = VAL_DIR / f"ILSVRC2012_val_{i:08d}.JPEG"
    if not src.exists():
        missing += 1
        if missing < 10:
            print(f"[WARN] Missing file: {src.name}")
        continue
    dst_dir = VAL_DIR / wnid
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.move(str(src), str(dst))
    moved += 1
    if moved % 5000 == 0:
        print(f"  moved {moved} / 50000")

print(f"Done. Moved={moved}, Missing={missing}")
print("Sample class dir:", wnids[0], "->", (VAL_DIR / wnids[0]).as_posix())