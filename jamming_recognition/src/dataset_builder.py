"""
数据集构建模块 - 小样本重复利用版
每个原始样本生成 N_REPEAT 个变体副本（不同噪声、位置、JNR）
"""
import os
import numpy as np
from scipy.signal import spectrogram
from PIL import Image
from tqdm import tqdm
import json
import cv2

from src.config import *
from src.signal_gen import (
    gen_single, gen_compound, SINGLE_GENERATORS, add_noise, gen_LFM
)

# ══════════════════════════════════════════════
#  每个样本重复生成次数
# ══════════════════════════════════════════════
N_REPEAT = 3

# ══════════════════════════════════════════════
#  动态标签配置
# ══════════════════════════════════════════════
_TF_BOX_DIM = {
    'CI':    [35.0, 20.0, 0.0],    # 加宽：最多4段×15μs间隔
    'SMSP':  [55.0, 20.0, 0.0],
    'DDJ':   [50.0, 20.0, 15.0],
    'DFTJ':  [75.0, 20.0, 0.0],   # 加宽：最多6个×9μs间距=45μs+T_P
    'ISDRJ': [55.0, 20.0, 0.0],
    'ISPRJ': [85.0, 20.0, 0.0],
    'ISCRJ': [80.0, 20.0, 0.0],
    'Comb':  [50.0, 28.0, 0.0],   # 频率加高：最多5频点×5MHz
}


# ══════════════════════════════════════════════
#  增强策略参数
# ══════════════════════════════════════════════
_GAMMA_PARAMS = {
    'CI':    0.45, 'SMSP':  0.50, 'ISCRJ': 0.40,
    'ISDRJ': 0.7,  'ISPRJ': 0.7,
    'DFTJ':  0.45,   # ★ gamma 降到 0.45，拉亮密集斜线束
}

_JNR_BOOST = {
    'CI': 0, 'SMSP': 0, 'ISCRJ': 0,
    'ISDRJ': 0, 'ISPRJ': 0,
    'DFTJ':  0,    # ★ JNR 提高到 6，增强信噪比
}


def apply_gamma(img_u8: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_u8, table)


def signal_to_tfimage(s: np.ndarray, label: str = None) -> np.ndarray:
    f, tt, Zxx = spectrogram(
        s, fs=FS, window=('hamming',),
        nperseg=WIN_LEN, noverlap=WIN_LEN - HOP_LEN,
        nfft=NFFT, return_onesided=False, scaling='spectrum',
    )
    f   = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(np.abs(Zxx), axes=0)

    mask = (f >= -F_SHOW) & (f <= F_SHOW)
    Sm = Zxx[mask, :] ** 0.5

    Sm = (Sm - Sm.min()) / (Sm.max() - Sm.min() + 1e-12)
    img_u8 = (np.flipud(Sm) * 255).astype(np.uint8)

    target_gamma = _GAMMA_PARAMS.get(label, 1.0)
    if target_gamma != 1.0:
        img_u8 = apply_gamma(img_u8, target_gamma)

    img = Image.fromarray(img_u8, mode='L')
    return np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def get_dynamic_yolo_box(label, t0_us):
    w_us, h_mhz, d_us = _TF_BOX_DIM.get(label, [22.0, 20.0, 0.0])

    t_start = t0_us + d_us - 2.0
    f_bottom = -h_mhz / 2

    t_range = 100.0
    f_total = 24.0

    x_center = (t_start + w_us / 2) / t_range
    y_center = 1.0 - (f_bottom + h_mhz / 2 + 12.0) / f_total
    w_norm = w_us / t_range
    h_norm = h_mhz / f_total

    return [np.clip(x_center, 0, 1), np.clip(y_center, 0, 1),
            np.clip(w_norm, 0.01, 1), np.clip(h_norm, 0.01, 1)]


# ═══════════════════════════════════════════════
#  主构建逻辑
# ═══════════════════════════════════════════════

def build_dataset(n_train_per_class=CORE_SHOT, n_val_per_class=None,
                  n_test_per_class=N_TEST_PER_CLASS, jnr_db=JNR_TRAIN_DB,
                  dataset_root=DATASET_ROOT, seed=42):

    if n_val_per_class is None:
        n_val_per_class = 50

    rng_m = np.random.default_rng(seed)
    splits = {'train': n_train_per_class, 'val': n_val_per_class, 'test': n_test_per_class}

    for split in splits:
        os.makedirs(os.path.join(dataset_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'labels', split), exist_ok=True)

    sample_idx = 0
    for split, n_samples in splits.items():
        repeat = N_REPEAT if split == 'train' else 1
        n_actual = n_samples * repeat

        print(f"\n[DatasetBuilder] 生成 {split} 集"
              f"（{n_samples}样本 × {repeat}副本 = {n_actual}张/类）...")

        # 1. 单一干扰
        for cls_idx, lbl in enumerate(SINGLE_LABELS):
            jnr_boost = _JNR_BOOST.get(lbl, 0.0) if split == 'train' else 0.0

            for k in tqdm(range(n_actual), desc=f"  {lbl}", leave=False):
                local_seed = int(rng_m.integers(0, 1_000_000))
                rng = np.random.default_rng(local_seed)

                if lbl == 'DDJ':
                    t0_us = rng.uniform(5, 30)
                elif lbl == 'DFTJ':
                    t0_us = rng.uniform(5, 20)   # ★ DFTJ 跨度大，限制起始位置
                else:
                    t0_us = rng.uniform(5, 45)

                if split == 'train' and n_actual > 1:
                    jnr_jitter = -3.0 + 6.0 * k / (n_actual - 1)
                else:
                    jnr_jitter = 0.0

                s_raw = SINGLE_GENERATORS[lbl](rng=rng, t0_us=t0_us)
                s = add_noise(s_raw, jnr_db + jnr_boost + jnr_jitter, rng)
                img = signal_to_tfimage(s, label=lbl)

                box = get_dynamic_yolo_box(lbl, t0_us)
                label_line = f"{cls_idx} {' '.join([f'{x:.6f}' for x in box])}"

                fname = f"{split}_{lbl}_{k:05d}"
                img_path = os.path.join(dataset_root, 'images', split, fname + '.png')
                lbl_path = os.path.join(dataset_root, 'labels', split, fname + '.txt')

                Image.fromarray(img).convert('RGB').save(img_path)
                with open(lbl_path, 'w') as f:
                    f.write(label_line)
                sample_idx += 1

        # 2. 复合干扰
        for cpd_idx, (i1, i2) in enumerate(COMPOUND_PAIRS):
            lbl1, lbl2 = SINGLE_LABELS[i1], SINGLE_LABELS[i2]
            cpd_lbl = COMPOUND_LABELS[cpd_idx]

            for k in tqdm(range(n_actual), desc=f"  {cpd_lbl[:12]}", leave=False):
                local_seed = int(rng_m.integers(0, 1_000_000))
                jnr_boost = max(_JNR_BOOST.get(lbl1, 0), _JNR_BOOST.get(lbl2, 0)) \
                            if split == 'train' else 0
                jnr_jitter = rng_m.uniform(-2, 2) if split != 'test' else 0

                s, t0_1, t0_2 = gen_compound(
                    lbl1, lbl2,
                    jnr_db + jnr_boost + jnr_jitter,
                    seed=local_seed
                )

                gamma_lbl = lbl1 if _GAMMA_PARAMS.get(lbl1, 1) < _GAMMA_PARAMS.get(lbl2, 1) else lbl2
                img = signal_to_tfimage(s, label=gamma_lbl)

                box1 = get_dynamic_yolo_box(lbl1, t0_1)
                box2 = get_dynamic_yolo_box(lbl2, t0_2)
                label_lines = [
                    f"{i1} {' '.join([f'{x:.6f}' for x in box1])}",
                    f"{i2} {' '.join([f'{x:.6f}' for x in box2])}",
                ]

                fname = f"{split}_{cpd_lbl.replace('+', '_')}_{k:05d}"
                img_path = os.path.join(dataset_root, 'images', split, fname + '.png')
                lbl_path = os.path.join(dataset_root, 'labels', split, fname + '.txt')

                Image.fromarray(img).convert('RGB').save(img_path)
                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                sample_idx += 1

        # 3. 噪声负样本
        if split in ('train', 'val'):
            n_neg = n_actual
            for k in range(n_neg):
                img_n = signal_to_tfimage(
                    add_noise(np.zeros_like(T_AXIS), jnr_db, rng_m))
                fname = f"{split}_noise_{k:05d}"
                Image.fromarray(img_n).convert('RGB').save(
                    os.path.join(dataset_root, 'images', split, fname + '.png'))
                open(os.path.join(dataset_root, 'labels', split, fname + '.txt'), 'w').close()
                sample_idx += 1

    _write_yaml(os.path.join(dataset_root, 'dataset.yaml'), dataset_root)
    print(f"\n[DatasetBuilder] 完成！总样本：{sample_idx}")
    print(f"  训练集：{n_train_per_class}样本 × {N_REPEAT}副本 = {n_train_per_class * N_REPEAT}张/类")
    return os.path.join(dataset_root, 'dataset.yaml')


def _write_yaml(path, root):
    abs_root = os.path.abspath(root)
    with open(path, 'w') as f:
        f.write(f"path: {abs_root}\ntrain: images/train\nval: images/val\ntest: images/test\n\n")
        f.write(f"nc: {N_SINGLE}\nnames:\n")
        for lbl in SINGLE_LABELS:
            f.write(f"  - {lbl}\n")
