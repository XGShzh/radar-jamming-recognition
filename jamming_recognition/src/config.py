"""
全局参数配置 - 与 MATLAB 代码参数完全对应
基于深度学习的小样本雷达复合干扰识别
"""
import numpy as np

# ─────────────────────────────────────────────
#  一、雷达与信号参数
# ─────────────────────────────────────────────
FS      = 100e6      # 采样频率 100 MHz
B       = 10e6       # 信号带宽 10 MHz
T_P     = 50e-6      # LFM 脉冲宽度 50 µs
K       = B / T_P    # 调频斜率 0.2 MHz/µs
T_OBS   = 100e-6     # 观测窗口 100 µs
N       = int(T_OBS * FS)           # 10000 点
T_AXIS  = np.arange(N) / FS         # 时间轴 0~100 µs

# ─────────────────────────────────────────────
#  二、STFT 参数
# ─────────────────────────────────────────────
NFFT    = 512
WIN_LEN = 128
HOP_LEN = 8
F_SHOW  = 12e6       # 频率显示范围 ±12 MHz

# ─────────────────────────────────────────────
#  三、干噪比
# ─────────────────────────────────────────────
JNR_TRAIN_DB = 5# 训练/测试 JNR = 15 dB
JNR_VIS_DB   = 18    # 可视化 JNR = 18 dB

# ─────────────────────────────────────────────
#  四、图像尺寸
# ─────────────────────────────────────────────
IMG_SIZE = 224

# ─────────────────────────────────────────────
#  五、干扰类别（★ 先定义两个列表，再合并）
# ─────────────────────────────────────────────
SINGLE_LABELS = ['CI', 'SMSP', 'DDJ', 'DFTJ', 'ISDRJ', 'ISPRJ', 'ISCRJ', 'Comb']

# 索引: 0=CI 1=SMSP 2=DDJ 3=DFTJ 4=ISDRJ 5=ISPRJ 6=ISCRJ 7=Comb
# 原则: 跨类型配对，时频特征差异大，去掉 ISDRJ/ISPRJ/ISCRJ 三者互相组合
COMPOUND_PAIRS = [
    (0, 2),   # CI    + DDJ
    (0, 3),   # CI    + DFTJ
    (0, 4),   # CI    + ISDRJ
    (0, 7),   # CI    + Comb
    (1, 2),   # SMSP  + DDJ
    (1, 3),   # SMSP  + DFTJ
    (1, 6),   # SMSP  + ISCRJ
    (1, 7),   # SMSP  + Comb
    (2, 4),   # DDJ   + ISDRJ
    (2, 5),   # DDJ   + ISPRJ
    (2, 7),   # DDJ   + Comb
    (3, 5),   # DFTJ  + ISPRJ
    (3, 7),   # DFTJ  + Comb
    (4, 7),   # ISDRJ + Comb
    (5, 7),   # ISPRJ + Comb
]

COMPOUND_LABELS = [
    'CI+DDJ', 'CI+DFTJ', 'CI+ISDRJ', 'CI+Comb',
    'SMSP+DDJ', 'SMSP+DFTJ', 'SMSP+ISCRJ', 'SMSP+Comb',
    'DDJ+ISDRJ', 'DDJ+ISPRJ', 'DDJ+Comb',
    'DFTJ+ISPRJ', 'DFTJ+Comb',
    'ISDRJ+Comb', 'ISPRJ+Comb',
]

# ★ ALL_LABELS 必须在 COMPOUND_LABELS 定义之后
ALL_LABELS = SINGLE_LABELS + COMPOUND_LABELS
N_SINGLE   = len(SINGLE_LABELS)    # 8
N_COMPOUND = len(COMPOUND_LABELS)  # 15
N_CLASS    = len(ALL_LABELS)       # 23

# ─────────────────────────────────────────────
#  七、YOLO 目标框 [t_left_µs, f_bot_MHz, t_width_µs, f_height_MHz]
# ─────────────────────────────────────────────
SINGLE_BOXES_TF = {
    'CI':    [ 0,   -5.5, 51, 11.0],
    'SMSP':  [ 0,   -5.5, 51, 11.0],
    'DDJ':   [15,   -5.5, 51, 11.0],
    'DFTJ':  [ 0,   -5.5, 66, 11.0],
    'ISDRJ': [ 0,   -5.5, 51, 11.0],
    'ISPRJ': [ 0,   -5.5, 51, 11.0],
    'ISCRJ': [ 0,   -5.5, 51, 11.0],
    'Comb':  [ 0,   -9.5, 51, 19.0],
}

# ─────────────────────────────────────────────
#  八、训练参数
# ─────────────────────────────────────────────
N_TEST_PER_CLASS  = 1000
FEW_SHOT_STEPS    = [3, 4, 5, 6, 7, 8]
CORE_SHOT         = 6
EPOCHS            = 100
BATCH_SIZE        = 8
LR_INIT           = 1e-4
WEIGHT_DECAY      = 5e-5
PATIENCE          = 20

# ─────────────────────────────────────────────
#  九、路径
# ─────────────────────────────────────────────
DATASET_ROOT     = 'dataset'
WEIGHTS_DIR      = 'weights'
RESULTS_DIR      = 'results'
PRETRAIN_WEIGHTS = 'yolov8n.pt'

FREEZE_LAYERS = 10
