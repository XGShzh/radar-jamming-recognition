"""
雷达欺骗干扰信号生成模块 - 参数随机化版
各类干扰的核心参数在合理范围内随机取值，增加样本多样性
"""
import numpy as np
from src.config import *


def gen_LFM(t=None, t0_us=0.0) -> np.ndarray:
    if t is None: t = T_AXIS
    t0 = t0_us * 1e-6
    s = np.zeros(len(t), dtype=complex)
    idx = (t >= t0) & (t <= (t0 + T_P))
    if np.any(idx):
        t_rel = t[idx] - t0
        s[idx] = (np.exp(1j * np.pi * K * t_rel ** 2)
                  * np.exp(-1j * np.pi * B * t_rel))
    return s


def add_noise(s: np.ndarray, jnr_db: float, rng=None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng()
    sp = np.mean(np.abs(s) ** 2)
    if sp == 0: sp = 1.0
    np_var = sp / (10 ** (jnr_db / 10))
    noise = np.sqrt(np_var / 2) * (rng.standard_normal(s.shape)
                                   + 1j * rng.standard_normal(s.shape))
    return s + noise


def gen_CI(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """切割交织 - 切片段数2~4，切片时长8~12μs，间隔10~15μs"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    s_lfm = gen_LFM(t, t0_us)

    nr = int(rng.integers(2, 5))          # 切片段数 2~4
    tc = rng.uniform(8e-6, 12e-6)         # 切片时长 8~12μs
    Tc_gap = rng.uniform(10e-6, 15e-6)    # 切片间隔 10~15μs

    sc = np.zeros(len(t), dtype=complex)
    t0 = t0_us * 1e-6
    for n in range(nr):
        w = (t >= t0 + n * Tc_gap) & (t < t0 + n * Tc_gap + tc)
        sl = s_lfm.copy()
        sl[~w] = 0
        sc += sl
    return sc


def gen_SMSP(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """频谱弥散 - 子脉冲数2~4"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()

    Ns = int(rng.integers(2, 5))          # 子脉冲数 2~4
    Ts = T_P / Ns
    Ks = Ns * K

    ss = np.zeros(len(t), dtype=complex)
    t0 = t0_us * 1e-6
    for i in range(Ns):
        ti = t0 + i * Ts
        w = (t >= ti) & (t < ti + Ts)
        if w.any():
            t_rel = t[w] - ti
            phi = rng.uniform(0, 2 * np.pi)
            ss[w] += np.exp(1j * (np.pi * Ks * t_rel ** 2 + phi))
    return ss / Ns


def gen_DDJ(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """距离欺骗 - 延迟10~20μs（原本就是随机的）"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    td_us = rng.uniform(10, 20)
    return gen_LFM(t, t0_us + td_us)


def gen_DFTJ(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """密集假目标 - 假目标数4~6，间距5~9μs，幅度递减"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    sf = np.zeros(len(t), dtype=complex)

    n_targets = int(rng.integers(4, 7))       # 假目标数 4~6
    spacing = rng.uniform(5.0, 9.0)           # 间距 5~9μs

    delays_us = [i * spacing for i in range(n_targets)]
    # 幅度从1.0线性递减到0.4
    amps = [1.0 - 0.6 * i / (n_targets - 1) for i in range(n_targets)]

    for dl, amp in zip(delays_us, amps):
        sf += amp * gen_LFM(t, t0_us + dl)
    return sf / n_targets


def gen_ISDRJ(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """直接转发 - 切片数8~12，切片时长1.5~3μs，间隔4~6μs"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    s_lfm = gen_LFM(t, t0_us)

    n_slices = int(rng.integers(8, 13))       # 切片数 8~12
    tpj = rng.uniform(1.5e-6, 3e-6)          # 切片时长 1.5~3μs
    t_gap = rng.uniform(4e-6, 6e-6)          # 间隔 4~6μs

    sc = np.zeros(len(t), dtype=complex)
    t0 = t0_us * 1e-6
    for i in range(n_slices):
        w = (t >= t0) & (t < t0 + tpj)
        seg = s_lfm[w]
        d_idx = round((i * t_gap) * FS)
        idx_s = round(t0 * FS) + d_idx
        if idx_s + len(seg) <= len(t):
            sc[idx_s: idx_s + len(seg)] += seg
    return sc


def gen_ISPRJ(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """重复转发 - 重复次数2~4，采样比例0.4~0.6，间隔比例0.5~0.7"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    s_lfm = gen_LFM(t, t0_us)
    t0 = t0_us * 1e-6

    n_repeat = int(rng.integers(2, 5))            # 重复次数 2~4
    sample_ratio = rng.uniform(0.4, 0.6)          # 采样比例
    interval_ratio = rng.uniform(0.5, 0.7)        # 间隔比例

    tpj = T_P * sample_ratio
    t_interval = T_P * interval_ratio

    w = (t >= t0) & (t < t0 + tpj)
    seg = s_lfm[w]
    sc = np.zeros(len(t), dtype=complex)

    for i in range(n_repeat):
        idx_s = round((t0 + i * t_interval) * FS)
        if idx_s + len(seg) <= len(t):
            sc[idx_s: idx_s + len(seg)] += seg
    return sc


def gen_ISCRJ(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """循环转发 - 循环切片数2~4，采样比例0.3~0.5，转发间隔0.8~1.2倍T_P"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    s_lfm = gen_LFM(t, t0_us)
    t0 = t0_us * 1e-6

    Ni = int(rng.integers(2, 5))                  # 循环切片数 2~4
    sample_ratio = rng.uniform(0.3, 0.5)          # 采样比例
    interval_ratio = rng.uniform(0.8, 1.2)        # 转发间隔比例

    tpj = T_P * sample_ratio
    Tsc = T_P * interval_ratio

    # 幅度从1.0线性递减到0.5
    amp_weights = [1.0 - 0.5 * n / (Ni - 1) if Ni > 1 else 1.0 for n in range(Ni)]

    sc2 = np.zeros(len(t), dtype=complex)
    for n in range(Ni):
        seg_start = t0 + n * tpj
        seg_end = t0 + (n + 1) * tpj
        w = (t >= seg_start) & (t < seg_end)
        seg = s_lfm[w]

        idx_s = round((t0 + n * Tsc) * FS)
        if idx_s + len(seg) <= len(t):
            sc2[idx_s: idx_s + len(seg)] += seg * amp_weights[n]
    return sc2


def gen_Comb(t=None, rng=None, t0_us=0.0) -> np.ndarray:
    """梳状谱 - 频点数3~5，频率间隔3~5MHz"""
    if t is None: t = T_AXIS
    if rng is None: rng = np.random.default_rng()
    s_lfm = gen_LFM(t, t0_us)

    n_freq = int(rng.integers(3, 6))              # 频点数 3~5
    f_spacing = rng.uniform(3e6, 5e6)             # 频率间隔 3~5MHz

    # 以0为中心对称排列
    fr = np.array([(i - (n_freq - 1) / 2) * f_spacing for i in range(n_freq)])

    sc3 = np.zeros(len(t), dtype=complex)
    for f in fr:
        sc3 += s_lfm * np.exp(1j * 2 * np.pi * f * (t - t0_us * 1e-6))
    return sc3 / n_freq


# ═══════════════════════════════════════════════
#  调度逻辑
# ═══════════════════════════════════════════════
SINGLE_GENERATORS = {
    'CI': gen_CI, 'SMSP': gen_SMSP, 'DDJ': gen_DDJ, 'DFTJ': gen_DFTJ,
    'ISDRJ': gen_ISDRJ, 'ISPRJ': gen_ISPRJ, 'ISCRJ': gen_ISCRJ, 'Comb': gen_Comb,
}


def gen_single(label: str, jnr_db: float = JNR_TRAIN_DB, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t0_us = rng.uniform(5, 50)
    s = SINGLE_GENERATORS[label](rng=rng, t0_us=t0_us)
    return add_noise(s, jnr_db, rng)


def gen_compound(label1: str, label2: str, jnr_db: float = JNR_TRAIN_DB,
                 a1: float = 0.7, a2: float = 0.7, seed: int = None):
    rng = np.random.default_rng(seed)
    t0_us = rng.uniform(5, 45)
    s1 = SINGLE_GENERATORS[label1](rng=rng, t0_us=t0_us)
    s2 = SINGLE_GENERATORS[label2](rng=rng, t0_us=t0_us)
    s = a1 * s1 + a2 * s2
    return add_noise(s, jnr_db, rng), t0_us, t0_us
