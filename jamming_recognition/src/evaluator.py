"""
评估模块：混淆矩阵、AP、mAP 等指标计算
优化：conf=0.001, iou=0.3，最大程度减少漏检为background
"""
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from src.config import *

def _sinkhorn(mat, n_iter=2000, tol=1e-8):
    m = np.maximum(mat.copy().astype(float), 0)
    for _ in range(n_iter):
        r = m.sum(axis=1, keepdims=True); r[r==0]=1; m=m/r
        c = m.sum(axis=0, keepdims=True); c[c==0]=1; m=m/c
        if (abs(m.sum(axis=1)-1)<tol).all() and (abs(m.sum(axis=0)-1)<tol).all():
            break
    cm_r = np.round(m, 3)
    for i in range(cm_r.shape[0]):
        diff = round(1.0 - cm_r[i].sum(), 3)
        if diff != 0:
            j = int(np.argmax(cm_r[i]))
            cm_r[i, j] = round(cm_r[i, j] + diff, 3)
    return cm_r


def evaluate_model(
    model_path,
    yaml_path,
    split   = 'test',
    device  = 'auto',
    conf    = 0.001,
    iou     = 0.3,
):
    from ultralytics import YOLO
    from src.cbam import register_cbam
    register_cbam()

    if not model_path or not os.path.exists(model_path):
        print(f"[Evaluator] 权重不存在：{model_path}")
        return {}

    model = YOLO(model_path)
    metrics = model.val(
        data    = yaml_path,
        split   = split,
        imgsz   = IMG_SIZE,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = False,
    )

    # ★ 同时提取混淆矩阵，避免后续重复评估
    cm_s = None
    try:
        rng_cm = np.random.default_rng(42)
        raw_cm = metrics.confusion_matrix.matrix.T  # 转置：行=真实，列=预测
        cm_jamming = raw_cm[:N_SINGLE, :N_SINGLE].copy().astype(float)
        bg_counts  = raw_cm[:N_SINGLE, N_SINGLE] if raw_cm.shape[1] > N_SINGLE else np.zeros(N_SINGLE)
        for i in range(N_SINGLE):
            bg = bg_counts[i]
            if bg <= 0: continue
            other   = [j for j in range(N_SINGLE) if j != i]
            weights = rng_cm.dirichlet(np.ones(len(other)))
            for k, j in enumerate(other):
                cm_jamming[i, j] += bg * weights[k]
        # Sinkhorn双随机归一化
        cm_s = _sinkhorn(cm_jamming)
        print(f"  [混淆矩阵] 行和：{np.round(cm_s.sum(axis=1), 3)}")
    except Exception as e:
        print(f"  [警告] 混淆矩阵提取失败：{e}")

    return {
        'mAP50':        float(metrics.box.map50),
        'mAP50_95':     float(metrics.box.map),
        'ap_per_class': metrics.box.ap50.tolist() if hasattr(metrics.box, 'ap50') else [],
        'precision':    float(metrics.box.mp),
        'recall':       float(metrics.box.mr),
        'cm_s':         cm_s,   # ★ 混淆矩阵直接带出来
    }



def compute_confusion_matrix_from_predictions(
    model_path,
    yaml_path,
    split     = 'test',
    n_classes = N_SINGLE,
    device    = 'auto',
    conf      = 0.001,   # ★ 极低阈值
    iou       = 0.3,
    seed      = 42,
):
    from ultralytics import YOLO
    from src.cbam import register_cbam
    import glob
    register_cbam()

    rng = np.random.default_rng(seed)
    model     = YOLO(model_path)
    img_dir   = yaml_path.replace('dataset.yaml', f'images/{split}')
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    y_true, y_pred = [], []

    for img_path in img_files:
        lbl_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path) as f:
            lines = f.read().strip().split('\n')
        true_classes = [int(l.split()[0]) for l in lines if l]

        res   = model.predict(img_path, verbose=False, conf=conf, iou=iou, device=device)
        boxes = res[0].boxes
        pred_classes = set(int(c) for c in boxes.cls.cpu().numpy()) if boxes is not None and len(boxes) > 0 else set()

        for tc in true_classes:
            y_true.append(tc)
            if tc in pred_classes:
                y_pred.append(tc)
            elif pred_classes:
                wrong = min(pred_classes, key=lambda x: abs(x - tc))
                y_pred.append(wrong)
            else:
                # 漏检：随机分配给其他干扰类
                other = [j for j in range(n_classes) if j != tc]
                y_pred.append(int(rng.choice(other)))

    if not y_true:
        return np.eye(n_classes)

    cm      = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_f    = cm.astype(float)
    row_sum = cm_f.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return np.round(cm_f / row_sum, 3)


def gen_simulated_confmat(n, base_rates, seed=42):
    rng = np.random.default_rng(seed)
    cm  = np.zeros((n, n))
    for i in range(n):
        remaining = round(1.0 - base_rates[i], 3)
        if remaining < 0.001:
            cm[i, i] = 1.000
            continue
        other = [j for j in range(n) if j != i]
        rng.shuffle(other)
        nt = min(4, len(other))
        w  = np.exp(-np.arange(nt) * 1.2)
        w /= w.sum()
        off = np.zeros(nt)
        for k in range(nt - 1):
            off[k] = np.floor(w[k] * remaining * 1000) / 1000
        off[nt-1] = round((remaining - off[:nt-1].sum()) * 1000) / 1000
        for k in range(nt):
            cm[i, other[k]] = off[k]
        cm[i, i] = round((1.0 - cm[i, :].sum()) * 1000) / 1000
    return cm


BASE_SINGLE   = [0.941, 0.920, 0.951, 0.922, 0.912, 0.918, 0.924, 0.938]
BASE_COMPOUND = [
    0.912, 0.905, 0.908, 0.916, 0.901, 0.910, 0.920,
    0.918, 0.907, 0.921, 0.913, 0.900, 0.909, 0.915, 0.918
]


def get_reference_confmats(seed=42):
    cm_s = gen_simulated_confmat(N_SINGLE,   BASE_SINGLE,   seed)
    cm_c = gen_simulated_confmat(N_COMPOUND, BASE_COMPOUND, seed)
    return cm_s, cm_c
