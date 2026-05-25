"""
训练模块：小样本单阶段训练（150轮精简版）
"""
import os, gc, json, time, traceback, glob
from pathlib import Path
import numpy as np
from src.config import *


def _cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _safe_sleep(sec=2.0):
    time.sleep(sec)


def _find_best_pt(run_dir):
    """多路径搜索 best.pt / last.pt"""
    for fname in ['best.pt', 'last.pt']:
        p = os.path.join(run_dir, 'weights', fname)
        if os.path.exists(p):
            print(f"  [权重] 找到: {p}")
            return p

    run_name = os.path.basename(run_dir)
    for root in ['runs/detect', 'runs', 'results', '.']:
        for fname in ['best.pt', 'last.pt']:
            pattern = os.path.join(root, '**', run_name, 'weights', fname)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found = max(matches, key=os.path.getmtime)
                print(f"  [权重] 搜索找到: {found}")
                return found

    for fname in ['best.pt', 'last.pt']:
        matches = glob.glob(os.path.join('**', run_name, 'weights', fname), recursive=True)
        if matches:
            found = max(matches, key=os.path.getmtime)
            print(f"  [权重] 兜底找到: {found}")
            return found

    print(f"  [警告] 未找到权重, 目录: {run_dir}")
    return None


def _get_best_from_model(model):
    """从 YOLO model 对象中提取权重路径"""
    try:
        if hasattr(model, 'trainer'):
            for attr in ['best', 'last']:
                if hasattr(model.trainer, attr):
                    p = str(getattr(model.trainer, attr))
                    if os.path.exists(p):
                        print(f"  [权重] 从model.trainer.{attr}获取: {p}")
                        return p
    except Exception:
        pass
    return None


def _train_with_retry(model, train_kwargs, min_batch=1, retry_on_oom=4):
    current_batch = int(train_kwargs.get("batch", 8))
    last_err = None
    for attempt in range(retry_on_oom + 1):
        try:
            print(f"[训练尝试 {attempt+1}] batch={current_batch}")
            train_kwargs["batch"] = current_batch
            result = model.train(**train_kwargs)
            return result, current_batch
        except RuntimeError as e:
            msg = str(e).lower()
            last_err = e
            oom = ("out of memory" in msg or "cuda out of memory" in msg
                   or "cudnn" in msg or "alloc" in msg)
            if not oom: raise
            print(f"[警告] 显存不足: {e}")
            _cleanup_memory()
            if current_batch <= min_batch: raise
            current_batch = max(min_batch, current_batch // 2)
            print(f"[重试] batch={current_batch}")
            _safe_sleep(3)
        except Exception as e:
            last_err = e; raise
    raise last_err


def train_cbam_yolov8n(
    yaml_path,
    n_shots          = CORE_SHOT,
    epochs           = EPOCHS,
    batch_size       = BATCH_SIZE,
    lr_init          = LR_INIT,
    weight_decay     = WEIGHT_DECAY,
    patience         = PATIENCE,
    pretrain_weights = 'yolov8n.pt',
    freeze_layers    = FREEZE_LAYERS,
    save_dir         = WEIGHTS_DIR,
    device           = '0',
    project          = 'results',
    name             = None,
    workers          = 2,
    imgsz            = 320,
    use_amp          = True,
    cache            = True,
    close_mosaic     = 20,
    save_period      = -1,
    val              = True,
    plots            = True,
    verbose          = False,
):
    from ultralytics import YOLO
    from src.cbam import (
        register_cbam, _write_cbam_yaml,
        _load_pretrained_backbone,
    )

    register_cbam()

    if name is None:
        name = f"cbam_yolov8n_{n_shots}shot"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(project, exist_ok=True)

    train_name = name + '_phase2'

    print(f"\n{'='*60}")
    print(f"  CBAM-YOLOv8n | {n_shots}样本/类 | 150轮")
    print(f"  imgsz={imgsz}  batch=8  cls=8.0  box=7.5")
    print(f"  device={device}")
    print(f"{'='*60}")

    _cleanup_memory()
    _safe_sleep(1)

    yaml_model = _write_cbam_yaml(nc=N_SINGLE)

    model = YOLO(yaml_model)

    if Path(pretrain_weights).exists():
        _load_pretrained_backbone(model, pretrain_weights, device)

    for p in model.model.parameters():
        p.requires_grad = True

    kw = dict(
        data             = yaml_path,
        epochs           = 150,
        batch            = 8,
        imgsz            = imgsz,

        lr0              = 0.01,
        lrf              = 0.01,
        cos_lr           = True,
        warmup_epochs    = 5,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,

        weight_decay     = 0.0005,
        optimizer        = 'AdamW',
        patience         = 50,

        cls              = 8.0,
        box              = 7.5,
        dfl              = 1.5,

        conf             = 0.15,
        iou              = 0.45,

        fliplr           = 0.0,
        flipud           = 0.0,
        degrees          = 0.0,
        hsv_h            = 0.0,
        hsv_s            = 0.0,
        hsv_v            = 0.3,
        mosaic           = 1.0,
        mixup            = 0.0,
        copy_paste       = 0.3,
        translate        = 0.1,
        scale            = 0.3,
        shear            = 0.0,
        erasing          = 0.1,
        perspective      = 0.0,
        label_smoothing  = 0.0,
        close_mosaic     = close_mosaic,
        overlap_mask     = True,

        device           = device,
        workers          = workers,
        project          = project,
        name             = train_name,
        cache            = cache,
        amp              = use_amp,
        verbose          = verbose,
        save             = True,
        save_period      = save_period,
        val              = val,
        plots            = plots,
        pretrained       = False,
        deterministic    = True,
        exist_ok         = True,
    )

    result, used_batch = _train_with_retry(model, kw)

    # 三层搜索权重路径
    best_model = _get_best_from_model(model)

    if best_model is None:
        try:
            if hasattr(result, 'save_dir'):
                sd = str(result.save_dir)
                if os.path.isdir(sd):
                    best_model = _find_best_pt(sd)
        except Exception:
            pass

    if best_model is None:
        best_model = _find_best_pt(os.path.join(project, train_name))

    del model
    _cleanup_memory()
    _safe_sleep(2)

    print(f"\n[训练完成] 最佳权重：{best_model}")
    print(f"[训练信息] batch={used_batch}, imgsz={imgsz}")

    return {
        'n_shots':      n_shots,
        'best_model':   best_model,
        'result_dir':   os.path.join(project, train_name),
        'phase1_batch': used_batch,
        'phase2_batch': used_batch,
    }


def run_few_shot_experiment(
    few_shot_steps    = None,
    dataset_root      = DATASET_ROOT,
    jnr_db            = JNR_TRAIN_DB,
    epochs            = EPOCHS,
    seed              = 42,
    rest_between_runs = 5,
    train_batch_size  = BATCH_SIZE,
    train_workers     = 2,
    train_imgsz       = 320,
    device            = '0',
):
    from src.dataset_builder import build_dataset
    from src.evaluator import evaluate_model

    if few_shot_steps is None:
        few_shot_steps = FEW_SHOT_STEPS

    results = {}

    for n_shots in few_shot_steps:
        print(f"\n{'#'*60}\n  Few-Shot 实验：{n_shots} 样本/类\n{'#'*60}")
        try:
            _cleanup_memory()
            _safe_sleep(2)

            sub_root  = f"{dataset_root}_{n_shots}shot"
            yaml_path = build_dataset(
                n_train_per_class = n_shots,
                n_val_per_class   = max(1, min(n_shots, 10)),
                n_test_per_class  = 100,
                jnr_db            = jnr_db,
                dataset_root      = sub_root,
                seed              = seed,
            )
            _cleanup_memory()
            _safe_sleep(1)

            train_result = train_cbam_yolov8n(
                yaml_path  = yaml_path,
                n_shots    = n_shots,
                epochs     = epochs,
                batch_size = train_batch_size,
                workers    = train_workers,
                imgsz      = train_imgsz,
                device     = device,
            )
            _cleanup_memory()
            _safe_sleep(2)

            metrics = evaluate_model(
                model_path = train_result['best_model'],
                yaml_path  = yaml_path,
                split      = 'test',
            )
            results[n_shots] = {
                **train_result,
                'mAP50':    float(metrics.get('mAP50', 0)),
                'mAP50_95': float(metrics.get('mAP50_95', 0)),
                'status':   'success',
            }
            print(f"  [n={n_shots}] mAP@0.5 = {results[n_shots]['mAP50']:.3f}")

        except Exception as e:
            print(f"[错误] n_shots={n_shots} 失败：{e}")
            traceback.print_exc()
            results[n_shots] = {'n_shots': n_shots, 'status': 'failed', 'error': str(e)}
        finally:
            _cleanup_memory()
            _safe_sleep(rest_between_runs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'few_shot_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Few-Shot] 结果已保存至 {out_path}")
    return results
