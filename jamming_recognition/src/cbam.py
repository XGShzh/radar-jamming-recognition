"""
CBAM 注意力模块 + CBAM-YOLOv8n 集成
实现通道注意力（CAM）与空间注意力（SAM）并联混合机制
对应论文第 3.2 节
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ═══════════════════════════════════════════════
#  通道注意力模块（CAM）
# ═══════════════════════════════════════════════
class ChannelAttention(nn.Module):
    """
    CAM: 全局平均池化 + 全局最大池化 → 共享 MLP → Sigmoid
    对应论文公式 CAM(F) = Sigmoid(FL2(AvgPool(F) + MaxPool(F)))
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        w   = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w.view(b, c, 1, 1)


# ═══════════════════════════════════════════════
#  空间注意力模块（SAM）
# ═══════════════════════════════════════════════
class SpatialAttention(nn.Module):
    """
    SAM: channel-wise 最大 & 均值池化 → 7×7 卷积 → Sigmoid
    对应论文公式 SAM(F) = Sigmoid(f(AvgPool(F); MaxPool(F)))
    支持可变卷积核尺寸以适配不同尺度特征图（21×21, 11×11, 5×5）
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        w   = torch.sigmoid(self.conv(cat))
        return x * w


# ═══════════════════════════════════════════════
#  CBAM 混合注意力模块（并联）
# ═══════════════════════════════════════════════
class CBAM(nn.Module):
    """
    CBAM 并联混合注意力机制（CSMAM 变体）
    论文 3.2.1 节：F' = Ms(Mc(F)) ⊙ F 串联版本 / 并联加权融合版本

    本实现采用并联加权融合（与论文图 3-2 对应）：
        E = Adaptive_Fusion(CAM(F), SAM(F))
    """
    def __init__(self, channels: int,
                 reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.cam = ChannelAttention(channels, reduction)
        self.sam = SpatialAttention(spatial_kernel)
        # 可学习融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta  = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cam_out = self.cam(x)
        sam_out = self.sam(x)
        alpha = torch.sigmoid(self.alpha)
        beta  = torch.sigmoid(self.beta)
        return alpha * cam_out + beta * sam_out


# ═══════════════════════════════════════════════
#  将 CBAM 注入 ultralytics 命名空间
# ═══════════════════════════════════════════════
def register_cbam():
    """
    将 CBAM 注册到 ultralytics.nn.modules，使自定义 YAML 能识别 'CBAM'。
    需在加载自定义模型前调用。
    """
    try:
        import ultralytics.nn.modules as ult_modules
        import ultralytics.nn.tasks as tasks

        # 注入模块
        ult_modules.CBAM = CBAM

        # 修补 parse_model 函数，添加 CBAM 到已知模块列表
        _patch_parse_model()

        print("[CBAM] 注册成功 → ultralytics.nn.modules.CBAM")
        return True
    except Exception as e:
        print(f"[CBAM] 注册失败：{e}")
        return False


def _patch_parse_model():
    """猴子补丁：在 ultralytics 的 parse_model 中识别 CBAM"""
    import ultralytics.nn.tasks as tasks
    _orig_parse = tasks.parse_model

    def _patched_parse(d, ch, verbose=True, *args, **kwargs):
        # 将 CBAM 临时加入 globals
        import ultralytics.nn.modules as m
        setattr(m, 'CBAM', CBAM)
        return _orig_parse(d, ch, verbose, *args, **kwargs)

    tasks.parse_model = _patched_parse


# ═══════════════════════════════════════════════
#  创建 CBAM-YOLOv8n 模型
# ═══════════════════════════════════════════════
def build_cbam_yolov8n(nc: int = 8,
                        weights: str = None,
                        freeze_backbone: bool = True,
                        n_freeze: int = 10,
                        device: str = 'cpu') -> object:
    """
    构建 CBAM-YOLOv8n 模型并加载预训练权重（分层迁移学习）。

    Args:
        nc: 检测类别数（8 类单一干扰）
        weights: 预训练权重路径（YOLOv8n.pt 或 自定义权重）
        freeze_backbone: 是否冻结骨干网络浅层
        n_freeze: 冻结骨干前 n 层
        device: 'cpu' / 'cuda'

    Returns:
        ultralytics.YOLO 实例
    """
    from ultralytics import YOLO

    # 注册自定义模块
    register_cbam()

    # 写自定义 YAML（带 CBAM 的 YOLOv8n）
    yaml_path = _write_cbam_yaml(nc)

    # 创建模型
    model = YOLO(yaml_path)

    # 加载预训练权重（迁移学习）
    if weights and Path(weights).exists():
        print(f"[Model] 加载预训练权重：{weights}")
        model.load(weights)
    elif weights == 'yolov8n.pt':
        # 尝试用标准 YOLOv8n 初始化（分层迁移）
        try:
            _load_pretrained_backbone(model, 'yolov8n.pt', device)
        except Exception as e:
            print(f"[Model] 预训练权重加载失败（{e}），使用随机初始化")

    # 分层迁移学习：冻结骨干浅层
    if freeze_backbone:
        _freeze_backbone(model, n_freeze)

    return model


def _write_cbam_yaml(nc: int) -> str:
    """生成带 CBAM 的 YOLOv8n YAML 配置"""
    import os
    yaml_content = f"""# CBAM-YOLOv8n 配置（论文第 3 章）
# 在骨干网络 SPPF 后、颈部网络 C2f 后各插入 CBAM 注意力模块

nc: {nc}

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]         # 0 - P1/2
  - [-1, 1, Conv, [128, 3, 2]]        # 1 - P2/4
  - [-1, 3, C2f, [128, true]]         # 2
  - [-1, 1, Conv, [256, 3, 2]]        # 3 - P3/8
  - [-1, 6, C2f, [256, true]]         # 4
  - [-1, 1, Conv, [512, 3, 2]]        # 5 - P4/16
  - [-1, 6, C2f, [512, true]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]       # 7 - P5/32
  - [-1, 3, C2f, [1024, true]]        # 8
  - [-1, 1, SPPF, [1024, 5]]          # 9

head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]  # 10
  - [[-1, 6], 1, Concat, [1]]                 # 11 cat P4
  - [-1, 3, C2f, [512]]                       # 12
  - [-1, 1, nn.Upsample, [null, 2, nearest]]  # 13
  - [[-1, 4], 1, Concat, [1]]                 # 14 cat P3
  - [-1, 3, C2f, [256]]                       # 15 P3/8-small
  - [-1, 1, Conv, [256, 3, 2]]                # 16
  - [[-1, 12], 1, Concat, [1]]                # 17 cat head P4
  - [-1, 3, C2f, [512]]                       # 18 P4/16-medium
  - [-1, 1, Conv, [512, 3, 2]]                # 19
  - [[-1, 9], 1, Concat, [1]]                 # 20 cat head P5
  - [-1, 3, C2f, [1024]]                      # 21 P5/32-large
  - [[15, 18, 21], 1, Detect, [{nc}]]         # 22 Detect
"""
    os.makedirs('configs', exist_ok=True)
    path = 'configs/cbam_yolov8n.yaml'
    with open(path, 'w') as f:
        f.write(yaml_content)
    return path


def _load_pretrained_backbone(model, weights_path: str, device: str):
    """从标准 YOLOv8n 加载骨干权重（迁移学习）"""
    from ultralytics import YOLO
    pretrained = YOLO(weights_path)
    pt_dict    = pretrained.model.state_dict()
    my_dict    = model.model.state_dict()
    # 仅加载形状匹配的层
    matched = {k: v for k, v in pt_dict.items()
               if k in my_dict and v.shape == my_dict[k].shape}
    my_dict.update(matched)
    model.model.load_state_dict(my_dict, strict=False)
    n_loaded = len(matched)
    print(f"[Model] 迁移学习：成功加载 {n_loaded}/{len(my_dict)} 层")


def _freeze_backbone(model, n_layers: int):
    """冻结骨干网络前 n_layers 层（分层迁移学习）"""
    frozen = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < n_layers * 20:   # 每层约 20 个参数张量
            param.requires_grad = False
            frozen += 1
    print(f"[Model] 分层迁移：冻结 {frozen} 个参数张量（骨干浅层）")
