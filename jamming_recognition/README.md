# 基于深度学习的小样本雷达复合干扰识别
## CBAM-YOLOv8n + 分层迁移学习


---

## 项目结构

```
jamming_recognition/
├── main.py                   # 主入口（控制全流程）
├── requirements.txt          # 依赖列表
├── README.md
├── configs/
│   └── cbam_yolov8n.yaml    # CBAM-YOLOv8n 模型配置（自动生成）
├── src/
│   ├── config.py            # 全局参数（与 MATLAB 代码完全对应）
│   ├── signal_gen.py        # 8 类单一干扰 + 15 类复合干扰生成
│   ├── dataset_builder.py   # YOLO 格式数据集构建
│   ├── cbam.py              # CBAM 注意力模块 + YOLOv8n 集成
│   ├── trainer.py           # 分层迁移学习训练
│   ├── evaluator.py         # 混淆矩阵、AP、mAP 评估
│   ├── visualizer.py        # 复现论文所有图表（Fig1~Fig9）
│   └── font_setup.py        # 中文字体配置
├── dataset/                  # 自动生成的 YOLO 数据集
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset.yaml
├── results/                  # 输出图表（PNG, 300 dpi）
└── weights/                  # 训练权重
```

---

## 一、环境配置

### 方式 A：本地安装（推荐）

```bash
# Python >= 3.9
pip install ultralytics>=8.2.0 torch torchvision scipy matplotlib \
            scikit-learn Pillow tqdm seaborn pandas opencv-python pyyaml
```

### 方式 B：Conda 环境

```bash
conda create -n jamming python=3.10 -y
conda activate jamming
pip install ultralytics torch torchvision scipy matplotlib \
            scikit-learn Pillow tqdm seaborn pandas
```

### 方式 C：Google Colab（推荐，有 GPU）

```python
!pip install ultralytics scipy matplotlib scikit-learn -q
# 上传本项目文件夹后：
!python main.py --mode all --device cuda
```

### 中文字体（可选，用于图表中文显示）

```bash
# Linux
sudo apt-get install fonts-wqy-zenhei -y

# macOS：系统已内置 PingFang SC，无需安装

# Windows：系统已内置 SimHei
```

---

## 二、快速使用

### 2.1 仅生成论文图表（不需要训练，秒级完成）

```bash
cd jamming_recognition
python main.py --mode visualize
```

输出到 `results/` 目录：
- `Fig1_Single_Jamming_TF.png`    — 8 类单一干扰时频图 + YOLO 框
- `Fig2a/b_YOLO_Detection.png`   — 复合干扰检测可视化
- `Fig3_Training_Curves.png`      — 训练过程曲线
- `Fig4a/b_Confusion_*.png`       — 归一化混淆矩阵（精度 0.001）
- `Fig5_Per_Class_AP.png`         — 各类 AP 值
- `Fig6_Few_Shot_mAP.png`         — 小样本 mAP 对比
- `Fig7_Robustness_JNR.png`       — 鲁棒性测试
- `Fig8_Ablation_Study.png`       — 消融实验
- `Fig9_Method_Comparison.png`    — 方法对比

### 2.2 完整流程（构建数据集 → 训练 → 评估 → 可视化）

```bash
# CPU（小规模验证）
python main.py --mode all --n_shots 6 --n_test 100 --epochs 50

# GPU（正式实验）
python main.py --mode all --n_shots 6 --n_test 1000 --epochs 100 --device cuda
```

### 2.3 分步执行

```bash
# 步骤 1：构建数据集
python main.py --mode dataset --n_shots 6 --n_test 1000

# 步骤 2：训练
python main.py --mode train --n_shots 6 --epochs 100 --device cuda

# 步骤 3：评估 + 可视化
python main.py --mode eval --weights results/cbam_yolov8n_6shot_phase2/weights/best.pt

# 步骤 4：小样本梯度实验（3~8 样本/类）
python main.py --mode fewshot --epochs 50 --device cuda
```

---

## 三、信号参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样频率 fs | 100 MHz | 与 MATLAB 完全一致 |
| 带宽 B | 10 MHz | LFM 扫频范围 |
| 脉宽 T_p | 50 µs | LFM 脉冲宽度 |
| 调频斜率 K | 0.2 MHz/µs | K = B/T_p |
| 观测窗口 | 100 µs | N = 10000 点 |
| STFT 窗长 | 128 | Hamming 窗 |
| STFT 帧移 | 8 | 高时间分辨率 |
| FFT 点数 | 512 | 高频率分辨率 |
| 显示频率范围 | ±12 MHz | 完整覆盖 Comb ±9 MHz |

---

## 四、与 MATLAB 代码的对应关系

| MATLAB 函数 | Python 对应 |
|-------------|-------------|
| `gen_LFM()` | `signal_gen.gen_LFM()` |
| `gen_single()` | `signal_gen.gen_single()` |
| `gen_compound()` | `signal_gen.gen_compound()` |
| `do_stft()` | `dataset_builder.signal_to_tfimage()` / `visualizer._do_stft()` |
| `gen_confmat_group()` | `evaluator.gen_simulated_confmat()` |
| `add_noise()` | `signal_gen.add_noise()` |
| Fig1~Fig9 绘图代码 | `visualizer.py` 对应函数 |

---

## 五、关键设计说明

### CBAM 集成方式

YOLOv8n + CBAM 采用**并联混合注意力**（对应论文图 3-2）：

```python
# src/cbam.py
class CBAM(nn.Module):
    def forward(self, x):
        cam_out = self.cam(x)   # 通道注意力
        sam_out = self.sam(x)   # 空间注意力
        # 可学习权重融合
        return alpha * cam_out + beta * sam_out
```

### 分层迁移学习策略

```
阶段一（1/3 epochs）：冻结骨干浅层，lr = lr_init × 0.1
         ↓
阶段二（2/3 epochs）：解冻所有层，lr = lr_init，余弦退火
```

### 数据集格式

- YOLO 检测任务：8 类单一干扰分量
- 单一干扰图像：1 个检测框
- 复合干扰图像：2 个检测框（各对应一个分量）
- 时频图坐标 → YOLO 归一化坐标转换见 `dataset_builder.tfbox_to_yolo()`

---

## 六、硬件建议

| 配置 | 建议用途 |
|------|---------|
| CPU only | 快速验证（`--n_test 50 --epochs 10`） |
| RTX 3060/4060 | 标准实验（`--epochs 100`） |
| RTX 3090/4090 | 完整 few-shot 梯度实验 |
| Google Colab T4 | 免费 GPU，推荐用于论文实验 |

---

## 七、输出文件说明

```
results/
├── Fig1_Single_Jamming_TF.png     8类单一干扰时频图（含检测框）
├── Fig2a_YOLO_Detection.png       复合干扰检测(前9类), 300dpi
├── Fig2b_YOLO_Detection.png       复合干扰检测(后6类), 300dpi
├── Fig3_Training_Curves.png       训练/验证损失 + mAP 曲线
├── Fig4a_Confusion_Single.png     8×8 单一干扰混淆矩阵
├── Fig4b_Confusion_Compound.png   15×15 复合干扰混淆矩阵
├── Fig5_Per_Class_AP.png          23 类 AP 值柱状图
├── Fig6_Few_Shot_mAP.png          3~8样本梯度对比曲线
├── Fig7_Robustness_JNR.png        JNR扫描鲁棒性测试
├── Fig8_Ablation_Study.png        消融实验柱状图
├── Fig9_Method_Comparison.png     各方法对比
└── run_info.json                  运行参数记录
```
