# YOLO11-M Baseline 训练配置记录

本文档记录本次 WaterScenes baseline 训练的固定配置，用于服务器训练留档和论文实验复现。

## 1. 实验名称

```text
Experiment: YOLO11-M baseline
Run name: yolo11m_baseline
Project: runs/aefc_yolo11
```

## 2. 模型配置

| 项目 | 配置 |
|---|---|
| 基础检测器 | Ultralytics YOLO11-M |
| 预训练权重 | `weights/yolo11m.pt` |
| 模型配置 | YOLO11-M 官方结构 |
| 自定义模块 | 不启用 UIAE / EAFC / MDCT |
| 训练目标 | 建立 WaterScenes 固定划分 baseline |

说明：当前 baseline 只训练 YOLO11-M，不接入 UIAE、EAFC、MDCT。后续消融实验必须以该结果作为基准。

## 3. 数据配置

数据配置文件：

```text
configs/waterscenes_full.yaml
```

YOLO 数据目录：

```text
path: waterscenes_yolo
train: images/train
val: images/val
test: images/test
```

类别定义：

| id | class |
|---:|---|
| 0 | pier |
| 1 | buoy |
| 2 | sailor |
| 3 | ship |
| 4 | boat |
| 5 | vessel |
| 6 | kayak |

固定划分来源：

```text
train.txt -> images/train
val.txt   -> images/val
test.txt  -> images/test
```

专项测试集：

```text
adverse_lighting.txt -> images/adverse_lighting
adverse_weather.txt  -> images/adverse_weather
```

## 4. 训练超参数

| 参数 | 数值 |
|---|---:|
| 输入尺寸 `imgsz` | 640 |
| 训练轮数 `epochs` | 200 |
| batch size | 16 |
| workers | 8 |
| 随机种子 `seed` | 42 |
| AMP | false |
| 余弦学习率 `cos_lr` | true |

## 5. 优化器与学习率

| 参数 | 数值 |
|---|---:|
| 优化器 | AdamW |
| 初始学习率 `lr0` | 0.001 |
| 最终学习率比例 `lrf` | 0.01 |
| weight decay | 0.0005 |
| momentum | 0.937 |
| warmup epochs | 3 |
| warmup momentum | 0.8 |
| warmup bias lr | 0.1 |

学习率策略：

```text
Cosine LR schedule
initial lr = 0.001
final lr = lr0 * lrf = 0.00001
```

## 6. 多卡训练配置

服务器计划使用 4 张 RTX PRO 6000 进行 DDP 训练。

推荐启动方式：

```bash
cd AEFC-YOLO11
GPUS=0,1,2,3 RUN_NAME=yolo11m_baseline bash tools/train_ddp_nohup.sh
```

等价手动命令：

```bash
nohup python tools/train_aefc.py \
  --cfg configs/train_aefc.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name yolo11m_baseline \
  --log-dir logs \
  --log-interval 100 \
  > logs/yolo11m_baseline.nohup.out 2>&1 &
```

注意：`batch: 16` 通常表示总 batch size。4 卡 DDP 下，每卡约 4。如果显存充足，可以后续尝试 `batch: 32` 或 `batch: 64`，但 baseline 首次训练建议先保持当前配置。

## 7. 日志配置

训练日志目录：

```text
logs/
```

日志文件：

```text
logs/yolo11m_baseline.log
logs/yolo11m_baseline_epoch_metrics.csv
logs/yolo11m_baseline.nohup.out
```

日志记录策略：

| 阶段 | 记录内容 |
|---|---|
| run_start | 模型、预训练权重、数据配置、batch、epoch、DDP 设备、优化器、学习率 |
| train_batch | 每 100 个 batch 记录 rank0 训练进度和 loss |
| train_epoch_end | 每个 epoch 结束记录训练汇总 |
| val_batch | 验证阶段每 100 个 batch 记录 rank0 验证进度 |
| val_end | 验证结束记录 mAP、recall 等指标 |
| run_end | 训练状态、耗时和异常信息 |

查看日志：

```bash
tail -f logs/yolo11m_baseline.log
```

## 8. 验证与测试

训练过程中每个 epoch 后自动在 `val.txt` 对应的验证集上验证。

训练完成后，需要额外执行常规测试集和专项测试集评价。

### 8.1 常规测试集

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_full.yaml \
  imgsz=640 \
  split=test
```

### 8.2 光照退化专项测试

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_lighting.yaml \
  imgsz=640 \
  split=val
```

### 8.3 天气退化专项测试

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_weather.yaml \
  imgsz=640 \
  split=val
```

## 9. 依赖环境

主要依赖来自 `requirements.txt`：

```text
ultralytics>=8.3.0
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.23.0
opencv-python>=4.8.0
Pillow>=9.5.0
PyYAML>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=1.5.0
```

服务器 CUDA 环境建议先单独安装与驱动匹配的 `torch` 和 `torchvision`，再执行：

```bash
pip install -r requirements.txt
```

## 10. 当前配置快照

```yaml
model: weights/yolo11m.pt
data: configs/waterscenes_full.yaml
imgsz: 640
epochs: 200
batch: 16
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
momentum: 0.937
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
cos_lr: true
workers: 8
device: 0
amp: false
seed: 42
project: runs/aefc_yolo11
name: yolo11m_baseline
log_interval: 100
```

## 11. 当前服务器实际训练配置

本节记录当前已经在远程服务器启动的 baseline 训练配置。该配置以 `configs/train_aefc.yaml` 为基础，并通过命令行覆盖了输入尺寸和 batch size。

```text
Date: 2026-05-22
GPU: 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition
Training mode: Ultralytics DDP
Run name: yolo11m_baseline
```

### 11.1 实际启动命令

```bash
nohup python tools/train_aefc.py \
  --cfg configs/train_aefc.yaml \
  --imgsz 1920 \
  --batch 32 \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name yolo11m_baseline \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/yolo11m_baseline.log \
  --plots false \
  > logs/yolo11m_baseline.nohup.out 2>&1 &
```

### 11.2 实际核心参数

| 参数 | 当前值 |
|---|---:|
| model | `weights/yolo11m.pt` |
| data | `configs/waterscenes_full.yaml` |
| imgsz | 1920 |
| epochs | 200 |
| global batch size | 32 |
| GPU 数量 | 4 |
| per-GPU batch size | 8 |
| optimizer | AdamW |
| lr0 | 0.001 |
| lrf | 0.01 |
| final lr | 0.00001 |
| weight decay | 0.0005 |
| momentum | 0.937 |
| warmup epochs | 3 |
| cos_lr | true |
| amp | false |
| seed | 42 |
| plots | false |
| log interval | 100 batches |

### 11.3 当前训练集 batch 计算

当前训练集数量：

```text
train images = 37884
```

当前全局 batch size：

```text
global batch = 32
```

因此每个 epoch 的训练 step 数为：

```text
ceil(37884 / 32) = 1184
```

这对应训练进度条中的：

```text
0/1184
1/1184
...
```

说明：这里的 `1184` 是全局 DDP step 数，不是单卡 batch 数。4 卡 DDP 下，每个 step 中每张卡处理约 8 张图像，总共处理 32 张图像。

### 11.4 日志文件

当前两个日志均采用覆盖写入：

```text
logs/yolo11m_baseline.log
logs/yolo11m_baseline.nohup.out
```

查看结构化训练日志：

```bash
tail -f logs/yolo11m_baseline.log
```

查看每轮指标 CSV：

```bash
tail -f logs/yolo11m_baseline_epoch_metrics.csv
```

查看 nohup 输出：

```bash
tail -f logs/yolo11m_baseline.nohup.out
```
