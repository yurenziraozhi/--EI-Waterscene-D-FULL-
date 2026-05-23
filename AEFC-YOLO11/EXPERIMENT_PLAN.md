# AEFC-YOLO11 实验计划

本文档列出 WaterScenes 鲁棒检测实验安排，包括 baseline、对比实验、消融实验和专项鲁棒性测试。

## 1. 数据划分

训练、验证和测试划分固定使用根目录中的 txt 文件：

```text
train.txt -> datasets/waterscenes_yolo/images/train
val.txt   -> datasets/waterscenes_yolo/images/val
test.txt  -> datasets/waterscenes_yolo/images/test
```

额外鲁棒性专项测试集：

```text
adverse_lighting.txt -> datasets/waterscenes_yolo/images/adverse_lighting
adverse_weather.txt  -> datasets/waterscenes_yolo/images/adverse_weather
```

所有模型必须使用相同的 `train.txt` 训练，相同的 `val.txt` 选择最佳权重，最后统一在 `test.txt`、`adverse_lighting.txt`、`adverse_weather.txt` 上评价。

## 2. 当前优先实验

当前代码已经支持 YOLO11-M baseline 的多卡 DDP、nohup 和日志记录。UIAE、EAFC、MDCT 已有骨架，但尚未接入 Ultralytics 训练图，因此第一阶段先完成 baseline。

| 编号 | 实验名称 | 模型 | 训练状态 | 目的 |
|---|---|---|---|---|
| B0 | YOLO11-M baseline | `yolo11m.pt` | 当前可跑 | 建立 WaterScenes 固定划分基准 |

训练命令：

```bash
cd AEFC-YOLO11
GPUS=0,1,2,3 RUN_NAME=yolo11m_baseline bash tools/train_ddp_nohup.sh
```

日志位置：

```text
logs/yolo11m_baseline_*.log
logs/yolo11m_baseline_*.nohup.out
```

## 3. Baseline 测试

baseline 训练完成后，分别测试常规测试集和两个专项鲁棒测试集。

### 3.1 常规测试集

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_full.yaml \
  imgsz=640 \
  split=test
```

### 3.2 光照退化专项测试

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_lighting.yaml \
  imgsz=640 \
  split=val
```

### 3.3 天气退化专项测试

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_weather.yaml \
  imgsz=640 \
  split=val
```

## 4. 主要对比实验

这些实验用于论文中证明 AEFC-YOLO11 相比常见检测器和图像增强式检测器的有效性。

| 编号 | 方法 | 说明 | 优先级 |
|---|---|---|---|
| C1 | YOLO11-M | 当前主 baseline | 必做 |
| C2 | YOLOv8-M | 常用 YOLO baseline | 建议做 |
| C3 | RT-DETR | Transformer 检测器 | 可选 |
| C4 | Faster R-CNN | 双阶段检测器 | 可选 |
| C5 | IA-YOLO-style | 多传统滤波器自适应增强 | 可选 |
| C6 | ERUP-YOLO-style | BPW + KBL 统一增强 | 建议做 |
| C7 | AEFC-YOLO11-M | 本文完整模型 | 必做 |

如果时间有限，最低对比组合为：

```text
YOLO11-M
YOLOv8-M
ERUP-YOLO-style / UIAE-YOLO11-M
AEFC-YOLO11-M
```

## 5. 模块级消融实验

消融实验按模块级设计，不再拆分 BPW-only 和 KBL-only。这样实验变量更清晰，论文叙事也更集中：先验证统一增强模块 UIAE，再验证特征校准模块 EAFC，最后验证完整方案。

| 编号 | 实验名称 | UIAE | EAFC | MDCT | 目的 |
|---|---|---|---|---|---|
| A | YOLO11-M | × | × | × | 检测器基线 |
| B | UIAE-YOLO11-M | ✓ | × | × | 验证统一图像自适应增强模块 |
| C | EAFC-YOLO11-M | × | ✓ | × | 验证增强感知特征校准模块 |
| D | AEFC-YOLO11-M | ✓ | ✓ | ✓ | 验证完整鲁棒检测方案 |

建议训练命名：

```text
ablation_a_yolo11m
ablation_b_uiae
ablation_c_eafc
ablation_d_full
```

B 组 UIAE-only 训练配置：

```text
configs/train_uiae.yaml
```

B 组启动命令：

```bash
nohup python tools/train_aefc.py \
  --cfg configs/train_uiae.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name ablation_b_uiae \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/ablation_b_uiae.log \
  --save-period -1 \
  --plots false \
  > logs/ablation_b_uiae.nohup.out 2>&1 &
```

## 6. 评价指标

每个模型至少记录以下指标：

```text
mAP@0.5
mAP@0.5:0.95
Precision
Recall
FPS
Params
GFLOPs
```

专项鲁棒性测试重点记录：

```text
Lighting mAP@0.5
Lighting mAP@0.5:0.95
Lighting Recall
Weather mAP@0.5
Weather mAP@0.5:0.95
Weather Recall
```

## 7. 结果汇总表

### 7.1 主结果表

| 方法 | Test mAP@0.5 | Test mAP@0.5:0.95 | Lighting mAP@0.5 | Weather mAP@0.5 | Params | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11-M |  |  |  |  |  |  |  |
| YOLOv8-M |  |  |  |  |  |  |  |
| UIAE-YOLO11-M |  |  |  |  |  |  |  |
| AEFC-YOLO11-M |  |  |  |  |  |  |  |

### 7.2 消融表

| 实验 | UIAE | EAFC | MDCT | Test mAP@0.5 | Lighting mAP@0.5 | Weather mAP@0.5 | Recall |
|---|---|---|---|---:|---:|---:|---:|
| A | × | × | × |  |  |  |  |
| B | ✓ | × | × |  |  |  |  |
| C | × | ✓ | × |  |  |  |  |
| D | ✓ | ✓ | ✓ |  |  |  |  |

## 8. 执行顺序

建议按以下顺序执行：

1. 准备数据目录，确认 `train/val/test/adverse_lighting/adverse_weather` 均生成成功。
2. 训练 `YOLO11-M baseline`。
3. 在 `test`、`adverse_lighting`、`adverse_weather` 上评价 baseline。
4. 接入 UIAE 训练图，执行 B 组实验。
5. 接入 EAFC 训练图，执行 C 组实验。
6. 接入完整 AEFC 方案，执行 D 组实验。
7. 汇总主结果表和消融表。

## 9. 注意事项

- 所有实验必须使用同一套 `train.txt`、`val.txt`、`test.txt`。
- 不要重新随机划分数据。
- 每个实验使用独立 `RUN_NAME`，避免覆盖权重和日志。
- 多卡训练日志只统计 rank0，避免重复写日志。
- 当前阶段不要开启 `--use-uiae`、`--use-eafc`、`--use-mdct`，这些开关会在完整训练图接入后使用。
