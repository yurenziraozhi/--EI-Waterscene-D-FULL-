# AEFC-YOLO11

本工程用于基于 WaterScenes 固定划分训练 YOLO11-M baseline，并逐步加入 UIAE、EAFC、MDCT，在 `adverse_lighting` 与 `adverse_weather` 两个专项测试集上评估鲁棒性。

## 1. 服务器目录

建议保持如下结构：

```text
workdir/
|-- AEFC-YOLO11/
|-- image/
|-- detection/
|   `-- yolo/
|-- train.txt
|-- val.txt
|-- test.txt
|-- adverse_lighting.txt
`-- adverse_weather.txt
```

`image/`、`detection/yolo/` 和五个 txt 与 `AEFC-YOLO11/` 同级。

## 2. 准备数据

先安装依赖：

```bash
pip install -r requirements.txt
```

在 `AEFC-YOLO11/` 目录执行：

```bash
python tools/prepare_waterscenes_yolo.py \
  --root .. \
  --image-dir image \
  --label-dir detection/yolo \
  --train-list train.txt \
  --val-list val.txt \
  --test-list test.txt \
  --lighting-list adverse_lighting.txt \
  --weather-list adverse_weather.txt \
  --out datasets/waterscenes_yolo \
  --mode hardlink
```

## 3. 多卡 DDP + nohup 训练

当前 `tools/train_aefc.py` 已适配 Ultralytics 多卡 DDP 和文件日志。默认训练 YOLO11-M baseline：

```bash
bash tools/train_ddp_nohup.sh
```

默认使用 `0,1,2,3` 四张卡。要改卡号：

```bash
GPUS=0,1 bash tools/train_ddp_nohup.sh
```

如果要使用物理卡 4、5，并通过 `CUDA_VISIBLE_DEVICES` 映射成逻辑卡 0、1：

```bash
VISIBLE_GPUS=4,5 DEVICE=0,1 GPUS=0,1 bash tools/train_ddp_nohup.sh
```

训练日志写入：

```text
logs/yolo11m_baseline.log
logs/yolo11m_baseline_epoch_metrics.csv
```

`nohup` 自身输出写入：

```text
logs/yolo11m_baseline.nohup.out
```

训练日志为 JSONL，每行一条记录，包含：

- `run_start`：模型、预训练权重、数据配置、batch、epoch、DDP 设备、优化器、学习率等；
- `train_batch`：每 100 个 batch 输出一次 rank0 的训练进度和 loss；
- `train_epoch_end`：每个 epoch 结束输出训练汇总；
- `val_batch`：验证阶段每 100 个 batch 输出一次 rank0 验证进度；
- `val_end`：验证结束输出 mAP、recall 等 Ultralytics 指标；
- `run_end`：训练是否成功、耗时和异常信息。

每轮验证结束后，脚本会同步写入一行 CSV：

```text
logs/yolo11m_baseline_epoch_metrics.csv
```

字段包括 `epoch`、`box_loss`、`cls_loss`、`dfl_loss`、`precision`、`recall`、`map50`、`map50_95`、`fitness` 和 `lr`。

## 4. 手动启动训练

也可以直接执行：

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

## 5. 专项测试

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_lighting.yaml \
  imgsz=640 \
  split=val

yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_adverse_weather.yaml \
  imgsz=640 \
  split=val
```

## 6. 当前实现状态

- `models/uiae.py`：已按 BPW + KBL 两滤波器方案实现 UIAE 骨架。
- `models/eafc.py`：已实现增强感知特征校准，当前保留逐通道空间门控。
- `tools/train_aefc.py`：已支持 YOLO11-M baseline 的 DDP、nohup 和 JSONL 训练日志。
- UIAE/EAFC/MDCT 尚未接入 Ultralytics 训练图。开启 `--use-uiae`、`--use-eafc` 或 `--use-mdct` 时脚本会拒绝启动，避免训练名和实际模型不一致。
