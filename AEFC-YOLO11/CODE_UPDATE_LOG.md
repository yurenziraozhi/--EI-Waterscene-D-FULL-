# AEFC-YOLO11 Code Update Log

本文档用于记录每次代码更新、训练问题修复、服务器上传说明和 GitHub 推送备注。后续每次改代码都应追加一条记录，避免不同实验版本混在一起。

## 记录格式

每次更新建议记录：

```text
日期:
版本/实验:
触发问题:
修复内容:
涉及文件:
服务器需要上传:
训练命令/验证命令:
GitHub 推送备注:
结果/后续观察:
```

## 2026-05-25 D-core condition-aware fix3

版本/实验：

```text
experiment_d_uiae_eafc_condition_aware_960_unfrozen_fix3
```

触发问题：

```text
baseline_yolo11m_960 在相同 imgsz=960 下收敛很快并明显强于 D fix2；主要差异是 baseline 全模型训练，而 D fix2 冻结了 backbone，只训练检测头、UIAE 和 EAFC。
```

修复内容：

```text
1. 将 freeze_backbone 从 true 改为 false，使 D 组和 baseline 一样允许 backbone/neck/head 全模型适应 WaterScenes。
2. 将 lr0 从 0.0001 降到 0.00005，降低 UIAE/EAFC 与全模型联合训练初期震荡。
3. 保持 model=weights/yolo11m.pt、imgsz=960、batch=32、use_uiae=true、use_eafc=true、use_mdct=false、condition_aware_enhancement=true。
4. 继续使用 adverse_lighting.txt/adverse_weather.txt 对正常图和恶劣图进行硬标签门控。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练命令：

```bash
cd ~/autodl-tmp/鲁棒检测-EI/AEFC-YOLO11
mkdir -p logs

PYTHONPATH="$(pwd):${PYTHONPATH:-}" nohup python tools/train_aefc.py \
  --cfg configs/train_aefc_d.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name experiment_d_uiae_eafc_condition_aware_960_unfrozen_fix3 \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/experiment_d_uiae_eafc_condition_aware_960_unfrozen_fix3.log \
  --save-period -1 \
  --plots false \
  > logs/experiment_d_uiae_eafc_condition_aware_960_unfrozen_fix3.nohup.out 2>&1 &
```

GitHub 推送备注：

```text
待推送。
```

结果/后续观察：

```text
重点比较 full/adverse_lighting/adverse_weather 三组 test 指标；如果 full 接近 baseline 且专项提升，说明条件增强方案成立。
```

## 2026-05-25 baseline 960 config

版本/实验：

```text
baseline_yolo11m_960
```

触发问题：

```text
D fix2 使用 imgsz=960，因此 baseline 也需要按相同图像尺寸和主要训练超参数重跑，避免继续拿旧的 1920 或 640 baseline 做不公平比较。
```

修复内容：

```text
1. 新增 configs/train_baseline_960.yaml。
2. 保持 model=weights/yolo11m.pt、data=configs/waterscenes_full.yaml、imgsz=960、batch=32、optimizer=AdamW、lr0=0.0001、amp=false、seed=42。
3. 显式关闭 use_uiae/use_eafc/use_mdct/condition_aware_enhancement，确保这是无创新模块 baseline。
4. 不修改 D 组配置 train_aefc_d.yaml。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_baseline_960.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_baseline_960.yaml
CODE_UPDATE_LOG.md
```

推荐训练命令：

```bash
cd ~/autodl-tmp/鲁棒检测-EI/AEFC-YOLO11
mkdir -p logs

PYTHONPATH="$(pwd):${PYTHONPATH:-}" nohup python tools/train_aefc.py \
  --cfg configs/train_baseline_960.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name baseline_yolo11m_960 \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/baseline_yolo11m_960.log \
  --save-period -1 \
  --plots false \
  > logs/baseline_yolo11m_960.nohup.out 2>&1 &
```

GitHub 推送备注：

```text
待推送。
```

结果/后续观察：

```text
训练完成后使用同样 imgsz=960 在 full test、adverse_lighting、adverse_weather 上评测。
```

## 2026-05-25 AEFC validation loader fix

版本/实验：

```text
tools/val_aefc.py
```

触发问题：

```text
使用 yolo detect val 直接加载 D fix2 best.pt 时，checkpoint 中保存了自定义 AEFC forward，Ultralytics CLI 未注册该方法，触发 AttributeError: DetectionModel object has no attribute _forward_with_aefc。
```

修复内容：

```text
1. 将 tools/val_aefc.py 改为 AEFC 专用验证入口，不再转调 yolo CLI。
2. 在 torch.load 前注册 DetectionModel._forward_with_aefc，保证旧 checkpoint 能正常反序列化。
3. 验证时 patch DetectionValidator.preprocess，根据 batch["im_file"] 生成 uiae_adverse_mask。
4. full test 按 adverse_lighting.txt/adverse_weather.txt 做条件增强；专项 adverse yaml 中样本会全部命中恶劣集合。
```

涉及文件：

```text
AEFC-YOLO11/tools/val_aefc.py
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
tools/val_aefc.py
CODE_UPDATE_LOG.md
```

GitHub 推送备注：

```text
待推送。
```

结果/后续观察：

```text
使用 python tools/val_aefc.py 代替 yolo detect val 评测 AEFC checkpoint。
```

## 2026-05-25 D-core condition-aware fix2

版本/实验：

```text
experiment_d_uiae_eafc_condition_aware_960_fix2
```

触发问题：

```text
全量 WaterScenes 数据中并非所有图片都是恶劣光照或恶劣天气；如果 UIAE/EAFC 对正常图片强制增强，可能破坏正常样本特征并干扰检测训练。
```

修复内容：

```text
1. 增加 condition_aware_enhancement 配置，按 adverse_lighting.txt 和 adverse_weather.txt 生成 batch 级硬标签。
2. 命中恶劣 txt 的样本 gate=1，UIAE 增强和 EAFC 校准正常生效。
3. 未命中恶劣 txt 的样本 gate=0，增强输出强制回到原图，避免正常图被改写。
4. 正常样本只承担 UIAE identity/consistency 约束，恶劣样本不再被该约束拉回恒等增强。
5. 日志和 CSV 增加 condition_aware、adverse_fraction、adverse_count、normal_count、enh_delta_adverse_mean、enh_delta_normal_mean。
6. 如果开启条件感知但两个 txt 未读到任何样本，训练直接报错，避免静默跑错实验。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/models/uiae_trainer.py
AEFC-YOLO11/tools/train_aefc.py
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
models/uiae_trainer.py
tools/train_aefc.py
CODE_UPDATE_LOG.md
adverse_lighting.txt
adverse_weather.txt
```

推荐训练命令：

```bash
cd ~/autodl-tmp/鲁棒检测-EI/AEFC-YOLO11
mkdir -p logs

PYTHONPATH="$(pwd):${PYTHONPATH:-}" nohup python tools/train_aefc.py \
  --cfg configs/train_aefc_d.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name experiment_d_uiae_eafc_condition_aware_960_fix2 \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/experiment_d_uiae_eafc_condition_aware_960_fix2.log \
  --save-period -1 \
  --plots false \
  > logs/experiment_d_uiae_eafc_condition_aware_960_fix2.nohup.out 2>&1 &
```

GitHub 推送备注：

```text
待推送。
```

结果/后续观察：

```text
先看第 1 轮日志中的 adverse_fraction 是否符合预期；如果长期为 0 或 1，说明 txt 路径或文件名匹配规则需要调整。
```

## 2026-05-24 D-core no-MDCT fix1i

版本/实验：

```text
experiment_d_uiae_eafc_no_mdct_960_conservative_fix1i
```

触发问题：

```text
严格消融实验必须保持与 baseline 相同的预训练起点，不能把 D 组切换到 pretrain-best.pt；同时 960 尺寸版本比 1920 稳定，但 loss 仍需要进一步收敛。
```

修复内容：

```text
1. 保持 model=weights/yolo11m.pt，保证 baseline/B/C/D 使用相同预训练模型。
2. 保持 imgsz=960、batch=32、use_mdct=false、freeze_backbone=true。
3. lr0 从 0.0002 降到 0.0001，降低 head+UIAE+EAFC 联合训练初期震荡。
4. uiae_blend_init 从 0.01 降到 0.001，让 UIAE 初始输出更接近原图。
5. eafc_alpha_init 从 0.01 降到 0.001，让 EAFC 初始校准更接近原始检测特征。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练命令：

```bash
cd ~/autodl-tmp/鲁棒检测-EI/AEFC-YOLO11
mkdir -p logs

PYTHONPATH="$(pwd):${PYTHONPATH:-}" nohup python tools/train_aefc.py \
  --cfg configs/train_aefc_d.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name experiment_d_uiae_eafc_no_mdct_960_conservative_fix1i \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/experiment_d_uiae_eafc_no_mdct_960_conservative_fix1i.log \
  --save-period -1 \
  --plots false \
  > logs/experiment_d_uiae_eafc_no_mdct_960_conservative_fix1i.nohup.out 2>&1 &
```

GitHub 推送备注：

```text
待推送。
```

结果/后续观察：

```text
观察第 1 轮 cls_loss 是否继续平滑下降，重点看 results.csv 和内部日志中的 uiae_grad_norm/eafc_grad_norm 是否非零且不过大。
```

## 2026-05-24 D-FULL fix1

版本/实验：

```text
experiment_d_full_aefc_fix1
```

触发问题：

```text
D 组全套方案训练 13 轮后验证 mAP 仍低于 baseline，说明当前 full AEFC 训练策略或参数传递存在问题。
```

定位结论：

```text
1. train_aefc.py 中 use_eafc/use_mdct/mdct_start_epoch 等自定义参数没有稳定传入 Ultralytics DDP 子进程。
2. freeze_uiae_epochs=3 会导致 optimizer 构建时 UIAE 参数可能处于 requires_grad=False 状态，后续解冻不一定进入 optimizer。
3. MDCT 从第 1 轮直接启用，可能在检测器尚未适应 UIAE/EAFC 时引入过强扰动。
4. 日志缺少 UIAE/EAFC 梯度范数，无法确认两个模块是否真的参与训练。
```

修复内容：

```text
1. 通过环境变量 AEFC_TRAIN_ARGS 将 AEFC 自定义参数传入 DDP 子进程。
2. freeze_uiae_epochs 改为 0，UIAE 从第 1 轮进入 optimizer，保持全套端到端训练。
3. uiae_blend_init 从 0.02 降到 0.01。
4. eafc_alpha_init 从 0.02 降到 0.01。
5. MDCT 改成第 5 轮开始，20 轮 warmup 后逐步达到 p_degrade=0.20。
6. 日志和 CSV 增加 uiae_trainable_params、uiae_grad_norm、eafc_trainable_params、eafc_grad_norm、mdct_effective_p。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/models/uiae_trainer.py
AEFC-YOLO11/tools/train_aefc.py
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
models/uiae_trainer.py
tools/train_aefc.py
```

推荐训练命令：

```bash
cd ~/autodl-tmp/鲁棒检测-EI/AEFC-YOLO11
mkdir -p logs

PYTHONPATH="$(pwd):${PYTHONPATH:-}" nohup python tools/train_aefc.py \
  --cfg configs/train_aefc_d.yaml \
  --device 0,1,2,3 \
  --project runs/aefc_yolo11 \
  --name experiment_d_full_aefc_fix1 \
  --log-dir logs \
  --log-interval 100 \
  --log-file logs/experiment_d_full_aefc_fix1.log \
  --save-period -1 \
  --plots false \
  > logs/experiment_d_full_aefc_fix1.nohup.out 2>&1 &
```

启动后重点检查：

```bash
tail -f logs/experiment_d_full_aefc_fix1.log
```

需要确认：

```text
use_eafc=true
use_mdct=true
uiae_trainable_params 非 0
eafc_trainable_params 非 0
uiae_grad_norm 有数值
eafc_grad_norm 有数值
mdct_effective_p 前 4 轮为 0，之后逐步增加
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Fix full AEFC DDP args and staged MDCT training
```

结果/后续观察：

```text
等待服务器重新训练 experiment_d_full_aefc_fix1 后补充。
```

### 2026-05-24 D-FULL fix1c

触发问题：

```text
experiment_d_full_aefc_fix1b 启动后虽然解决了双分支显存问题，但用户观察训练 loss 仍明显不如 baseline。
```

定位结论：

```text
从 COCO 预训练权重 weights/yolo11m.pt 直接训练 full AEFC，相当于同时适配 WaterScenes 检测任务和 UIAE/EAFC 新模块。
相比 baseline 已经在 WaterScenes 上收敛好的检测器，这会导致 full AEFC 早期 loss 更高、收敛更慢。
```

修复内容：

```text
1. 采用方案 1：使用 WaterScenes baseline best.pt 作为 D-FULL 初始化权重。
2. 配置从 model: weights/yolo11m.pt 改为 model: weights/pretrain-best.pt。
3. 保持 UIAE、EAFC、MDCT 的 full 方案不变。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
weights/pretrain-best.pt
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_full_aefc_pretrain_fix1c
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Use WaterScenes baseline best weight for full AEFC pretraining
```

### 2026-05-24 D-core no-MDCT fix1d

触发问题：

```text
服务器继续观察到 full AEFC 训练日志不理想，loss 仍不如 baseline。
用户指出 EAFC 必须基于 UIAE 增强特征才有意义，因此不能改成纯 EAFC-only。
```

策略调整：

```text
保留 UIAE + EAFC，暂时关闭 MDCT。
这样仍然保留增强分支和增强感知特征校准逻辑：
F_raw_s, F_enh_s, F_enh_s - F_raw_s -> EAFC -> F_out_s
只先去掉在线退化扰动，降低训练早期不稳定因素。
```

修复内容：

```text
1. configs/train_aefc_d.yaml 中 use_mdct 从 true 改为 false。
2. model 继续使用 weights/pretrain-best.pt，即 WaterScenes baseline best 权重。
3. UIAE 和 EAFC 继续开启。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_uiae_eafc_no_mdct_pretrain_fix1d
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Disable MDCT for UIAE EAFC pretrain experiment
```

### 2026-05-24 D-core no-MDCT fix1e

触发问题：

```text
用户决定不再使用 weights/pretrain-best.pt 初始化 D 组，要求将预训练模型重新换回 YOLO11 官方权重。
```

修复内容：

```text
configs/train_aefc_d.yaml 中 model 从 weights/pretrain-best.pt 改回 weights/yolo11m.pt。
当前配置仍保持 UIAE + EAFC 开启、MDCT 关闭。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_uiae_eafc_no_mdct_yolo11_fix1e
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Switch D no-MDCT experiment back to YOLO11 pretrained weight
```

### 2026-05-24 D-core no-MDCT fix1f

触发问题：

```text
用户检查 experiment_d_uiae_eafc_no_mdct_pretrain_fix1d.nohup.out 后发现第一轮中后段 cls_loss 明显反弹。
本地解析 nohup 后确认：epoch 1 的 cls_loss 在 batch 421 附近最低约 0.6327，随后逐步升高，到 epoch 结束约 1.269。
```

定位结论：

```text
这不是单个坏 batch，而是训练过程中模型被 UIAE/EAFC 新模块快速拖离原检测器分布。
如果检测器主干和检测头同时更新，新的增强/融合分支会和检测器参数一起漂移，导致 loss 曲线不稳。
```

修复内容：

```text
1. 新增 freeze_detector 配置。
2. freeze_detector=true 时冻结 YOLO 检测器主体，只训练 UIAE 和 EAFC。
3. loss 仍通过冻结的 YOLO 检测器反向传播到 UIAE/EAFC，因此不是关闭检测器监督。
4. 日志诊断增加 detector_frozen、detector_trainable_params，用于确认检测器是否被锚住。
```

涉及文件：

```text
AEFC-YOLO11/tools/train_aefc.py
AEFC-YOLO11/models/uiae_trainer.py
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
tools/train_aefc.py
models/uiae_trainer.py
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_uiae_eafc_no_mdct_frozen_detector_fix1f
```

后续策略：

```text
先用 freeze_detector=true 跑 3-5 个 epoch。
如果 cls_loss 不再中途反弹，并且 val mAP 不崩，再用该权重作为下一阶段初始化，考虑解冻检测头或全模型小学习率微调。
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Freeze detector for stable UIAE EAFC adapter training
```

### 2026-05-24 D-core no-MDCT fix1g

触发问题：

```text
experiment_d_uiae_eafc_no_mdct_frozen_detector_fix1f 启动后 cls_loss 从 6-7 开始，明显异常。
日志中出现大量 “setting requires_grad=True for frozen layer” 警告。
```

定位结论：

```text
当前配置已经把模型权重切回 weights/yolo11m.pt。YOLO11 官方权重是 COCO 80 类，加载 WaterScenes 7 类时检测头会部分重建或随机初始化。
如果 freeze_detector=true，则 7 类检测头也被冻住，分类头无法学习，cls_loss 高是必然的。
同时原 freeze_detector 在 setup_model 阶段设置，容易被 Ultralytics 后续冻结/解冻逻辑覆盖，警告说明冻结策略不干净。
```

修复内容：

```text
1. 新增 freeze_backbone 配置。
2. 使用 weights/yolo11m.pt 时不再冻结整个 detector。
3. 当前策略改为 freeze_detector=false、freeze_backbone=true：
   冻结 YOLO 主干/颈部，只训练 Detect head + UIAE + EAFC。
4. 将冻结策略移动到 build_optimizer 阶段执行，确保在 Ultralytics 自身冻结处理之后生效。
5. 日志诊断增加 backbone_frozen。
```

涉及文件：

```text
AEFC-YOLO11/tools/train_aefc.py
AEFC-YOLO11/models/uiae_trainer.py
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
tools/train_aefc.py
models/uiae_trainer.py
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_uiae_eafc_no_mdct_freeze_backbone_fix1g
```

期望日志：

```text
backbone_frozen=true
detector_frozen=false
detector_trainable_params 非 0
uiae_trainable_params 非 0
eafc_trainable_params 非 0
cls_loss 不应再维持在 6-7
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Train YOLO head with frozen backbone for UIAE EAFC
```

### 2026-05-24 D-core no-MDCT fix1h

触发问题：

```text
用户观察到输入尺寸较小时，初期 cls_loss 显著降低，说明高分辨率 1920 直接训练 UIAE+EAFC 过于激进。
```

策略调整：

```text
采用 progressive resizing 的第一阶段：先用 imgsz=960 训练，batch 保持 32 不变。
后续如果 960 阶段 loss 和 val mAP 稳定，再加载该阶段 best.pt 升到 1280 或 1920 微调。
```

修复内容：

```text
configs/train_aefc_d.yaml 中 imgsz 从 1920 改为 960。
batch 保持 32。
其他当前策略保持不变：YOLO11 权重、UIAE+EAFC 开启、MDCT 关闭、freeze_backbone=true。
```

涉及文件：

```text
AEFC-YOLO11/configs/train_aefc_d.yaml
AEFC-YOLO11/CODE_UPDATE_LOG.md
```

服务器需要上传：

```text
configs/train_aefc_d.yaml
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_uiae_eafc_no_mdct_960_fix1h
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Use 960 image size for stable UIAE EAFC warmup
```

### 2026-05-24 D-FULL fix1a

触发问题：

```text
服务器启动 experiment_d_full_aefc_fix1 时崩溃：
TypeError: MultiScaleEAFC.__init__() got an unexpected keyword argument 'alpha_init'
```

定位结论：

```text
服务器上的 models/eafc.py 与本地 D-FULL 版本不一致，旧版 MultiScaleEAFC 构造函数不支持 alpha_init 参数。
这说明只上传 train_aefc.py、uiae_trainer.py 和 train_aefc_d.yaml 不够，服务器仍可能保留旧 eafc.py。
```

修复内容：

```text
1. uiae_trainer.py 中 _attach_eafc 增加兼容逻辑。
2. 优先使用 MultiScaleEAFC(DETECT_FEATURE_CHANNELS, alpha_init=alpha_init)。
3. 如果服务器旧版 eafc.py 不支持 alpha_init，则回退到 MultiScaleEAFC(DETECT_FEATURE_CHANNELS)。
4. 如果 block 存在 _init_near_raw，则回退后手动按 alpha_init 做近原始分支初始化。
```

涉及文件：

```text
AEFC-YOLO11/models/uiae_trainer.py
```

服务器需要上传：

```text
必须上传：
models/uiae_trainer.py

强烈建议同时上传，避免版本不一致：
models/eafc.py
models/uiae.py
models/degradation.py
tools/train_aefc.py
configs/train_aefc_d.yaml
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Fix EAFC alpha_init compatibility for DDP servers
```

### 2026-05-24 D-FULL fix1b

触发问题：

```text
experiment_d_full_aefc_fix1a 启动后在第 1 个 batch 前向阶段 CUDA OOM。
报错位置在 enhanced 分支执行 YOLO backbone 的 C2PSA attention 时：
torch.OutOfMemoryError: Tried to allocate 1.55 GiB，单卡已占用约 93.92 GiB。
```

定位结论：

```text
当前 EAFC 训练路径同时对 raw_img 和 enhanced 执行 YOLO backbone，并且两条分支都保留反向传播图。
在 imgsz=1920、global batch=32、per-GPU batch=8 下，显存接近双倍 baseline，因此 95GB 显存仍然不够。
```

修复内容：

```text
1. raw 分支改为 torch.no_grad()，只作为 EAFC 可靠性判断和差异参考。
2. raw_features 显式 detach，避免保存 raw backbone 的反向图。
3. enhanced 分支、UIAE、EAFC 和检测头仍然正常参与反向传播，保持 full AEFC 端到端训练。
```

涉及文件：

```text
AEFC-YOLO11/models/uiae_trainer.py
```

服务器需要上传：

```text
models/uiae_trainer.py
CODE_UPDATE_LOG.md
```

推荐训练名：

```text
experiment_d_full_aefc_fix1b
```

如果仍然 OOM：

```text
优先保持 imgsz=1920，将 batch 从 32 降到 24 或 16。
不建议首先降低分辨率，因为 baseline 对比是在 1920 下建立的。
```

GitHub 推送备注：

```text
待推送。建议提交信息:
Reduce full AEFC memory by detaching raw reference branch
```

## 2026-05-24 Baseline rebuild

版本/实验：

```text
YOLO11-M baseline reconstruction
```

触发问题：

```text
baseline 训练发生在创建 Git 仓库之前，需要重建一个单独 baseline 仓库，方便回溯 baseline 代码与配置。
```

修复内容：

```text
1. 在临时 worktree 中重建 baseline-only 版本。
2. 移除 UIAE/EAFC/MDCT 相关实验模块和消融配置。
3. 保留 YOLO11-M baseline 训练入口、配置、数据准备脚本、yolo11m.pt 权重和一张样例图。
4. 新增纯 baseline 训练脚本 tools/train_baseline.py。
```

推送仓库：

```text
https://github.com/yurenziraozhi/Rubust-EI-baseline.git
```

GitHub 推送备注：

```text
Commit: 3cc1b2c Reconstruct YOLO11-M baseline repo
Branch: main
```

结果/后续观察：

```text
baseline 仓库与当前 D-FULL 主代码隔离，临时 worktree 已删除。
```

## 2026-05-23 D-FULL staged diagnostics

版本/实验：

```text
experiment_d_full_aefc_staged_diag
```

触发问题：

```text
D 组全套方案早期训练出现指标全 0 或明显失效，需要增加训练过程诊断。
```

修复内容：

```text
1. 将 UIAE、EAFC 和 MDCT 接入 full AEFC 训练路径。
2. 增加内部 JSONL 日志和 epoch CSV 输出。
3. 增加输入、增强图、增强差值、EAFC attention、预测张量统计。
4. 初始采用 staged 策略，尝试先冻结 UIAE 若干轮。
```

涉及文件：

```text
AEFC-YOLO11/models/uiae_trainer.py
AEFC-YOLO11/models/eafc.py
AEFC-YOLO11/models/uiae.py
AEFC-YOLO11/tools/train_aefc.py
AEFC-YOLO11/configs/train_aefc_d.yaml
```

GitHub 推送备注：

```text
Commit: 71c2346 Use staged full AEFC training and internal diagnostics
Remote: dfull/main
Repository: https://github.com/yurenziraozhi/--EI-Waterscene-D-FULL-.git
```

结果/后续观察：

```text
服务器训练反馈 D 组 13 轮后 mAP 仍不如 baseline，因此进入 2026-05-24 D-FULL fix1。
```

## 2026-05-23 B 组 UIAE 消融调试

版本/实验：

```text
ablation_b_uiae
```

触发问题：

```text
B 组 UIAE 消融训练过程中多次出现 DDP import、DDP unused parameters、inplace gradient、指标退化等问题。
```

主要处理：

```text
1. 修复 DDP 子进程找不到 models 包的问题。
2. 修复 DistributedDataParallel 包裹后访问 self.model.uiae 的问题。
3. 避免 inplace 激活和 BatchNorm 状态干扰。
4. 尝试 find_unused_parameters=True 解决 unused parameters。
5. 观察到 UIAE-only 训练指标退化，暂不作为最终方案。
```

GitHub 推送备注：

```text
部分 B 组代码曾推送到:
https://github.com/yurenziraozhi/--EI-Waterscne--B.git
```

结果/后续观察：

```text
用户决定简化实验，只保留 UIAE、EAFC、Full 三组消融/全套实验。
```

## 2026-05-22 YOLO11-M baseline

版本/实验：

```text
yolo11m_baseline
```

训练配置：

```text
model: weights/yolo11m.pt
imgsz: 1920
batch: 32
device: 0,1,2,3
optimizer: AdamW
lr0: 0.001
lrf: 0.01
amp: false
epochs: 200
```

记录结果：

```text
Images: 10824
Instances: 40517
Precision: 0.711
Recall: 0.562
mAP50: 0.629
mAP50-95: 0.373
```

备注：

```text
baseline 训练发生在 Git 仓库正式建立之前，后续通过 baseline rebuild 单独恢复并上传。
```
