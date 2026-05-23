# 鲁棒目标检测网络模型技术文档：AEFC-YOLO11

## 0. 文档目标

本文档用于指导从零开始搭建一个面向恶劣视觉环境的鲁棒目标检测网络，适用于 EI 会议论文实验实现与方法撰写。

模型暂命名为：**AEFC-YOLO11**。

全称建议：

**Unified Image-Adaptive Enhancement and Feature Calibration YOLO11 for Robust Object Detection under Adverse Visual Conditions**

中文名称：

**面向恶劣视觉环境鲁棒目标检测的统一图像自适应增强与特征校准 YOLO11 网络**

本文档围绕三个创新点展开：

1. **UIAE：Unified Image-Adaptive Enhancement Module**  
   统一图像自适应增强模块。借鉴 IA-YOLO 的端到端图像自适应训练框架，并吸收 ERUP-YOLO 的统一滤波思想，用 **BPW 全局像素映射 + KBL 局部自适应滤波** 替代多个手工语义滤波器，实现对低光、雾、雨、反光、模糊等多退化的统一处理。

2. **EAFC：Enhancement-Aware Feature Calibration Module**  
   增强感知特征校准模块。用于解决 BPW/KBL 统一增强可能引入的过曝、伪影、背景噪声放大和水面反光误增强问题，在特征层自适应选择原图特征与增强图特征。

3. **MDCT：Multi-Degradation Consistency Training Strategy**  
   多退化一致性训练策略。通过显式退化视图和 BPW 随机参数增强视图约束模型输出稳定，提高模型对 WaterScenes 复杂视觉退化的泛化能力。

推荐将第 1 点作为论文最核心创新点，第 2 点作为区别于 ERUP-YOLO 的关键扩展，第 3 点作为训练策略或辅助贡献。

---

## 1. 研究问题定义

### 1.1 任务背景

普通目标检测模型在正常光照、清晰图像上表现较好，但在以下场景下容易失效：

- 雾天场景：目标边缘模糊，局部对比度下降；
- 低光照场景：目标亮度低，纹理信息弱；
- 雨天场景：雨纹、雨雾遮挡目标；
- 水面场景：水面反光、波纹、背景干扰强；
- 远距离小目标：目标尺度小，细节缺失；
- 夜间或逆光：目标与背景对比度极低。

因此，本课题的目标不是单纯提高普通检测精度，而是提高模型在恶劣视觉条件下的**鲁棒检测能力**。

### 1.2 核心思想

传统图像增强方法通常追求人眼视觉效果，但视觉质量提升并不一定带来检测性能提升。

本模型的核心思想是：

> 图像增强应该服务于目标检测，而不是单纯服务于人眼视觉观感。

因此，AEFC-YOLO11 采用如下策略：

1. 在输入端进行自适应图像增强；
2. 在特征层判断增强特征是否可靠；
3. 在训练阶段通过多退化一致性约束提高模型稳定性。

---

## 2. 整体网络结构

### 2.1 总体流程

AEFC-YOLO11 的整体结构如下：

```text
Input Image I
    │
    ├── Branch 1: Original Image I
    │        ↓
    │     Backbone
    │        ↓
    │  Original Features F_raw
    │
    └── Branch 2: UIAE Enhancement Module
             ↓
       Enhanced Image I_enh
             ↓
          Backbone
             ↓
       Enhanced Features F_enh

F_raw + F_enh
      ↓
EAFC Feature Calibration Module
      ↓
Robust Multi-scale Features
      ↓
Neck / FPN / PAN
      ↓
YOLO Detection Head
      ↓
Detection Results
```

如果考虑计算量，可以采用共享 backbone 的简化版本：

```text
Input Image I
      ↓
UIAE Enhancement Module
      ↓
Enhanced Image I_enh
      ↓
Backbone
      ↓
Multi-scale Features
      ↓
EAFC Calibration
      ↓
Detection Head
```

但为了体现创新性，推荐使用“双输入特征校准结构”：

- 原图分支保留真实纹理；
- 增强图分支提供更清晰的目标边缘和亮度信息；
- EAFC 模块自适应融合两类特征。

### 2.2 推荐网络版本

本项目建议以 **Ultralytics YOLO11-M** 作为默认基础检测器，命名和配置统一如下：

```text
Baseline: YOLO11-M
Baseline weight/config: yolo11m.pt / yolo11m.yaml
Proposed: AEFC-YOLO11-M
Ablation-1: UIAE-YOLO11-M
Ablation-2: EAFC-YOLO11-M
```

如果显存或训练时间不足，可以使用轻量版本：

```text
Baseline: YOLO11-S
Baseline weight/config: yolo11s.pt / yolo11s.yaml
Proposed: AEFC-YOLO11-S
```

YOLOv8-M 可以作为补充对比模型，但本文档后续命令和代码结构默认围绕 YOLO11-M 展开，避免 YOLOv8、YOLOv11 混用导致配置不一致。论文中可以写：

> To ensure a fair comparison, the proposed modules are inserted into the YOLO11 baseline detector while keeping the detection head and label definition unchanged.

也就是：为了公平比较，只修改输入增强、特征校准和训练策略，不改变检测头类别定义。

## 3. 创新点一：UIAE 统一图像自适应增强模块

### 3.1 模块目标

UIAE 的目标是针对每张输入图像，自适应生成统一图像处理参数，使增强后的图像更有利于目标检测。

旧版多滤波器增强方案采用亮度、对比度、Gamma、白平衡、锐化、反光抑制、平滑等多个传统可微滤波器串联。该设计解释性强，但存在三个问题：

1. 每个滤波器都需要人工设定参数范围；
2. 多滤波器串联容易产生相互干扰，例如低光增强可能放大水面反光；
3. 对低光、雾、雨、反光、模糊等混合退化，固定滤波器组合不一定稳定。

参考 ERUP-YOLO 的思想，本文将 UIAE 设计为统一图像自适应增强模块：不再显式堆叠多个传统滤波器，而是用两个可微滤波器统一表达多种图像处理能力：

1. **BPW：Bezier curve-based Pixel-wise Filter**，负责全局像素强度映射；
2. **KBL：Kernel-based Local Filter**，负责局部空间结构调整。

这样可以把“全局亮度/色调问题”和“局部雾化/锐化/纹理问题”分开建模，同时保持端到端训练。

模块输入输出如下：

```text
Input:  I ∈ R^{H×W×3}
Output: I_enh ∈ R^{H×W×3}
        P = {P_bpw, P_kbl}
```

其中：

- `I` 是原始输入图像；
- `I_enh` 是增强后的图像；
- `P_bpw` 是 BPW 全局像素映射参数；
- `P_kbl` 是 KBL 局部卷积调整参数。

### 3.2 与 IA-YOLO 和 ERUP-YOLO 的关系

`Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions` 的核心思想可以概括为：

1. 使用一个小型 CNN 参数预测网络（CNN-PP）从低分辨率图像中预测图像处理参数；
2. 使用可微图像处理模块（DIP）对高分辨率输入图像做自适应增强；
3. 将增强图送入 YOLO，并用检测损失端到端优化 CNN-PP、DIP 和检测器；
4. 训练时混合正常图像、合成雾图、低光图，使模型同时适应正常和恶劣视觉条件；
5. 实验重点报告正常测试集、合成雾测试集、真实雾数据集以及低光数据集上的检测性能。

IA-YOLO 中 DIP 的滤波器组合为 Defog、White Balance、Gamma、Contrast、Tone、Sharpen，其中 Defog 主要服务于雾天场景。ERUP-YOLO 进一步指出：这些传统滤波器可以抽象成两类统一操作，即全局像素映射和局部卷积调整。因此，本文采用“IA-YOLO 的端到端图像自适应训练框架 + ERUP-YOLO 的统一滤波器表达”。

| 方法 | 核心设计 | 局限或启发 |
|---|---|---|
| IA-YOLO | 使用 Defog、WB、Gamma、Contrast、Tone、Sharpen 六个可微滤波器 | 需要人工选择滤波器组合和参数范围 |
| ERUP-YOLO | 使用 BPW + KBL 两个统一滤波器 | 统一滤波更简洁，但增强过度时仍可能影响检测 |
| 本文 AEFC-YOLO11 | 使用 BPW + KBL 做统一增强，再用 EAFC 做特征可靠性校准 | 兼顾统一增强和增强伪影抑制 |

因此，本文的 UIAE 不再是多个传统滤波器的拼接，而是一个统一图像自适应处理模块。它吸收 ERUP-YOLO 的两个关键优点：

1. 用 BPW 统一 Gamma、对比度、Tone、低光增强等全局像素级操作；
2. 用 KBL 统一锐化、去雾、局部去噪、局部纹理调整等空间邻域操作。

同时，本文保留 EAFC 模块，用于解决 ERUP-YOLO 中可能出现的增强过度问题。例如，ERUP-YOLO 指出 BPW 在雾、沙尘等高亮散射场景中可能导致远处区域过曝；WaterScenes 中的水面反光也有类似风险。因此，EAFC 用于在特征层判断增强特征是否可靠，而不是无条件相信增强图。

### 3.3 UIAE 的结构设计

UIAE 包含三个部分：

```text
Low-resolution Image
      ↓
Parameter Prediction Network, PPN
      ↓
P_bpw, P_kbl
      ↓
BPW Global Pixel-wise Filter
      ↓
KBL Local Adaptive Filter
      ↓
Enhanced Image
```

#### 3.3.1 参数预测网络 PPN

PPN 输入低分辨率图像，例如 256×256，输出 BPW 和 KBL 两组增强参数。

推荐结构：

```text
Input: 256×256×3
Conv 3×3, stride=2, channels=16 + BN + SiLU
Conv 3×3, stride=2, channels=32 + BN + SiLU
Conv 3×3, stride=2, channels=64 + BN + SiLU
Conv 3×3, stride=2, channels=64 + BN + SiLU
Global Average Pooling
FC 64 → 32 + SiLU
FC 32 → K_bpw + K_kbl
Sigmoid / Tanh parameter mapping
```

其中：

- `K_bpw` 是 BPW 参数数量；
- `K_kbl` 是 KBL 参数数量。

```text
K_bpw = 12
K_kbl = 2 × 9 × 9 × 3 = 486
```

BPW 使用每个 RGB 通道 4 个参数，共 12 个参数。KBL 使用两个 9×9 局部卷积核，每个通道单独预测，因此共 486 个参数。

如果担心参数量偏大，可以采用轻量版 KBL。当前工程代码默认使用该轻量版：

```text
K_kbl_light = 2 × 5 × 5 × 3 = 150
```

#### 3.3.2 参数范围设计

PPN 输出需要限制到合理范围，避免增强过度。推荐设计为“零参数对应恒等映射”，使模型在正常图像上可以选择不增强。

```text
P_bpw = tanh(z_bpw) × α_bpw
P_kbl = tanh(z_kbl) × α_kbl
```

推荐初始范围：

```text
α_bpw = 0.5
α_kbl = 0.1
```

训练稳定后可以逐步放大 KBL 范围，例如将 `α_kbl` 从 0.1 提升到 0.2。

### 3.4 可微统一滤波器设计

#### 3.4.1 BPW 全局像素映射滤波器

BPW 使用三次 Bezier 曲线表达输入像素强度到输出像素强度的映射。起点和终点固定为：

```text
[0, 0], [1, 1]
```

中间控制点由 PPN 预测。对每个颜色通道分别建模：

```text
Po_c = BPW_c(Pi_c; θ_c), c ∈ {R, G, B}
```

其中：

- `Pi_c` 是输入像素强度；
- `Po_c` 是输出像素强度；
- `θ_c` 是该通道的 Bezier 控制参数。

BPW 的作用是统一表达：

```text
Gamma
Contrast
Tone mapping
Brightness adjustment
Low-light enhancement
White balance tendency
```

优点是避免单独设计 Gamma、Contrast、Tone 等多个滤波器及其参数范围。

#### 3.4.2 KBL 局部自适应滤波器

KBL 用局部卷积形式统一表达去雾、锐化、局部纹理增强和局部去噪：

```text
I_kbl = I_bpw * Conv(I_bpw, K1) + Conv(I_bpw, K2) + I_bpw
```

其中：

- `K1` 和 `K2` 是 PPN 预测的局部卷积核；
- `*` 表示逐像素乘法；
- `Conv` 是按通道执行的局部卷积。

KBL 的作用是统一表达：

```text
Sharpen
Defog
Local denoise
Local contrast adjustment
Texture recovery
Reflection-region local correction
```

对 WaterScenes 而言，KBL 特别适合处理水面波纹、雨雾模糊、目标边缘弱化和局部反光干扰。

#### 3.4.3 最终增强图

最终增强图为：

```text
I_bpw = BPW(I; P_bpw)
I_enh = clamp(KBL(I_bpw; P_kbl), 0, 1)
```

为了避免增强过度，训练时对 `P_bpw` 和 `P_kbl` 加入参数正则。

### 3.5 UIAE PyTorch 伪代码

```python
class ParameterPredictor(nn.Module):
    def __init__(self, bpw_dim=12, kbl_dim=486):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, bpw_dim + kbl_dim)
        )
        self.bpw_dim = bpw_dim

    def forward(self, x):
        x_small = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        z = self.fc(self.features(x_small))
        z_bpw = torch.tanh(z[:, :self.bpw_dim]) * 0.5
        z_kbl = torch.tanh(z[:, self.bpw_dim:]) * 0.1
        return z_bpw, z_kbl
```

```python
class UIAE(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.ppn = ParameterPredictor(
            bpw_dim=12,
            kbl_dim=2 * kernel_size * kernel_size * 3
        )

    def forward(self, x):
        p_bpw, p_kbl = self.ppn(x)
        x_bpw = bpw_filter(x, p_bpw)
        x_kbl = kbl_filter(x_bpw, p_kbl, self.kernel_size)
        return torch.clamp(x_kbl, 0, 1), {"bpw": p_bpw, "kbl": p_kbl}
```

注意：`bpw_filter` 和 `kbl_filter` 需要在工程实现中单独写成可微模块。BPW 推荐用分段线性近似 Bezier 曲线，KBL 推荐用 `unfold + grouped convolution` 或按样本循环卷积实现动态核。

---

## 4. 创新点二：EAFC 增强感知特征校准模块

### 4.1 模块动机

图像增强不是绝对可靠的。

在恶劣环境下，增强模块可能带来以下问题：

- 雾天图像去雾后产生颜色偏移；
- 暗光图像增强后噪声被放大；
- 锐化后目标边缘更清楚，但背景纹理也被增强；
- 水面反光抑制可能误伤真实目标高亮部分。

因此，不能直接把增强图像作为唯一输入。

EAFC 的目标是：

> 在特征层自适应判断原图特征和增强图特征的可靠性，并融合二者。

### 4.2 输入输出

对于 backbone 输出的多尺度特征：

```text
F_raw  = {F_raw3, F_raw4, F_raw5}
F_enh  = {F_enh3, F_enh4, F_enh5}
```

EAFC 输出：

```text
F_out = {F_out3, F_out4, F_out5}
```

其中：

- P3：小目标特征，分辨率最高；
- P4：中等目标特征；
- P5：大目标特征，语义最强。

### 4.3 融合公式

基础融合公式：

```text
A_s = sigmoid(Conv([F_raw_s, F_enh_s, F_enh_s - F_raw_s]))
```

```text
F_out_s = A_s · F_enh_s + (1 - A_s) · F_raw_s
```

其中：

- `s ∈ {3,4,5}` 表示特征尺度；
- `A_s` 是增强特征可靠性权重；
- `A_s` 越大，越相信增强特征；
- `A_s` 越小，越相信原图特征。

进一步可以加入残差：

```text
F_out_s = F_raw_s + A_s · (F_enh_s - F_raw_s)
```

这个形式更稳定，推荐使用。

### 4.4 EAFC 结构

每个尺度使用一个轻量校准模块：

```text
Input: concat(F_raw, F_enh, F_enh - F_raw)
       ↓
1×1 Conv 降维
       ↓
3×3 Depthwise Conv
       ↓
1×1 Conv
       ↓
Sigmoid
       ↓
Attention Map A
```

### 4.5 EAFC PyTorch 伪代码

```python
class EAFC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_raw, f_enh):
        diff = f_enh - f_raw
        x = torch.cat([f_raw, f_enh, diff], dim=1)
        a = self.attn(x)
        f_out = f_raw + a * diff
        return f_out, a
```

### 4.6 多尺度 EAFC 插入位置

推荐插入在 backbone 输出到 neck 之前：

```text
Backbone P3_raw, P4_raw, P5_raw
Backbone P3_enh, P4_enh, P5_enh
        ↓
EAFC on P3, P4, P5
        ↓
Neck / FPN / PAN
        ↓
Detection Head
```

如果使用 YOLOv8，可以插在 backbone 的 C2f 输出之后、neck 输入之前。

如果使用 YOLOv11，可以插在 backbone 主干输出的多尺度特征与 neck 之间。

---

## 5. 创新点三：MDCT 多退化一致性训练策略

### 5.1 模块目标

MDCT 的目标是提高模型在不同退化条件下的预测稳定性。

同一张图像经过不同退化处理后，目标类别和位置应该保持一致。

例如：

```text
I_clean
I_foggy
I_dark
I_rainy
I_blur
```

这些图像中的同一目标应该被检测为相同类别，检测框位置也应该接近。

### 5.2 多退化生成方式

训练阶段对输入图像随机生成退化版本。

推荐退化类型：

| 退化类型 | 实现方式 |
|---|---|
| 低光 | `I_dark = I ^ gamma`, gamma ∈ [1.5, 4.0] |
| 雾天 | 大气散射模型或简单白雾叠加 |
| 模糊 | Gaussian blur |
| 雨天 | 随机雨线叠加 |
| 反光 | 局部高亮区域增强 |
| 噪声 | Gaussian noise |

低光退化：

```text
I_dark = I ^ γ
```

雾天退化简化版：

```text
I_fog = I · t + A · (1 - t)
```

其中：

```text
t ∈ [0.4, 0.9]
A ∈ [0.7, 1.0]
```

### 5.3 一致性损失

总损失：

```text
L_total = L_det + λ1 L_cons + λ2 L_param + λ3 L_smooth
```

其中：

- `L_det`：检测损失；
- `L_cons`：多退化一致性损失；
- `L_param`：增强参数正则；
- `L_smooth`：特征校准权重平滑约束。

### 5.4 检测损失

如果使用 YOLO11，检测损失一般包括：

```text
L_det = L_box + L_cls + L_dfl
```

其中：

- `L_box`：边界框回归损失；
- `L_cls`：分类损失；
- `L_dfl`：Distribution Focal Loss。

### 5.5 一致性损失设计

简化版一致性损失可以使用特征一致性：

```text
L_cons = || GAP(F_clean) - GAP(F_degraded) ||_2
```

其中：

- `GAP` 是全局平均池化；
- `F_clean` 是正常图像特征；
- `F_degraded` 是退化图像特征。

更简单的实现：

只在训练后半阶段启用一致性损失。

```text
Epoch 0-50:   只使用 L_det
Epoch 50-end: 使用 L_det + L_cons
```

这样训练更稳定。

### 5.6 参数正则

为了避免 UIAE 过度增强，可以加入参数正则：

```text
L_param = mean((P - P_identity)^2)
```

其中 `P_identity` 表示不增强时的参数。

例如：

```text
brightness = 1
contrast = 1
gamma = 1
sharpness = 0
reflection = 0
white balance = 1
```

参数正则的目的不是禁止增强，而是防止增强过度。

---

## 6. 项目代码结构设计

当前工作目录已经包含原始 WaterScenes 检测数据：

```text
waterscene-鲁棒检测/
├── image/                    # WaterScenes 全量图像，共 54120 张 jpg
├── detection/
│   └── yolo/                 # YOLO 格式标签，共 54120 个 txt
├── train.txt                 # 固定训练集编号
├── val.txt                   # 固定验证集编号
├── test.txt                  # 固定测试集编号
├── adverse_lighting.txt      # 低光/光照退化测试编号，5463 行，5456 个唯一编号
├── adverse_weather.txt       # 恶劣天气测试编号，11331 行，11321 个唯一编号
└── Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions.pdf
```

为了让 Ultralytics YOLO11 直接训练，需要额外生成标准 YOLO 数据目录。建议工程目录如下：

```text
AEFC-YOLO11/
├── configs/
│   ├── waterscenes_full.yaml
│   ├── waterscenes_adverse_lighting.yaml
│   ├── waterscenes_adverse_weather.yaml
│   ├── yolo11m_aefc.yaml
│   └── train_aefc.yaml
├── datasets/
│   └── waterscenes_yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   ├── test/
│       │   ├── adverse_lighting/
│       │   └── adverse_weather/
│       └── labels/
│           ├── train/
│           ├── val/
│           ├── test/
│           ├── adverse_lighting/
│           └── adverse_weather/
├── models/
│   ├── __init__.py
│   ├── uiae.py
│   ├── eafc.py
│   ├── aefc_yolo.py
│   ├── degradation.py
│   └── losses.py
├── tools/
│   ├── prepare_waterscenes_yolo.py
│   ├── train_aefc.py
│   ├── val_aefc.py
│   ├── infer_aefc.py
│   └── visualize_enhancement.py
├── runs/
└── README.md
```

`prepare_waterscenes_yolo.py` 负责把当前 `image/` 和 `detection/yolo/` 组织成 Ultralytics 标准目录：

- `images/train`、`images/val`、`images/test` 分别按 `train.txt`、`val.txt`、`test.txt` 中的编号生成，作为固定训练、验证和测试划分；
- `images/adverse_lighting` 与 `labels/adverse_lighting` 按 `adverse_lighting.txt` 的唯一编号生成，作为光照退化专项测试集；
- `images/adverse_weather` 与 `labels/adverse_weather` 按 `adverse_weather.txt` 的唯一编号生成，作为天气退化专项测试集；
- 图像复制或硬链接均可，标签从 `detection/yolo` 按同名编号复制。

### 6.1 核心文件说明

| 文件 | 作用 |
|---|---|
| `configs/waterscenes_full.yaml` | 全量 WaterScenes 训练配置 |
| `configs/waterscenes_adverse_lighting.yaml` | 光照退化专项测试配置 |
| `configs/waterscenes_adverse_weather.yaml` | 天气退化专项测试配置 |
| `configs/yolo11m_aefc.yaml` | YOLO11-M + UIAE + EAFC 网络结构配置 |
| `models/uiae.py` | 实现 BPW + KBL 统一退化感知图像增强模块 |
| `models/eafc.py` | 实现增强感知特征校准模块 |
| `models/aefc_yolo.py` | 组装完整 AEFC-YOLO11 网络 |
| `models/degradation.py` | 训练阶段生成低光、雾、雨、模糊、反光等退化视图，也可扩展 BPW 随机参数增强 |
| `models/losses.py` | 实现一致性损失和正则损失 |
| `tools/prepare_waterscenes_yolo.py` | 将当前目录数据整理为标准 YOLO 训练/测试目录 |
| `tools/train_aefc.py` | 训练主入口 |
| `tools/val_aefc.py` | 全量或专项测试入口 |
| `tools/infer_aefc.py` | 推理可视化 |
| `tools/visualize_enhancement.py` | 可视化 UIAE 增强前后效果 |

## 7. 数据集准备

### 7.1 当前数据来源

本项目使用 **WaterScenes 全量数据集** 训练。当前目录中的数据关系如下：

```text
image/00001.jpg
detection/yolo/00001.txt
```

每张图像对应一个 YOLO 标签文件：

```text
class_id x_center y_center width height
```

所有坐标归一化到 `[0, 1]`。当前已确认：

```text
图像数量: 54120
标签数量: 54120
图像格式: jpg
标签格式: YOLO txt
```

### 7.2 类别定义

WaterScenes 检测类别保持 7 类，训练、验证、推理和可视化必须统一使用以下顺序：

```yaml
names:
  0: pier
  1: buoy
  2: sailor
  3: ship
  4: boat
  5: vessel
  6: kayak
```

不要在可视化或 checkpoint 中保留 `class_0` 到 `class_6` 这类占位名称。

### 7.3 标准 YOLO 目录生成

Ultralytics 默认从 `images/` 路径推导对应的 `labels/` 路径。因此不能直接把当前 `image/` 与 `detection/yolo/` 原样交给训练命令，建议先生成标准目录：

```text
datasets/waterscenes_yolo/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── adverse_lighting/
│   └── adverse_weather/
└── labels/
    ├── train/
    ├── val/
    ├── test/
    ├── adverse_lighting/
    └── adverse_weather/
```

生成规则：

1. 训练集：读取 `train.txt`，按其中编号生成 `images/train` 和 `labels/train`；
2. 验证集：读取 `val.txt`，按其中编号生成 `images/val` 和 `labels/val`；
3. 测试集：读取 `test.txt`，按其中编号生成 `images/test` 和 `labels/test`；
4. 光照专项测试集：读取 `adverse_lighting.txt`，按唯一编号生成 `adverse_lighting` 图像和标签；
5. 天气专项测试集：读取 `adverse_weather.txt`，按唯一编号生成 `adverse_weather` 图像和标签；
6. 如果 txt 中有重复编号，只保留一份图像和一份标签；
7. 如果出现图像或标签缺失，脚本必须输出缺失清单并停止训练。

### 7.4 数据集配置文件

`configs/waterscenes_full.yaml`：

```yaml
path: datasets/waterscenes_yolo
train: images/train
val: images/val
test: images/test
names:
  0: pier
  1: buoy
  2: sailor
  3: ship
  4: boat
  5: vessel
  6: kayak
```

说明：当前训练、验证和测试划分直接以根目录中的 `train.txt`、`val.txt`、`test.txt` 为准，三者不能重新随机划分。训练阶段使用 `train`，调参和保存最佳权重使用 `val`，最终常规测试使用 `test`；`adverse_lighting` 和 `adverse_weather` 只作为额外鲁棒性专项测试集。

`configs/waterscenes_adverse_lighting.yaml`：

```yaml
path: datasets/waterscenes_yolo
train: images/train
val: images/adverse_lighting
test: images/adverse_lighting
names:
  0: pier
  1: buoy
  2: sailor
  3: ship
  4: boat
  5: vessel
  6: kayak
```

`configs/waterscenes_adverse_weather.yaml`：

```yaml
path: datasets/waterscenes_yolo
train: images/train
val: images/adverse_weather
test: images/adverse_weather
names:
  0: pier
  1: buoy
  2: sailor
  3: ship
  4: boat
  5: vessel
  6: kayak
```

### 7.5 恶劣环境测试逻辑

参考 IA-YOLO 的实验逻辑，训练时不只学习正常清晰图像，而是混合正常图像和合成退化图像；测试时同时报告正常/全量性能和专项恶劣环境性能。本文档采用如下映射：

| IA-YOLO 逻辑 | 本项目 WaterScenes 落地方式 |
|---|---|
| 正常数据参与训练，保证普通场景不退化 | WaterScenes 全量 54120 张图像作为训练基础 |
| 合成雾图、低光图参与训练，提高恶劣条件适应性 | `models/degradation.py` 在线生成 fog、dark、rain、blur、reflection、noise 视图 |
| 真实雾/低光数据单独测试 | `adverse_weather`、`adverse_lighting` 作为专项鲁棒测试集 |
| 检测损失弱监督增强参数 | UIAE 的 PPN 通过 YOLO 检测损失和参数正则共同优化 |

因此，本文实验至少报告三组结果：

| 测试集 | 来源 | 目的 |
|---|---|---|
| Full / Train-set check | `images/train` | 检查全量训练收敛和基础检测能力 |
| Adverse-Lighting | `adverse_lighting.txt` 对应图像 | 测试低光、逆光、照度不足等光照退化鲁棒性 |
| Adverse-Weather | `adverse_weather.txt` 对应图像 | 测试雾、雨、模糊、能见度差等天气退化鲁棒性 |

## 8. 训练配置

### 8.1 环境配置

推荐环境：

```text
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
torchvision
ultralytics >= 8.3
opencv-python
numpy
matplotlib
pyyaml
tqdm
scipy
pandas
```

`requirements.txt` 示例：

```text
torch>=2.0.0
torchvision
ultralytics>=8.3.0
opencv-python
numpy
matplotlib
pyyaml
tqdm
scipy
pandas
```

### 8.2 基础训练参数

默认训练配置：

```yaml
model: yolo11m.pt
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
```

`amp` 默认写为 `false`，是为了避免离线服务器触发额外权重下载或 AMP 检查失败。确认环境稳定后可以改为 `true`。

如果显存不足：

```yaml
model: yolo11s.pt
batch: 8
accumulate: 2
```

### 8.3 推荐训练阶段

建议分三阶段训练。

#### 阶段一：YOLO11-M baseline

目的：建立全量 WaterScenes 基准结果。

```text
Model: YOLO11-M
Epochs: 150-200
Input: 原始 WaterScenes 全量图像
Loss: YOLO11 detection loss
```

#### 阶段二：UIAE-YOLO11-M

目的：验证 BPW + KBL 统一自适应增强是否提升光照和天气专项测试集。

```text
Model: YOLO11-M + UIAE
Epochs: 150-200
Loss: L_det + λ2 L_param
λ2 = 0.01
```

训练策略：前 5-10 个 epoch 可以冻结 UIAE 或降低 UIAE 学习率，避免训练初期 BPW/KBL 参数剧烈波动。

#### 阶段三：AEFC-YOLO11-M

目的：训练最终模型。

```text
Model: YOLO11-M + UIAE + EAFC + MDCT
Epochs: 200
Loss: L_det + λ1 L_cons + λ2 L_param + λ3 L_smooth
λ1 = 0.1
λ2 = 0.01
λ3 = 0.005
```

前 50 个 epoch 不开启 `L_cons`，后 150 个 epoch 开启，降低一致性损失造成训练不稳定的风险。

### 8.4 数据增强配置

常规增强：

```yaml
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1
copy_paste: 0.0
close_mosaic: 20
```

鲁棒退化增强参考 IA-YOLO 和 ERUP-YOLO 的混合训练思想：每个 batch 中保留一部分原图，同时以一定概率生成退化视图。退化生成可以有两种方式：

1. 显式退化：低光、雾、雨、模糊、反光或噪声；
2. BPW 随机参数增强：随机采样 Bezier 像素映射，统一模拟低光、对比度变化、雾化亮度偏移等退化。

```yaml
robust_aug: true
p_degrade: 0.67
p_dark: 0.20
p_fog: 0.20
p_rain: 0.15
p_blur: 0.10
p_noise: 0.10
p_reflection: 0.12
bpw_aug: true
p_bpw_aug: 0.30
```

注意：`p_degrade` 控制一张图是否进入退化分支，各退化类型可单独采样或按权重采样，不能让所有图都变成退化图，否则正常场景性能可能下降。`bpw_aug` 用于替代一部分手工低光/雾增强，减少数据集特定增强策略。

## 9. 损失函数设计

### 9.1 总损失

最终模型训练损失：

```text
L_total = L_det + λ1 L_cons + λ2 L_param + λ3 L_smooth
```

### 9.2 检测损失

直接沿用 YOLO 检测损失：

```text
L_det = L_box + L_cls + L_dfl
```

不建议一开始修改检测头损失，否则实验变量太多，不利于消融分析。

### 9.3 一致性损失

推荐使用特征一致性损失：

```text
L_cons = mean(|| GAP(F_normal) - GAP(F_degraded) ||_2)
```

更稳定的版本：

```text
L_cons = 1 - cosine_similarity(GAP(F_normal), GAP(F_degraded))
```

### 9.4 参数正则损失

```text
L_param = mean((P - P_identity)^2)
```

目的是避免 UIAE 在正常图像上过度增强。

### 9.5 校准权重平滑损失

EAFC 会输出注意力图 `A`，可以加平滑约束：

```text
L_smooth = mean(|∂A/∂x| + |∂A/∂y|)
```

目的是避免校准权重在空间上剧烈波动。

---

## 10. 训练命令示例

### 10.1 生成标准 YOLO 数据目录

```bash
python tools/prepare_waterscenes_yolo.py \
  --root . \
  --image-dir image \
  --label-dir detection/yolo \
  --train-list train.txt \
  --val-list val.txt \
  --test-list test.txt \
  --lighting-list adverse_lighting.txt \
  --weather-list adverse_weather.txt \
  --out datasets/waterscenes_yolo \
  --mode copy
```

如果磁盘空间紧张，后续实现脚本时可以把 `--mode copy` 改成 `--mode hardlink`。

### 10.2 训练 YOLO11-M baseline

```bash
yolo detect train \
  model=yolo11m.pt \
  data=configs/waterscenes_full.yaml \
  imgsz=640 \
  epochs=200 \
  batch=16 \
  device=0 \
  workers=8 \
  optimizer=AdamW \
  lr0=0.001 \
  cos_lr=True \
  amp=False \
  project=runs/aefc_yolo11 \
  name=yolo11m_baseline
```

### 10.3 训练 UIAE-YOLO11-M

```bash
python tools/train_aefc.py \
  --model yolo11m.pt \
  --data configs/waterscenes_full.yaml \
  --imgsz 640 \
  --epochs 200 \
  --batch 16 \
  --device 0 \
  --use-uiae \
  --lambda-param 0.01 \
  --project runs/aefc_yolo11 \
  --name uiae_yolo11m
```

### 10.4 训练 AEFC-YOLO11-M

```bash
python tools/train_aefc.py \
  --model yolo11m.pt \
  --data configs/waterscenes_full.yaml \
  --imgsz 640 \
  --epochs 200 \
  --batch 16 \
  --device 0 \
  --use-uiae \
  --use-eafc \
  --use-mdct \
  --lambda-cons 0.1 \
  --lambda-param 0.01 \
  --lambda-smooth 0.005 \
  --cons-start-epoch 50 \
  --project runs/aefc_yolo11 \
  --name aefc_yolo11m
```

### 10.5 全量检查与专项测试

全量检查：

```bash
yolo detect val \
  model=runs/aefc_yolo11/yolo11m_baseline/weights/best.pt \
  data=configs/waterscenes_full.yaml \
  imgsz=640 \
  split=val
```

光照退化专项测试：

```bash
yolo detect val \
  model=runs/aefc_yolo11/aefc_yolo11m/weights/best.pt \
  data=configs/waterscenes_adverse_lighting.yaml \
  imgsz=640 \
  split=val
```

天气退化专项测试：

```bash
yolo detect val \
  model=runs/aefc_yolo11/aefc_yolo11m/weights/best.pt \
  data=configs/waterscenes_adverse_weather.yaml \
  imgsz=640 \
  split=val
```

### 10.6 推理可视化

```bash
python tools/infer_aefc.py \
  --weights runs/aefc_yolo11/aefc_yolo11m/weights/best.pt \
  --source adverse_weather \
  --imgsz 640 \
  --save \
  --save-enhanced \
  --save-attention
```

建议保存三类可视化：

```text
原始图像检测结果
增强图像检测结果
EAFC 注意力图
```

这样论文图更有说服力。

## 11. 实验设计

### 11.1 对比实验

建议对比以下模型：

| 方法 | 说明 |
|---|---|
| Faster R-CNN | 双阶段检测器 |
| RT-DETR | Transformer 检测器 |
| YOLOX-M | 单阶段检测器 |
| YOLOv8-M | 强 baseline |
| YOLO11-M | 最新 YOLO baseline，可选 |
| IA-YOLO-style | 多传统滤波器图像自适应增强 |
| ERUP-YOLO-style | BPW + KBL 统一图像自适应增强 |
| UIAE-YOLO11-M | 本文统一增强模块 |
| AEFC-YOLO11-M | 本文完整模型：统一增强 + 特征校准 |

如果篇幅有限，至少对比：

```text
YOLOv8-M
YOLO11-M
UIAE-YOLO11
AEFC-YOLO11
```

### 11.2 消融实验

消融实验建议如下：

| 编号 | BPW | KBL | EAFC | MDCT | 目的 |
|---|---|---|---|---|---|
| A | × | × | × | × | YOLO11 baseline |
| B | ✓ | × | × | × | 验证全局像素映射 |
| C | × | ✓ | × | × | 验证局部自适应滤波 |
| D | ✓ | ✓ | × | × | 验证统一增强模块 |
| E | ✓ | ✓ | ✓ | × | 验证增强特征校准 |
| F | ✓ | ✓ | ✓ | ✓ | 完整模型 |

建议表格指标：

```text
mAP@0.5
mAP@0.5:0.95
Precision
Recall
FPS
Params
GFLOPs
```

### 11.3 恶劣环境子集实验

专项测试集直接使用当前目录中已经生成的两类图像：

| 子集 | 图像来源 | 唯一图像数 | 评价目的 |
|---|---|---:|---|
| Adverse-Lighting | `adverse_lighting.txt` / `adverse_lighting/` | 5456 | 低光、逆光、照度不足等光照退化 |
| Adverse-Weather | `adverse_weather.txt` / `adverse_weather/` | 11321 | 雾、雨、模糊、能见度差等天气退化 |

建议表格：

| 方法 | Full check | Adverse-Lighting | Adverse-Weather | 平均专项 mAP |
|---|---:|---:|---:|---:|
| YOLO11-M |  |  |  |  |
| UIAE-YOLO11-M |  |  |  |  |
| AEFC-YOLO11-M |  |  |  |  |

其中 Full check 用于确认模型在全量 WaterScenes 上已充分收敛；论文中真正体现鲁棒性的核心指标是两个专项测试集。

### 11.4 类别级 AP 实验

WaterScenes 类别可以写：

```text
Pier, Buoy, Sailor, Ship, Boat, Vessel, Kayak
```

表格格式：

| Method | Pier | Buoy | Sailor | Ship | Boat | Vessel | Kayak | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8-M |  |  |  |  |  |  |  |  |
| AEFC-YOLO11 |  |  |  |  |  |  |  |  |

### 11.5 可视化实验

建议论文放 3 类可视化图：

#### 图 1：整体网络结构图

重点突出：

```text
UIAE
EAFC
MDCT
```

#### 图 2：增强效果图

展示：

```text
原图
增强图
检测结果
```

#### 图 3：EAFC 注意力图

展示模型在哪些区域更信任增强特征。

可以显示：

```text
原图
增强图
EAFC attention map
最终检测结果
```

---

## 12. 评价指标

### 12.1 检测指标

主指标：

```text
mAP@0.5
mAP@0.5:0.95
Precision
Recall
```

鲁棒检测建议重点报告：

```text
Full mAP@0.5:0.95
Adverse-Lighting mAP@0.5:0.95
Adverse-Weather mAP@0.5:0.95
Adverse-Lighting Recall
Adverse-Weather Recall
```

恶劣环境下更容易漏检，所以 Recall 必须和 mAP 一起报告。

### 12.2 鲁棒性增益指标

为了突出 AEFC-YOLO11 相对 baseline 的专项收益，建议额外计算：

```text
ΔmAP_lighting = mAP_AEFC_lighting - mAP_YOLO11_lighting
ΔmAP_weather  = mAP_AEFC_weather  - mAP_YOLO11_weather
ΔRecall_lighting = Recall_AEFC_lighting - Recall_YOLO11_lighting
ΔRecall_weather  = Recall_AEFC_weather  - Recall_YOLO11_weather
```

如果全量 check 小幅波动但两个专项测试集明显提升，可以在论文中解释为：模型优化目标更关注恶劣视觉场景下的可见性恢复和可靠特征选择。

### 12.3 效率指标

需要报告：

```text
Params
GFLOPs
FPS
Inference time
```

建议说明：

> 虽然 AEFC-YOLO11 引入了 UIAE 和 EAFC，但 BPW/KBL 统一滤波器结构紧凑，参数量和计算量增加可控，仍满足实际部署需求。

## 13. 论文方法部分写法建议

### 13.1 总体方法描述

可以写：

> To improve object detection robustness under adverse visual conditions, we propose AEFC-YOLO11, a unified image-adaptive enhancement and feature calibration framework. The proposed method consists of a unified degradation-aware image enhancement module, an enhancement-aware feature calibration module, and a multi-degradation consistency training strategy. The enhancement module uses a Bezier curve-based pixel-wise filter and a kernel-based local filter to adaptively handle global intensity degradation and local structure degradation, while the feature calibration module suppresses unreliable enhancement artifacts by fusing original and enhanced features. Furthermore, the consistency training strategy encourages stable predictions under diverse image degradations.

### 13.2 创新点一：统一图像自适应增强

可以写：

> Existing image-adaptive detectors usually rely on multiple manually designed filters, such as gamma correction, white balance, contrast enhancement, tone mapping, defogging and sharpening. These filters require condition-dependent combinations and parameter ranges, and their sequential interaction may introduce unstable enhancement artifacts. To address this issue, we design a unified image-adaptive enhancement module with two differentiable filters: a Bezier curve-based pixel-wise filter for global intensity mapping and a kernel-based local filter for local structure adjustment. This design unifies pixel-wise enhancement, low-light correction, local sharpening, defogging and denoising into a compact end-to-end trainable preprocessing module optimized by detection loss.

### 13.3 创新点二：增强感知特征校准

可以写：

> Although unified image-adaptive enhancement improves degraded images, it may still cause over-exposure in foggy or reflective water-surface regions and amplify background noise. Instead of directly replacing the original image with the enhanced image, we introduce an enhancement-aware feature calibration module. The module estimates the reliability of enhanced features at multiple scales and adaptively fuses them with original features, allowing the detector to benefit from useful enhancement while suppressing unreliable artifacts.

### 13.4 创新点三：多退化一致性训练

可以写：

> To further improve robustness, we introduce a multi-degradation consistency training strategy. During training, explicit degraded views and BPW-augmented views of the same image are generated, and the detector is encouraged to learn degradation-invariant representations through feature-level consistency constraints. This strategy improves prediction stability under low-light, foggy, rainy, blurred and reflective water-surface conditions.

---

## 14. 推荐论文贡献写法

论文贡献可以写成三点：

1. We propose AEFC-YOLO11, a unified image-adaptive enhancement and feature calibration framework for robust object detection under adverse visual conditions.

2. We design a unified degradation-aware enhancement module based on BPW and KBL filters, replacing multiple manually combined conventional filters with a compact end-to-end trainable module for global intensity mapping and local structure correction.

3. We introduce an enhancement-aware feature calibration module and a multi-degradation consistency training strategy to suppress enhancement-induced artifacts and improve detection stability under complex WaterScenes degradations.

中文解释：

1. 提出 AEFC-YOLO11，一种面向恶劣视觉环境鲁棒目标检测的统一图像自适应增强与特征校准框架；
2. 设计基于 BPW 和 KBL 的统一退化感知增强模块，用紧凑的端到端可训练模块替代多个手工组合的传统滤波器，实现全局像素映射和局部结构校正；
3. 引入增强感知特征校准模块和多退化一致性训练策略，抑制统一增强可能带来的过曝、伪影和噪声放大，并提高模型在 WaterScenes 复杂退化场景下的稳定性。

---

## 15. 推荐实验表格清单

EI 论文建议至少准备以下表格：

### 表 1：实验环境

| Item | Configuration |
|---|---|
| GPU | NVIDIA RTX 3090 / A30 / V100 |
| CPU | Intel Xeon / AMD EPYC |
| Framework | PyTorch + Ultralytics |
| CUDA | 11.8 / 12.x |
| Input size | 640×640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Epochs | 200 |

### 表 2：主对比实验

| Method | Params | GFLOPs | Full mAP@0.5 | Full mAP@0.5:0.95 | FPS |
|---|---:|---:|---:|---:|---:|
| YOLO11-M |  |  |  |  |  |
| YOLOv8-M |  |  |  |  |  |
| RT-DETR |  |  |  |  |  |
| UIAE-YOLO11-M |  |  |  |  |  |
| AEFC-YOLO11-M |  |  |  |  |  |

### 表 3：恶劣环境专项测试

| Method | Adverse-Lighting mAP@0.5 | Adverse-Lighting Recall | Adverse-Weather mAP@0.5 | Adverse-Weather Recall |
|---|---:|---:|---:|---:|
| YOLO11-M |  |  |  |  |
| UIAE-YOLO11-M |  |  |  |  |
| AEFC-YOLO11-M |  |  |  |  |

### 表 4：消融实验

| BPW | KBL | EAFC | MDCT | Full mAP@0.5 | Lighting mAP@0.5 | Weather mAP@0.5 | Recall |
|---|---|---|---|---:|---:|---:|---:|
| × | × | × | × |  |  |  |  |
| ✓ | × | × | × |  |  |  |  |
| × | ✓ | × | × |  |  |  |  |
| ✓ | ✓ | × | × |  |  |  |  |
| ✓ | ✓ | ✓ | × |  |  |  |  |
| ✓ | ✓ | ✓ | ✓ |  |  |  |  |

### 表 5：类别级 AP

| Method | Pier | Buoy | Sailor | Ship | Boat | Vessel | Kayak | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11-M |  |  |  |  |  |  |  |  |
| AEFC-YOLO11-M |  |  |  |  |  |  |  |  |

## 16. 实现优先级

如果时间紧，推荐按以下顺序实现：

### 第一阶段：必须完成

1. 跑通 YOLO baseline；
2. 加入 UIAE（BPW + KBL）；
3. 对比增强前后检测性能；
4. 做 BPW、KBL 消融。

### 第二阶段：推荐完成

1. 加入 EAFC；
2. 做 UIAE + EAFC 消融；
3. 输出注意力图可视化。

### 第三阶段：有时间再完成

1. 加入 MDCT；
2. 做多退化一致性实验；
3. 写鲁棒性分析。

如果只做 EI 短文，最稳的组合是：

```text
UIAE + EAFC
```

MDCT 可以作为训练策略简单加入，不一定展开太复杂。

---

## 17. 风险与解决方案

### 17.1 风险一：增强模块没有提升

可能原因：

- 增强过度；
- 参数范围太大；
- BPW/KBL 影响正常图像；
- 训练初期检测器不稳定。

解决方法：

- 缩小参数范围；
- 加强 `L_param`；
- 前 10 个 epoch 冻结 UIAE；
- 或先训练 baseline，再加载权重训练 UIAE。

### 17.2 风险二：双分支计算量太大

解决方法：

- 两个分支共享 backbone；
- 只在 P3、P4、P5 层做校准；
- 使用 YOLO11-S 作为基础模型；
- 只对增强图走完整 backbone，原图走浅层辅助分支。

### 17.3 风险三：BPW 在雾天或水面反光场景中过曝

ERUP-YOLO 的实验分析指出，BPW 可能在雾、沙尘等高亮散射区域造成过曝。WaterScenes 中的水面反光也可能出现类似问题。

解决方法：

- 限制 `α_bpw` 的初始范围；
- 对高亮区域加入轻量曝光惩罚；
- 保留原图分支，并通过 EAFC 降低不可靠增强特征权重；
- 在 `adverse_weather` 和 `adverse_lighting` 上分别统计增强前后的 Recall 和误检率。

### 17.4 风险四：一致性损失导致训练不稳定

解决方法：

- 后半训练阶段再开启；
- 将 `λ1` 从 0.1 降到 0.05；
- 只做特征一致性，不做检测框一致性。

---

## 18. 最终推荐实现版本

最适合当前项目和 EI 会议论文的版本如下：

```text
Dataset: WaterScenes 全量数据训练
Train split: train.txt -> datasets/waterscenes_yolo/images/train
Val split: val.txt -> datasets/waterscenes_yolo/images/val
Test split: test.txt -> datasets/waterscenes_yolo/images/test
Special test 1: adverse_lighting
Special test 2: adverse_weather

Baseline: YOLO11-M
Proposed: AEFC-YOLO11-M

Modules:
1. UIAE: BPW + KBL 统一退化感知图像增强模块
2. EAFC: 增强感知特征校准模块
3. MDCT: 多退化一致性训练策略

Training:
imgsz = 640
batch = 16
epochs = 200
optimizer = AdamW
lr0 = 0.001
weight_decay = 0.0005
cosine scheduler = true
warmup_epochs = 3
AMP = false by default, true after environment verification
```

论文主线可以总结为：

> 本文从图像输入层和深层特征层两个角度提升恶劣视觉条件下的检测鲁棒性。首先，统一退化感知图像增强模块通过 BPW 全局像素映射和 KBL 局部自适应滤波，自适应处理低光、雾、雨、反光和模糊等退化；其次，增强感知特征校准模块在多尺度特征层面融合原始特征与增强特征，抑制 BPW/KBL 可能引入的过曝、伪影和背景噪声放大；最后，多退化一致性训练策略进一步提升模型在多种恶劣环境下的预测稳定性。

工程实现优先级：先完成数据整理脚本和 YOLO11-M baseline，再实现 UIAE 的 BPW 和 KBL，确认两个专项测试集提升后再加入 EAFC 和 MDCT。

## 19. 后续工作建议

下一步建议依次完成：

1. 创建 `AEFC-YOLO11/` 工程骨架；
2. 实现 `tools/prepare_waterscenes_yolo.py`，把当前 `image/` 与 `detection/yolo/` 生成标准 YOLO 目录；
3. 写入 `configs/waterscenes_full.yaml`、`configs/waterscenes_adverse_lighting.yaml`、`configs/waterscenes_adverse_weather.yaml`；
4. 先训练 YOLO11-M baseline，得到全量 check 和两个专项测试集结果；
5. 实现 UIAE，先做 BPW-only、KBL-only、BPW+KBL 三组消融，验证是否提升 `adverse_lighting` 和 `adverse_weather`；
6. 实现 EAFC，做 UIAE + EAFC 消融；
7. 加入 MDCT，做完整 AEFC-YOLO11 消融；
8. 输出增强图、注意力图、检测图，用于论文可视化；
9. 根据实验结果调整论文题目、贡献表述和最终模型命名。

当前数据已经满足开始搭建网络模型结构代码的前置条件。
