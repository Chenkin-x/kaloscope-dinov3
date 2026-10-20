# 详细流程

## 数据

直接使用 ImageFolder 结构，无需生成标签表：

```text
base/
  类别A/
    001.jpg
    002.png
  类别B/
    003.jpg
    004.jpg
```

类别按文件夹名称排序，映射保存在 checkpoint 和 `classes.json`。`data.val` 可设置为另一套同结构的验证集；训练和验证集必须有完全相同的类别文件夹。默认不划分验证集，也不会把训练准确率当成验证准确率。每个类别至少需要一张可读图片；训练总图数必须不小于 `batch_size × GPU 数量`。训练每轮丢弃不足一批的尾部，验证不重复、不丢弃图片

## 图像 Transform

训练和推理的预处理 transform 由 `data` 配置控制。默认使用纯评估管线（Resize + CenterCrop + ToTensor + Normalize），**不开启数据增强**。归一化参数默认为 ImageNet (LVD-1689M) 统计量：

```yaml
data:
  image_size: 224
  # mean 和 std 默认为 ImageNet 统计量，无需显式设置
  # mean: [0.485, 0.456, 0.406]
  # std: [0.229, 0.224, 0.225]
```

### 数据增强

默认不开启任何数据增强。如果需要，通过 `data.augment` 配置逐项开启，每项都可独立控制：

```yaml
data:
  image_size: 224
  augment:
    random_resized_crop: true          # 替换默认 Resize+CenterCrop
    random_resized_crop_scale: [0.5, 1.0]  # 裁剪范围，默认 (0.5, 1.0)
    random_horizontal_flip: true       # 随机水平翻转
    color_jitter:                     # 颜色抖动
      brightness: 0.3
      contrast: 0.3
      saturation: 0.3
      hue: 0.1
      p: 0.8                         # 应用概率
    random_grayscale: true            # 随机灰度化
    random_grayscale_p: 0.2           # 灰度化概率
```

开启 `augment` 中的任意一项后，默认的 Resize + CenterCrop 会被增强操作替换；未开启的项保持关闭。验证集和推理始终使用无增强的评估管线，不受 `augment` 影响

### 仅修改归一化参数

如果只需要更换均值/标准差，设置 `data.mean` 和 `data.std` 即可，其他预处理步骤不变。例如使用 SAT-493M 卫星图像预训练权重时：

```yaml
data:
  image_size: 224
  mean: [0.430, 0.411, 0.296]
  std: [0.213, 0.156, 0.143]
```

### 完全自定义 Transform

如果需要完全替换默认 transform 管线（包括 resize、ToTensor、Normalize 等全部步骤），使用 `data.custom_transform`，值是一个 Python 可调用对象的完整点分路径。该可调用对象接收 `resize_size` 关键字参数，返回一个 `torchvision.transforms.Compose`（或等价对象）。当设置了 `custom_transform` 时，`mean`/`std`/`augment` 参数均被忽略，因为整个管线已由自定义 transform 负责

库内置两个官方 transform 预设：

**LVD-1689M（标准 ImageNet 评估 transform）**：

```yaml
data:
  image_size: 224
  custom_transform: dinov3.finetune.data.lvd_transform
```

等价 Python 代码：

```python
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
```

**SAT-493M（卫星图像 transform）**：

```yaml
data:
  image_size: 224
  custom_transform: dinov3.finetune.data.sat_transform
```

等价 Python 代码：

```python
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    ])
```

### 编写自定义 Transform

在自己的模块中定义一个可调用对象，签名必须为 `(resize_size: int)`，返回值需能接受 PIL Image 并输出 float 张量：

```python
# my_transforms.py
from torchvision.transforms import v2

def my_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
```

在配置中引用：

```yaml
data:
  image_size: 224
  custom_transform: my_transforms.my_transform
```

> 注意：`custom_transform` 设置后将**完全替换**默认 transform，确保你的自定义 transform 包含所有必要的步骤（ToImage / ToDtype / Normalize 等）

## 快速开始

编辑 `configs/finetune.yaml`：

```yaml
model:
  name_or_path: facebook/dinov3-vits16-pretrain-lvd1689m
  pooling: cls
  projection_dim: 128
data:
  train: /path/to/base
  val: null
  image_size: 224
  workers: 4
train_scope: all
classification_weight: 1.0
contrastive_weight: 0.2
contrastive_mode: supervised
memory_bank_size: 4096
precision: bf16
batch_size: 16
epochs: 20
output: runs/dinov3
```

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train.py --config configs/finetune.yaml
```

示例默认通过 Transformers 加载预训练模型。`model.name_or_path` 可以是 HF 模型 ID 或包含 `config.json`、权重文件的本地目录；官方 gated 模型需要获得权限并登录 HF。`bf16` 需要硬件支持，不支持时用 `fp32`；CUDA 也可使用 `fp16`。Windows 数据加载出现问题时可以将 `workers` 设为 `0`。

## 三种训练范围

| 目的 | train_scope | classification_weight | contrastive_weight |
| --- | --- | --- | --- |
| 只训练骨干和对比投影器，不创建分类头 | backbone | 0 | 1 |
| 冻结骨干，只训练分类头（linear probing） | head | 1 | 0 |
| 联合训练骨干、分类头，可同时训练投影器 | all | 1 | 0 或正数 |

`head` 模式还会将骨干保持在 eval 模式，关闭随机深度等行为。`all` 配合 `classification_weight: 0` 和正的对比权重也可做无分类头训练。投影器仅用于对比损失；下游输出的是投影前的骨干特征。

如果要保留已有分类头但冻结它，让分类损失只更新骨干，可设置 `train_scope: backbone`、正的 `classification_weight`，并通过 `init_checkpoint` 加载完整已训练模型；不能对随机初始化且冻结的分类头直接开始训练。

重新训练分类头时，先把训练骨干导出为 Transformers 目录：

```python
from dinov3.finetune.infer import Predictor
model = Predictor(checkpoint="runs/dinov3/last.pt", device="cpu").model
model.backbone.save_pretrained("exported_backbone")
```

然后设置 `model.name_or_path: exported_backbone` 和 `train_scope: head`。
`init_checkpoint: /path/to/last.pt` 加载当前格式的完整模型参数，用于新一轮训练，要求架构、分类头和投影器匹配；不会恢复优化器、epoch 或队列。

## 对比学习与 memory bank

每张图分别进行两次随机增强，得到一个正样本对：

- `contrastive_mode: supervised`：同类别是正样本，包括当前批次和队列中的同类别图像。
- `contrastive_mode: instance`：当前批次中同一图像的另一视图是正样本，其他图像为负样本；队列中同一图像的旧特征会被排除。
- `temperature` 控制相似度温度，必须大于 0。
- `memory_bank_size: 0` 禁用队列；正整数指定最大存储视图数。

队列存储归一化的、停止梯度的历史投影特征、类别和样本 ID，按 FIFO 更新。它不使用动量编码器，历史特征可能随模型更新而陈旧；队列容量需要根据数据和训练速度调整。分布式训练使用可微的跨进程批次特征收集，队列在所有进程同步更新。双视图会使一次前向包含 `2 × batch_size` 张图；队列不额外运行骨干，但相似度矩阵仍消耗显存。

## 训练技巧与调度

通用训练技巧集中在 `dinov3/finetune/optimization.py`，由 `optimization` 配置控制。`configs/finetune.yaml` 已启用 warmup + cosine、分层学习率、免 decay 参数分组和 label smoothing；EMA、输出层暂时冻结默认关闭。

```yaml
optimization:
  scheduler: cosine
  schedule_epochs: null
  warmup_epochs: 2
  warmup_start_lr: 0.000001
  min_lr: 0.000001
  weight_decay_end: 0.05
  layer_decay: 0.75
  exclude_bias_norm_decay: true
  backbone_lr_multiplier: 1.0
  head_lr_multiplier: 1.0
  projector_lr_multiplier: 1.0
  patch_embed_lr_multiplier: 1.0
  freeze_last_layer_epochs: 0
  label_smoothing: 0.1
  ema_decay: null
  ema_decay_end: null
```

各项行为：

- `scheduler` 为 `constant` 或 `cosine`，都支持线性 warmup。warmup 从 `warmup_start_lr` 开始，其结束后的第一个更新达到基础 `lr`，随后 cosine 衰减到 `min_lr`。这些是应用各参数组倍率之前的学习率。
- `schedule_epochs` 是计划的调度长度，默认采用首次启动时的 `epochs`。warmup 必须短于调度长度。epoch 长度会转换为优化器更新次数，小数步向上取整；FP16 溢出跳过 optimizer step 时，调度、EMA 和队列都不推进。
- `weight_decay_end` 非空时，weight decay 从顶层 `weight_decay` 余弦变化到此值；留空保持固定。
- `layer_decay` 在 `(0, 1]` 内，越浅的骨干层学习率越低。ViT 按 Transformer block 排序；ConvNeXt 按 stage 内各 block 顺序展开，下采样层归到对应阶段入口。取 `1` 关闭分层衰减。
- `backbone_lr_multiplier`、`head_lr_multiplier`、`projector_lr_multiplier` 分别控制三个部分的学习率；`patch_embed_lr_multiplier` 额外作用于 ViT patch embedding 或 ConvNeXt stem。
- `exclude_bias_norm_decay: true` 将 bias、归一化、LayerScale/一维参数、可学习 tokens 和位置编码从 weight decay 中排除。冻结的参数不会进入优化器。参数组及名称写入 `parameter_groups.json`，可检查分层与倍率。
- `freeze_last_layer_epochs` 在开头若干轮将分类头和对比投影器最后一个线性层的学习率及 weight decay 设为零，保证参数不变；梯度仍向骨干传播，Adam 动量仍累计。它是微调输出层的暂时冻结，不是原始 DINO 蒸馏头的梯度冻结实现。
- `label_smoothing` 在 `[0, 1)` 内，仅作用于训练分类交叉熵；验证交叉熵和准确率不使用平滑。

### 可选 EMA

将 `ema_decay` 设为例如 `0.999` 开启指数滑动平均；可通过 `ema_decay_end: 0.9999` 让衰减系数随训练余弦变化。EMA 对训练模型的参数做平滑，不参与损失计算，也不作为 memory bank 的动量编码器。训练中的验证指标仍使用当前模型。

EMA 在 rank 0 的 CPU 上以 FP32 保存，浮点状态做滑动平均，整数状态直接复制。开启后需要额外 CPU 内存及每步数据传输。ZeRO-2 下直接读取 rank 0 的完整参数更新 EMA。checkpoint 同时保存普通模型、EMA 权重和 EMA 更新次数。

推理时明确选择 EMA：

```bash
python -m dinov3.finetune.infer --checkpoint runs/dinov3/last.pt --ema --input /path/to/images --output outputs/ema
```

Python 对应 `Predictor(checkpoint="runs/dinov3/last.pt", use_ema=True)`；checkpoint 中没有 EMA 时会直接报错。

### 调度与恢复兼容

checkpoint 保存成功更新次数、调度总长度、warmup/冻结长度和 EMA 状态。恢复不会重新开始 warmup，也不会因为修改总 `epochs` 而重新拉伸原有余弦曲线；超出原调度长度后保持终点学习率、weight decay 和 EMA 衰减系数。若计划先跑短任务再继续，应从开始就用 `schedule_epochs` 指定完整计划长度，例如先跑 `epochs: 2`、`schedule_epochs: 20`，以后恢复到 `epochs: 20`

## 模型型号与自定义结构

`model.name_or_path` 接受 Transformers 的 DINOv3 ViT 或 ConvNeXt 模型，例如
`facebook/dinov3-vitb16-pretrain-lvd1689m`。实现和参数命名由 Transformers 5.8.x 管理。

`model.pooling` 支持 `cls`、`mean`（patch 平均）、`cls_mean`（CLS 与 patch 平均拼接）。
ViT 的 patch 特征排除 register tokens；ConvNeXt 的 CLS 对应归一化后的全局池化。
输入图像尺寸应与骨干下采样倍率兼容。既有数据预处理不替换为 HF processor，避免改变训练配方的裁剪和增强。

显式随机初始化使用 `name_or_path: null` 和 HF `backbone_config`。ViT 示例见 `configs/custom_vit.yaml`；ConvNeXt 示例：

```yaml
model:
  name_or_path: null
  backbone_config:
    model_type: dinov3_convnext
    depths: [3, 3, 9, 3]
    hidden_sizes: [96, 192, 384, 768]
```

新 checkpoint 的 `model_config.backbone_config` 保存完整架构，恢复时通过 `AutoModel.from_config` 构造，无需重新下载初始权重。
当前 checkpoint 格式为 3；旧原生 `.pth` 和旧格式断点不直接兼容，不做自动猜测或回退。

## Accelerate / ZeRO-2

普通微调和时间窗口训练均使用 Accelerate；默认启动配置为 ZeRO-2：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train.py --config configs/finetune.yaml
```

`--num_processes` 改为实际 GPU 数量。训练 YAML 不再设置 `deepspeed` 字段，
并行配置统一放在 `configs/accelerate_zero2.yaml` 与其引用的 `deepspeed_zero2.json` 中。
`precision` 仍由训练 YAML 控制；`grad_clip` 写入 DeepSpeed 配置，由引擎在更新时执行。
优化器和学习率调度保留项目原有实现，交给 `accelerator.prepare` 包装模型和优化器。

当前梯度累积为 1，`batch_size` 是每卡原始图片数，global batch 为 `batch_size × GPU 数量`。
双视图会使实际前向图像数翻倍。两个入口都已有按 rank 分片的采样器，不再对 DataLoader 调用 `prepare`。
可导特征 gather 保留 autograd，不用无梯度的指标 gather 替换。评估直接使用 ZeRO-2 每卡的完整模型参数。
这里只维护 ZeRO-2 的训练路径，不能仅修改配置中的 stage 就当作 ZeRO-3 使用。

## 保存与恢复

输出目录包含：

```text
config.json       解析后的配置
classes.json      类别名称顺序
metrics.jsonl     每轮损失、准确率、学习率、队列容量和可选验证指标
parameter_groups.json  优化器分组、参数名称和学习率/decay 倍率
last.pt           模型权重、配置、memory bank、调度、EMA 和训练位置；可独立推理
state/            Accelerate 保存的最新模型、优化器分片、scaler 和各 rank RNG
```

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train.py --config configs/finetune.yaml --resume runs/dinov3/last.pt
```

保存和恢复由 `accelerator.save_state / load_state` 负责，每个 rank 都参与。
`last.pt` 和 `state/` 必须来自同一次保存，并放在同一输出目录；每次保存覆盖最新状态。
完整恢复还包括 memory bank、学习率/weight decay 调度、EMA、epoch、global step 和 W&B run ID。
迁移前手动 DDP / DeepSpeed 保存的训练状态不能直接续训，但 Transformers 格式 3 的模型权重仍可用于推理或初始化。

恢复发生在完整 epoch 边界，不支持中途 minibatch 精确续跑。保持数据内容、类别映射、进程数和训练配置一致；可修改 `epochs`（总轮数）、`output` 和 W&B 配置。需要更换学习率、数据或训练范围时应导出骨干并设置 `model.name_or_path`，或用 `init_checkpoint` 开始新任务

## W&B

使用环境中已有的 W&B 并登录（也可以只设置 `WANDB_API_KEY` 环境变量）：

```bash
wandb login
```

在训练 YAML 中设置：

```yaml
wandb:
  project: my-dinov3
  name: vits16-experiment
  mode: online
  upload_checkpoint: true
```

然后运行：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train.py --config configs/finetune.yaml
```

`mode: online` 实时上传；没有网络时用 `mode: offline`，训练完成后在 W&B 输出目录执行 `wandb sync <offline-run-dir>`。`mode: disabled` 关闭上传。`upload_checkpoint: true` 仅在训练成功完成后上传 artifact；设置为 `false` 或删除该字段只上传指标

只由 rank 0 上传运行配置、每轮训练与验证指标，恢复时复用 run ID。`upload_checkpoint: true` 会在训练完成后上传 `last.pt` 和 `classes.json` 模型 artifact；不上传原始图片。DeepSpeed artifact 可推理，但恢复训练还需要本地分片目录。离线记录可设 `mode: offline`，后续用 `wandb sync` 上传；完全关闭用 `wandb: null`。账号、网络和 SDK 服务需在目标环境可用

## 分类推理与下游特征

```bash
python -m dinov3.finetune.infer --checkpoint runs/dinov3/last.pt --input /path/to/images --output outputs/features --topk 5 --patch-tokens --normalize
```

输入可以是单张图片或递归目录。没有分类头时只输出特征，不产生随机分类结果。有分类头时同时输出类别 ID、类别名称、softmax 概率和 logits。`--normalize` 仅对导出的全局特征做 L2 归一化，不改变分类输入或 patch tokens

结果按批次分片写入，避免将整个数据集的 patch tokens 堆在内存中：

- `features_000000.pt` 等：`paths`、`features`，可选 `patch_tokens`、`logits`。
- `predictions.jsonl`：有分类头时每图一行 top-k 结果，无分类头时为空文件。
- `manifest.json`：分片列表、数量和类别映射

全局特征形状为 `[N, D]`（`cls_mean` 为 `[N, 2D]`），patch tokens 为 `[N, P, D]`，不包含 CLS / storage tokens。它们对应缩放、中心裁剪后的输入。建议每次使用新的输出目录

直接对预训练骨干推理，使用包含 `model` 和 `data.image_size` 的 YAML：

```bash
python -m dinov3.finetune.infer --config configs/finetune.yaml --input /path/to/images --output outputs/pretrained
```

完整训练 checkpoint 使用 `--checkpoint`，自动还原预处理、骨干、分类头和类别。配置加载仅加载骨干，不自动探测原生权重或分类头。

独立 `nn.Linear.state_dict()` 分类头（`weight` 和可选的 `bias`）可用 `--head-weights /path/to/head.pth` 加载；官方 ViT-7B 分类头需要 `model.pooling: cls_mean`。`--classes classes.json` 可指定按分类头输出顺序排列的名称数组。未知第三方分类头结构不会被自动猜测

Python 调用：

```python
from PIL import Image
from dinov3.finetune.infer import Predictor

predictor = Predictor(checkpoint="runs/dinov3/last.pt")
result = predictor.predict([Image.open("example.jpg")], patches=True)
features = result["features"]
```
