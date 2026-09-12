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

## 快速开始

编辑 `configs/finetune.yaml`：

```yaml
model:
  name: dinov3_vits16
  weights: /path/to/dinov3_vits16_pretrain.pth
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
python -m dinov3.finetune.train --config configs/finetune.yaml
```

示例文件默认 `weights: null`，即随机初始化；使用预训练必须填写权重路径、HTTPS URL 或仓库支持的权重枚举。预训练文件需要自行通过官方授权渠道取得。`bf16` 需要硬件支持，不支持时用 `fp32`；CUDA 也可使用 `fp16`。Windows 数据加载出现问题时可以将 `workers` 设为 `0`。

## 三种训练范围

| 目的 | train_scope | classification_weight | contrastive_weight |
| --- | --- | --- | --- |
| 只训练骨干和对比投影器，不创建分类头 | backbone | 0 | 1 |
| 冻结骨干，只训练分类头（linear probing） | head | 1 | 0 |
| 联合训练骨干、分类头，可同时训练投影器 | all | 1 | 0 或正数 |

`head` 模式还会将骨干保持在 eval 模式，关闭随机深度等行为。`all` 配合 `classification_weight: 0` 和正的对比权重也可做无分类头训练。投影器仅用于对比损失；下游输出的是投影前的骨干特征。

如果要保留已有分类头但冻结它，让分类损失只更新骨干，可设置 `train_scope: backbone`、正的 `classification_weight`，并通过 `init_checkpoint` 加载完整已训练模型；不能对随机初始化且冻结的分类头直接开始训练。

加载一个已训练模型的骨干，重新训练分类头：将它的 `last.pt` 填入 `model.weights`，设置 `train_scope: head`。新的分类头根据当前类别数随机初始化。`model.weights` 也支持官方纯骨干 state_dict，以及含 `teacher` / `student` / `state_dict` / `model` 和 `backbone.` 前缀的普通 PyTorch 权重封装；严格匹配模型参数，尺寸不匹配会直接报错，不静默忽略。

`init_checkpoint: /path/to/last.pt` 加载整个模型参数用于新一轮训练，要求架构、分类头和投影器结构匹配；不会恢复优化器、epoch 或队列。若类别含义不同，即使类别数量相同，也应使用 `model.weights` 重新初始化分类头。

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

EMA 在 rank 0 的 CPU 上以 FP32 保存，浮点状态做滑动平均，整数状态直接复制。开启后需要额外 CPU 内存及每步数据传输。ZeRO-3 每次 EMA 更新都需要所有进程参与完整参数收集，成本较高；大模型建议先关闭 EMA。checkpoint 同时保存普通模型、EMA 权重和 EMA 更新次数。

推理时明确选择 EMA：

```bash
python -m dinov3.finetune.infer --checkpoint runs/dinov3/last.pt --ema --input /path/to/images --output outputs/ema
```

Python 对应 `Predictor(checkpoint="runs/dinov3/last.pt", use_ema=True)`；checkpoint 中没有 EMA 时会直接报错。

### 调度与恢复兼容

checkpoint 保存成功更新次数、调度总长度、warmup/冻结长度和 EMA 状态。恢复不会重新开始 warmup，也不会因为修改总 `epochs` 而重新拉伸原有余弦曲线；超出原调度长度后保持终点学习率、weight decay 和 EMA 衰减系数。若计划先跑短任务再继续，应从开始就用 `schedule_epochs` 指定完整计划长度，例如先跑 `epochs: 2`、`schedule_epochs: 20`，以后恢复到 `epochs: 20`

## 模型型号与自定义结构

支持当前仓库提供的所有骨干：

```text
dinov3_vits16       dinov3_vits16plus
dinov3_vitb16       dinov3_vitl16       dinov3_vitl16plus
dinov3_vith16plus   dinov3_vit7b16
dinov3_convnext_tiny   dinov3_convnext_small
dinov3_convnext_base   dinov3_convnext_large
```

`model.weights` 可用 `LVD1689M` 或 `SAT493M`，但只有官方实际发布且当前构造器支持的型号/数据集组合能下载成功。大模型的内存与显存需求不会因为入口统一而降低

`model.pooling` 支持 `cls`、`mean`（patch 平均）、`cls_mean`（CLS 与 patch 平均拼接）。输入图像尺寸应与 ViT 的 patch size 兼容

需要修改宽度、深度、头数、patch size 等架构参数，使用 `custom_vit` 或 `custom_convnext`，把构造参数写入 `model.kwargs`。参见 `configs/custom_vit.yaml`；ViT 参数对应 `DinoVisionTransformer`，ConvNeXt 参数对应 `ConvNeXt`，例如：

```yaml
model:
  name: custom_convnext
  kwargs:
    depths: [3, 3, 9, 3]
    dims: [96, 192, 384, 768]
```

预设型号的固定架构参数不要通过 `kwargs` 重复传入。自定义结构加载官方权重时，必须匹配原架构，包括 storage tokens、FFN 和归一化配置

## DDP / DeepSpeed

```bash
torchrun --standalone --nproc_per_node=2 -m dinov3.finetune.train --config configs/finetune.yaml
```

使用 DeepSpeed：

```bash
pip install -e ".[deepspeed]"
```

在训练 YAML 设置 `deepspeed: configs/deepspeed_zero2.json`，可用官方 DeepSpeed launcher：

```bash
deepspeed --num_gpus=2 -m dinov3.finetune.train --config configs/finetune.yaml
```

也支持官方常用的 torchrun launcher：

```bash
torchrun --standalone --nproc_per_node=2 -m dinov3.finetune.train --config configs/finetune.yaml
```

两种方式都会由 DeepSpeed 初始化 NCCL process group；不要在外层再包一层 DDP。单卡 DeepSpeed 也可用 `deepspeed --num_gpus=1` 或 `torchrun --standalone --nproc_per_node=1`，确保分布式环境变量完整

提供 ZeRO-2 示例；也可将 `zero_optimization.stage` 改为 `3`。ZeRO-3 导出时所有进程参与参数合并，rank 0 写出可直接推理的 `last.pt`，需要足够 CPU 内存。优化器由训练入口统一创建；DeepSpeed 配置不接受另设 optimizer / scheduler。当前梯度累积固定为 1，DeepSpeed 的 batch 参数和混合精度必须与训练 YAML 一致，冲突会报错。`batch_size` 指每进程原始图片数，`train_batch_size = batch_size × GPU 数量 × gradient_accumulation_steps`，入口会强制校验这些值

## 保存与恢复

输出目录包含：

```text
config.json       解析后的配置
classes.json      类别名称顺序
metrics.jsonl     每轮损失、准确率、学习率、队列容量和可选验证指标
parameter_groups.json  优化器分组、参数名称和学习率/decay 倍率
last.pt           最新完整 checkpoint，也可直接用于推理
deepspeed/        仅 DeepSpeed：按 epoch 保存的分片训练状态
```

```bash
python -m dinov3.finetune.train --config configs/finetune.yaml --resume runs/dinov3/last.pt
```

DeepSpeed 恢复时用 `torchrun` 执行上述入口。模型、优化器、FP16 scaler、memory bank、学习率/weight decay 调度、可选 EMA、epoch、global step、各进程随机数状态以及 W&B run ID 会恢复。DeepSpeed 必须保留 `last.pt` 同目录下的 `deepspeed/` 及对应 epoch 分片；单独的 `last.pt` 仅足够推理

恢复发生在完整 epoch 边界，不支持中途 minibatch 精确续跑。保持数据内容、类别映射、进程数和训练配置一致；可修改 `epochs`（总轮数）、`output` 和 W&B 配置。需要更换学习率、数据或训练范围时应使用 `model.weights` 或 `init_checkpoint` 开始新任务

## W&B

安装 W&B 额外依赖并登录（也可以只设置 `WANDB_API_KEY` 环境变量）：

```bash
pip install -e ".[finetune]"
wandb login
```

普通 PyTorch 单卡或 `torchrun` 多卡训练都使用同一个 YAML 配置：

```yaml
wandb:
  project: my-dinov3
  name: vits16-experiment
  mode: online
  upload_checkpoint: true
```

然后运行：

```bash
python -m dinov3.finetune.train --config configs/finetune.yaml
# 多卡普通 PyTorch
torchrun --standalone --nproc_per_node=2 -m dinov3.finetune.train --config configs/finetune.yaml
# 多卡 DeepSpeed（配置中同时设置 deepspeed）
deepspeed --num_gpus=2 -m dinov3.finetune.train --config configs/finetune.yaml
```

`mode: online` 实时上传；没有网络时用 `mode: offline`，训练完成后在 W&B 输出目录执行 `wandb sync <offline-run-dir>`。`mode: disabled` 可用于本地冒烟测试。`upload_checkpoint: true` 仅在训练成功完成后上传 artifact；设置为 `false` 或删除该字段只上传指标

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

如果 `model.weights` 的权重中含 `backbone.*` 和 `head.*` / `linear_head.*`，推理会检测并加载分类头；若没有类别元数据则返回数字类别名称。完整训练 checkpoint 最好使用 `--checkpoint`，自动还原预处理和结构

官方独立分类头可用 `--head-weights /path/to/head.pth` 额外加载；官方 ViT-7B 分类头需要 `model.pooling: cls_mean`。`--classes classes.json` 可指定按分类头输出顺序排列的名称数组。未知第三方分类头结构不会被自动猜测

Python 调用：

```python
from PIL import Image
from dinov3.finetune.infer import Predictor

predictor = Predictor(checkpoint="runs/dinov3/last.pt")
result = predictor.predict([Image.open("example.jpg")], patches=True)
features = result["features"]
```
