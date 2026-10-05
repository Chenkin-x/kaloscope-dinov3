# 画师时间窗口 sigmoid 训练

本入口将画师与发布时间作为弱监督信号，学习图像表征，不使用 CE、VICReg 或 Gram Anchoring。
主包不依赖研究脚本、实验记录或服务器固定目录。

## 安装与启动

先安装适合运行环境的 PyTorch / torchvision，再执行：

```bash
pip install -e '.[temporal]'
# 修改配置中的数据和权重路径；8 可以换成实际 GPU 数量。
torchrun --standalone --nproc-per-node=8 -m dinov3.finetune.temporal.train --config configs/temporal.yaml
# 同一配置、同一 world size、同一数据，恢复 last.pt：
torchrun --standalone --nproc-per-node=8 -m dinov3.finetune.temporal.train --config configs/temporal.yaml --resume
```

单卡使用 `python -m dinov3.finetune.temporal.train --config ...`。无 CUDA 时支持 CPU，测试建议 `precision: fp32`。
配置路径相对于启动时工作目录；`paths.npy` 内相对图片路径则相对于该文件所在目录。
`output` 首次运行必须不存在；运行过程中创建 `output/STOP` 可在一个训练步骤结束后保存并停止，恢复前删除该文件。
`--max-steps N` 是累计步数上限，可用于有限验证；它不表示已收敛。

## 数据接口

所有 NPY 文件均为非 pickle 数组。先完成去重、画师年份分组和固定 train/dev/test 划分，再提供以下文件。
本入口消费已有分组，不重新划分测试集，也不生成新标签。

| 文件 | 格式与含义 |
| --- | --- |
| `paths.npy` | 长度 N 的定长 UTF-8 bytes 或 Unicode 图片路径 |
| `metadata/artist_ids.npy` | 长度 N 的整数，单画师 ID |
| `metadata/days.npy` | 长度 N 的有符号整数，同一日期基准下的发布天数 |
| `metadata/work_family_id.npy` | 长度 N 的整数，重复/衍生作品家族 ID |
| `metadata/group_ids.npy` | 长度 N 的整数，全局唯一的画师年份组 ID；负值表示不参与训练 |
| `metadata/split.npy` | 长度 N 的整数，0=train、1=dev、2=test |
| `panel/gallery.npy` | 不重复的训练集行索引，至少 10 个 |
| `panel/query.npy` | 不重复的 dev 行索引 |
| `panel/content.json` | `{"行索引": ["空格分隔角色标签", "空格分隔作品标签"]}`，覆盖 gallery/query |

数组必须严格逐行对应。一个作品家族不能跨数据划分；每个年份组只能属于一个画师。
gallery 和 query 分别至少包含 world size 个样本。每个被计入的查询需要至少 10 个允许的候选，且其中至少一个正样本。
跨内容评估要求双方角色与作品标签均已知，且角色、作品均无交集；标签缺失不等于跨内容。
程序校验这些数据边界，并对所用数据清单计算 SHA-256，恢复时拒绝配置或清单变化。
图片文件本身不逐张哈希，运行期间必须保持图片内容不变。

## 目标、采样和预处理

- 正样本：同画师、发布时间相差不超过 `window_days`（默认 365 天）、不同作品家族。
- 负样本：不同画师、不同作品家族。同画师但超出窗口的图像对忽略，自身/同家族对也忽略。
- 原始 sigmoid：`sum(softplus(-sign * (exp(log_temperature) * cosine + bias))) / global_batch`，仅计有效对。
  对称有序对均参与计算；不做 batch 中心化或正负类分别归一化。初始温度 10，bias 为 `-log(global_batch-2)`。
- 每个 global batch 一半为锚点、一半为配对图；锚点中一半遍历全部已分配训练样本，一半先均衡画师再均衡年份组。
  无时间内正样本的锚点仍保留，额外位置尝试选其他画师；没有可用配对数据时明确报错。
- `batch_size` 为每进程图片数，global batch 必须至少 4 且能被 4 整除。梯度累积不等价于增加对比样本池，本入口不提供该替代。
- 图像按 EXIF 转正、透明区域合成白底，短边缩放至 `image_size` 后中心裁剪，无拉伸、无随机裁剪。
- 默认训练 ViT 第 8 层起的 blocks、最终 norm、投影头、温度与 bias；其他骨干层冻结。

年份组用于均衡采样，正负关系仍由锚点相对天数判定；分组边界不会截断相邻年份的有效正样本。
发布时间仅是创作时间的代理信号，不能等同于真实绘制日期。

## 评估、早停与检查点

用固定 train gallery / dev query 分别报告 backbone 与 projector 的时间窗口 Hit@1/10、Precision@10、画师 Top1、年份组 Hit@10；
所有候选和跨内容候选分别统计。这里的画师 Top1 是检索命中率，不是分类头准确率。
主选择指标为 projector 的跨内容、按画师宏平均 temporal Hit@10。它只度量满足候选与正样本条件的查询，输出同时报告查询和画师数量。
测试集不参与早停或检查点选择，独立最终测试应在完成模型选择后另行执行。

一个 coverage pass 表示均匀锚点遍历训练集一次，不是所有 batch 位置只看一次图片。
默认至少遍历一次才因平台期降学习率；每连续 3 次无有效提升减半，到 1/16 后连续 6 次无提升且至少 2 passes 才判定收敛。
连续 3 次显著退化或表征不健康会独立停止，不标记收敛。`max_coverage_passes` 是运行上限，也不表示收敛。

输出包含 `last.pt`、改善时的 `best.pt`、配置、步骤/验证日志及本运行独有的采样顺序缓存。
checkpoint 保存模型、优化器、早停状态、逐 rank RNG 和 epoch 内位置。输入采用确定性中心裁剪，支持相同步数的恢复回归测试。
已有研究入口的 checkpoint 格式不同，**不支持直接用 `--resume` 恢复**；正在运行的旧任务继续使用原入口，不自动迁移。

导出特征时加载本入口的编码器，而不是 ImageFolder 分类预测器：

```python
import torch
from dinov3.finetune.temporal.model import Encoder

checkpoint = torch.load('runs/temporal/best.pt', map_location='cpu', weights_only=False)
model = Encoder(checkpoint['model_config'], initialize=False).eval()
model.load_state_dict(checkpoint['model'])
# images: 经相同缩放/裁剪的 NCHW uint8 RGB。
with torch.no_grad():
    features = model(images)
backbone = features[:, :model.feature_dim]
projector = features[:, model.feature_dim:]
```

只加载自己生成且信任的训练检查点。特征做 L2 归一化后计算 cosine 相似度，与训练评估保持一致。

## 开发验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

测试使用临时小图和随机初始化的小型 ViT，覆盖预处理、采样覆盖、时间边界、同家族排除、dev/test 隔离、
单进程恢复、多进程 Gloo 训练与恢复，以及通用入口的单/双视图 CE 和对比学习。
CPU 测试不能替代目标硬件上的 NCCL/bf16 性能验证，也不证明模型学习效果。

## Kaloscope3 版本化配方

正式 2K 基线与 4K＋跨内容配对＋5% 灰度候选的参数及兼容性说明见
[configs/kaloscope3](configs/kaloscope3/README.md)。v1 提供原生 CLI 配置；v2 是完整策略记录，尚不能直接通过当前原生入口执行，不能仅提高 batch 后视作同一版本。
