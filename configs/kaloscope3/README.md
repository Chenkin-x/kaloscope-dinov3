# Kaloscope3 版本化训练配置

两版配置从正式运行的 `config.json` 转换，保留实际 batch、学习率、曝光量、采样、增强和评估协议；机器路径替换为可编辑占位符。这里不包含图像、权重、实验日志或私有路径。

| 配方 | v1-sigmoid-2k | v2-sigmoid-4k-cross-gray |
|---|---|---|
| 状态（2026-10-05） | 已完成的正式基线 | 正在训练的候选版本 |
| 全局 batch / GPU 数 | 2048 / 8 | 4096 / 8 |
| 每卡图像 / 计算 microbatch | 256 / 256 | 512 / 256，精确梯度缓存 |
| 跨内容 mate 替换尝试比例 | 0 | 50%，无候选保留原配对 |
| 灰度概率 | 0 | 5% |
| warmup / 评估 / 保存步数 | 1000 / 1000 / 500 | 500 / 500 / 250 |
| 每轮锚点覆盖步数 | 12831 | 6415.5，拼接原始基线流 |
| 输入 / 投影维度 | 512 / 256 | 512 / 256 |
| 主分支原生 CLI | 支持 | 尚不支持全部策略扩展 |

共同设置：DINOv3 ViT-B/16，CLS+mean patch，最后四个 block 和最终 norm 可训练，AdamW，bf16，sigmoid loss；同画师 ±365 天且不同作品族为正样本，不同画师且不同作品族为负样本。全部 6,569,159 张已分配训练图像，短边缩放后中心裁剪，不拉伸。测试集保持封存。不使用 CE、VICReg、Gram Anchoring 或 EMA。

## 文件格式和运行兼容性

每个目录的 `recipe.yaml` 使用统一的 **配方记录格式 v1**，不是 temporal CLI 的扁平配置。`trainer_config` 表示原生入口可表达的参数，`runtime`、`sampling`、`augmentation` 等记录其余复现要求，`compatibility` 明确缺失功能。

**不要直接把 `recipe.yaml` 传给 `--config`，也不要只抽取 v2 的 `trainer_config` 运行并称之为 v2。** 只提高 batch 不会实现跨内容配对、灰度增强或真实 4K 梯度缓存。

v1 另提供了主分支支持的扁平 `temporal.yaml`。编辑 paths、metadata、panel、model.weights、output 后：

```bash
ulimit -n 65536
torchrun --standalone --nproc_per_node=8 \
  -m dinov3.finetune.temporal.train \
  --config configs/kaloscope3/v1-sigmoid-2k/temporal.yaml
```

v2 的完整训练当前由专用运行器执行；所需扩展已全部列入 `compatibility.required_extensions`。本次仅发布完整配置记录，不把实验运行器并入主分支。旧研究版检查点不能直接作为主分支 `format_version=1` 训练断点恢复；模型权重加载与完整优化器续训是不同接口。

## 复现和对照约束

- 两个连续的 v1 batch 合成一个 v2 batch，原始锚点顺序保留。基线每轮 12831 为奇数，合并必须跨轮连续，不能丢弃最后半个 batch。
- v2 在每个 4096 batch 的 2048 对中，选 1024 对尝试替换 mate。候选须同画师、±365 天、不同作品族，角色和作品标签均非空且分别不相交。仅用已有标签，候选全枚举后均匀抽取。
- 梯度缓存保留当前 4096 张图像间全部有效配对，不是历史特征队列或普通梯度累积；缓存与重放必须使用一致灰度决策及随机状态。
- v2 的一个覆盖周期不是整数步。不能简单按 6416 步重新生成采样序列，否则会偏离原始锚点流。
- 按相同图像曝光量比较：v2 第 S 步对应 v1 第 2S 步。loss 的绝对值受池大小影响，不能直接判断优劣。学习率后期由各自验证结果独立调整。
- v1 只每 1000 步验证，没有第 12500 步准确率记录。不能把相邻值或插值称为该步实测。
- v2 2026-10-05 的资源修复关闭块间无实际复用的 persistent workers，并统一进程 FD 软上限；不改变模型、loss、batch、采样或增强。配置中保留这个运行差异。
- 配方版本号用于区分训练策略，不代表 v2 已证明优于 v1，也不是软件包或模型发布版本号。
