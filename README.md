# Kaloscope DINOv3

画师时间窗口 sigmoid 表征训练。唯一训练入口为 `train_temporal.py`，训练配置为
`configs/temporal.yaml`。使用 Transformers 5.8.x 加载 DINOv3 ViT，Accelerate 1.13.x
管理训练，默认采用 DeepSpeed ZeRO-2。

Clone 后在仓库根目录直接运行，使用机器上已有的依赖环境，无需安装本项目：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train_temporal.py --config configs/temporal.yaml
```

先填写配置中的 `paths`、`metadata`、`panel` 和 `output`；`--num_processes` 改为实际 GPU 数量。
`model.name_or_path` 可以是 Hugging Face 模型 ID 或 Transformers 格式的本地目录。
官方模型需要先获得对应模型的访问权限。

默认设置：

- DINOv3 ViT-B/16，CLS 与平均 patch 特征拼接，投影维度 256。
- 只训练最后 4 个 Transformer blocks、最终 norm、投影头、温度和 bias。
- 输入分辨率 512，每卡 batch 256，BF16，AdamW，预热 1000 步。
- 同画师、相距不超过 365 天且不同作品家族为正样本；不同画师且不同作品家族为负样本。
- 每步 gather 所有 rank 的投影特征，计算 sigmoid 配对损失；梯度累积固定为 1。

配对采样器已按 rank 分片，Accelerate 只包装模型和优化器，避免再次切分样本。

```text
train_temporal.py    训练主循环、评估和保存
configs/
  temporal.yaml
  accelerate_zero2.yaml
  deepspeed_zero2.json
dinov3/
  model.py       Transformers 骨干、投影头、温度和 bias
  data.py        读图和元数据
  sampling.py    时间窗口配对采样
  loss.py        sigmoid 配对损失
  evaluate.py    固定 gallery/query 检索评估
  control.py     验证驱动的学习率调整和早停
  config.py      训练配置
  runtime.py     Accelerate 初始化
```

续训使用相同配置和 GPU 数量：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train_temporal.py --config configs/temporal.yaml --resume
```

`last.pt` 保存模型、完整骨干配置和训练位置，`state/` 保存 Accelerate 的优化器分片及各 rank RNG。
续训需要同时保留两者；`best.pt` 用于最佳模型的特征提取。模型恢复不依赖初始预训练目录。

数据准备、采样含义、评估指标和特征提取示例见 [训练说明](TEMPORAL_TRAINING_ZH.md)。
`requirements.txt` 仅列出环境依赖，脚本不自动安装依赖。
