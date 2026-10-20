# Kaloscope DINOv3

**为画师风格检索，训练适合自己作品库的图像特征。**

Kaloscope 利用画师身份与作品发布时间微调 DINOv3，让同一画师在相近时期的不同作品在特征空间中更接近。训练得到的图像编码器可以将作品转换为特征向量，用于搭建相似画风检索、作品库浏览等应用。

它提供从训练、评估到模型保存的完整流程：

- **用自己的作品库训练**：通过画师与时间信息构造图像配对，时间窗口默认 365 天，可按数据调整。
- **检查检索是否有效**：在固定图库上评估同画师、相近时期作品的检索表现，并单独统计角色与作品标签均不同的样本。
- **保存模型、接着训练**：根据验证结果保存最佳模型，自动调整学习率，支持早停与断点续训。

开始使用只需要准备数据、填写配置，再启动训练：

1. 按[数据说明](TEMPORAL_TRAINING_ZH.md#数据接口)准备图片清单、画师与时间等元数据，以及固定的评估集。
2. 编辑 [configs/temporal.yaml](configs/temporal.yaml)，填写 `paths`、`metadata`、`panel` 和 `output`，设置模型路径与每卡 `batch_size`。
3. 在已有[依赖环境](requirements.txt)中，从仓库根目录运行。Clone 后即可使用，无需安装本项目：

   ```bash
   accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
     train_temporal.py --config configs/temporal.yaml
   ```

将 `--num_processes` 改为实际 GPU 数量。模型可以使用 Hugging Face ID 或本地 Transformers 模型目录；使用官方权重需先获得访问权限。

训练结果保存在 `output` 指定的目录中：`best.pt` 用于提取图像特征，日志用于查看训练与检索效果。保留 `last.pt` 和 `state/`，在相同配置与 GPU 数量下给启动命令加上 `--resume`，即可继续训练。

技术上，项目采用 **DINOv3 + 投影头 + sigmoid 配对损失**，通过 Transformers 加载模型，使用 Accelerate 与 DeepSpeed ZeRO-2 进行多卡训练。默认骨干为 ViT-B/16。

数据格式、配对规则、评估指标与特征提取示例见[完整使用说明](TEMPORAL_TRAINING_ZH.md)。
