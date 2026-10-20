# Kaloscope DINOv3

使用 Transformers 加载 DINOv3，提供 ImageFolder 分类／对比微调，以及画师时间窗口 sigmoid 表征训练。
骨干统一走 `AutoModel.from_pretrained`，不再维护内置 ViT、ConvNeXt 或 Torch Hub 下载代码。
本项目使用 Transformers 5.8.x、Accelerate 1.13.x。两个训练入口统一通过 Accelerate 运行，
默认启动配置为 DeepSpeed ZeRO-2。

Clone 后在仓库根目录直接运行，不需要安装本项目。使用已有的 PyTorch / torchvision、
Transformers、Accelerate 和 DeepSpeed 环境；`requirements.txt` 仅列出依赖。

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train.py --config configs/finetune.yaml
# 时间窗口训练：
accelerate launch --config_file configs/accelerate_zero2.yaml --num_processes 8 \
  train_temporal.py --config configs/temporal.yaml
```

修改数据目录，并通过 `model.name_or_path` 指定 HF 模型 ID 或 Transformers 格式的本地目录：

```yaml
model:
  name_or_path: facebook/dinov3-vitb16-pretrain-lvd1689m
  pooling: cls_mean
  projection_dim: 256
```

官方模型需要先在 [Hugging Face 模型页面](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
取得访问权限，然后 `hf auth login`。加载失败会直接报错，不会改成随机初始化。
显式随机初始化使用 `name_or_path: null` 和 `backbone_config`，见 [custom_vit.yaml](configs/custom_vit.yaml)。

| 骨干 | `model.name_or_path` |
| --- | --- |
| ViT-S/16 | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| ConvNeXt Tiny | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` |

`pooling` 支持 `cls`、`mean` 和 `cls_mean`。ViT 的 patch 特征排除 CLS 和 register tokens；
ConvNeXt 使用归一化后的全局池化和空间特征。图像增强、EXIF 方向、透明白底和归一化仍由现有数据管线处理。

```bash
python infer.py --checkpoint runs/dinov3/last.pt \
  --input base --output outputs/features --patch-tokens
```

`batch_size` 是每卡原始图像数，梯度累积固定为 1；训练精度读取训练 YAML 中的 `precision`。
已有采样器按 rank 分片，因此只对模型和优化器调用 `accelerator.prepare`，避免二次切分配对样本。
保存和恢复使用 [Accelerate 标准接口](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)：
`last.pt` 可独立推理，续训需要同时保留相邻的 `state/`（模型、优化器分片和各 rank RNG）。

新 checkpoint 保存完整 HF 模型配置，可在没有初始预训练文件时恢复。旧原生 `.pth` 骨干和旧训练
checkpoint 的参数命名不同，不支持直接加载或断点续训；本次不添加自动转换分支。
需要单独导出微调骨干时，调用 `model.backbone.save_pretrained("exported_backbone")`，
后续将该目录填入 `model.name_or_path`。

完整配置、Accelerate ZeRO-2、W&B 和推理见 [微调说明](FINETUNE_ZH.md)。
画师时间窗口训练见 [训练说明](TEMPORAL_TRAINING_ZH.md) 和 [配置](configs/temporal.yaml)。
脚本不自动安装依赖，也不修改现有环境。
