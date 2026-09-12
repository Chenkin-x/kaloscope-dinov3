# DINOv3 kaloscope

先安装与 CUDA 匹配的 PyTorch / torchvision，再安装本项目：

```bash
pip install -e ".[finetune]"
python -m dinov3.finetune.train --config configs/finetune.yaml
```

运行前修改 `configs/finetune.yaml` 中的数据目录和预训练权重路径。默认 `weights: null` 是随机初始化。

```bash
python -m dinov3.finetune.infer --checkpoint runs/dinov3/last.pt --input base --output outputs/features --patch-tokens
```

完整配置、恢复训练、DeepSpeed、W&B 和推理用法见 [中文使用说明](FINETUNE_ZH.md)。

## 预训练模型与下载

貌似meta官方所有的dinov3下载链接都死掉了，只能找modelscope上别人上传的仓库了，比如：
- https://www.modelscope.cn/models/ciqiangxu/DINOv3/files

支持的预设骨干名称（填写到 `model.name`）：

| 型号 | 配置名称 |
| --- | --- |
| ViT-S/16 | `dinov3_vits16` |
| ViT-S+/16 | `dinov3_vits16plus` |
| ViT-B/16 | `dinov3_vitb16` |
| ViT-L/16 | `dinov3_vitl16` |
| ViT-L+/16 | `dinov3_vitl16plus` |
| ViT-H+/16 | `dinov3_vith16plus` |
| ViT-7B/16 | `dinov3_vit7b16` |
| ConvNeXt Tiny | `dinov3_convnext_tiny` |
| ConvNeXt Small | `dinov3_convnext_small` |
| ConvNeXt Base | `dinov3_convnext_base` |
| ConvNeXt Large | `dinov3_convnext_large` |

下载与型号匹配的官方原生 PyTorch 骨干 `.pth` 权重后，在配置中填写本地路径：

```yaml
model:
  name: dinov3_vits16
  weights: /path/to/dinov3_vits16_pretrain.pth
  pooling: cls
  projection_dim: 128
```