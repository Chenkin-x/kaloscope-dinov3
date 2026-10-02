import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from .data import image_transform, rgb_loader
from .model import FineTuneModel, load_weights, read_config, state_dict_from


class Predictor:
    def __init__(self, checkpoint=None, config=None, device=None, head_weights=None, classes=None, use_ema=False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = classes
        embedded_head = None
        if use_ema and not checkpoint:
            raise ValueError("EMA inference requires a training checkpoint")
        if checkpoint:
            payload = load_weights(checkpoint)
            if "model_config" not in payload:
                raise ValueError("For raw pretrained weights, use a config with model.weights instead of --checkpoint")
            self.model = FineTuneModel(payload["model_config"], payload["num_classes"], initialize=False)
            if use_ema and not payload.get("ema"):
                raise ValueError("Checkpoint does not contain EMA weights")
            self.model.load_state_dict(payload["ema"]["model"] if use_ema else payload["model"], strict=True)
            self.classes = (classes or payload["classes"]) if payload["num_classes"] else None
            data = payload["config"]["data"]
        else:
            if config is None:
                raise ValueError("Provide checkpoint or config")
            data = config.get("data", {})
            model_config = config.get("model", config)
            self.model = FineTuneModel(model_config, 0)
            weights = model_config.get("weights")
            if weights and weights not in ("LVD1689M", "SAT493M"):
                payload = load_weights(weights)
                state = state_dict_from(payload)
                if "head.weight" in state or "linear_head.weight" in state:
                    embedded_head = state
                    self.classes = classes or payload.get("classes")
        if head_weights or embedded_head is not None:
            state = state_dict_from(load_weights(head_weights)) if head_weights else embedded_head
            for prefix in ("head.", "linear_head."):
                if any(key.startswith(prefix) for key in state):
                    state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
                    break
            count, dimension = state["weight"].shape
            if embedded_head is not None and "pooling" not in model_config:
                if dimension == 2 * self.model.backbone.embed_dim:
                    self.model.pooling = "cls_mean"
                    self.model.feature_dim = dimension
            if dimension != self.model.feature_dim:
                raise ValueError(
                    "Head input dimension differs from pooling output; official ViT-7B head needs cls_mean"
                )
            self.model.head = torch.nn.Linear(dimension, count, bias="bias" in state)
            self.model.head.load_state_dict(state, strict=True)
            if head_weights:
                self.classes = classes
        if (
            self.classes is not None
            and self.model.head is not None
            and len(self.classes) != self.model.head.out_features
        ):
            raise ValueError("Class names do not match classifier output dimension")
        self.model.to(self.device).eval()
        self.transform = image_transform(
            data.get("image_size", 224),
            mean=data.get("mean"),
            std=data.get("std"),
            custom_transform=data.get("custom_transform"),
        )

    @torch.inference_mode()
    def predict(self, images, topk=5, patches=False, normalize=False):
        if topk <= 0:
            raise ValueError("topk must be positive")
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        result = self.model(batch, patches=patches)
        output = {
            "features": (functional.normalize(result["features"], dim=1) if normalize else result["features"]).cpu()
        }
        if patches:
            output["patch_tokens"] = result["patch_tokens"].cpu()
        if "logits" in result:
            logits = result["logits"].float()
            scores, indices = logits.softmax(1).topk(min(topk, logits.shape[1]), dim=1)
            output["logits"] = logits.cpu()
            output["predictions"] = [
                [
                    {
                        "class_id": index,
                        "class_name": self.classes[index] if self.classes else str(index),
                        "probability": score,
                    }
                    for index, score in zip(row_indices, row_scores)
                ]
                for row_indices, row_scores in zip(indices.tolist(), scores.tolist())
            ]
        return output


def main():
    parser = argparse.ArgumentParser(description="Export DINOv3 features and optional classification predictions")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--patch-tokens", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--ema", action="store_true", help="Use EMA weights saved in the training checkpoint")
    parser.add_argument("--head-weights")
    parser.add_argument("--classes", help="JSON array of class names in head output order")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    config = read_config(args.config) if args.config else None
    classes = json.loads(Path(args.classes).read_text(encoding="utf-8")) if args.classes else None
    predictor = Predictor(args.checkpoint, config, args.device, args.head_weights, classes, use_ema=args.ema)
    source_path = Path(args.input)
    paths = (
        sorted(
            path
            for path in source_path.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        )
        if source_path.is_dir()
        else [source_path]
    )
    if not paths or not all(path.is_file() for path in paths):
        raise ValueError("Input contains no supported images or does not exist")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            images = [rgb_loader(path) for path in batch_paths]
            result = predictor.predict(images, args.topk, args.patch_tokens, args.normalize)
            shard = f"features_{start // args.batch_size:06d}.pt"
            torch.save(
                {
                    "paths": [str(path) for path in batch_paths],
                    **{key: value for key, value in result.items() if key != "predictions"},
                },
                output / shard,
            )
            manifest.append({"file": shard, "count": len(batch_paths)})
            if "predictions" in result:
                for path, predictions in zip(batch_paths, result["predictions"]):
                    stream.write(json.dumps({"path": str(path), "predictions": predictions}, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {"shards": manifest, "classes": predictor.classes, "normalized": args.normalize},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {len(paths)} images to {output}")


if __name__ == "__main__":
    main()
