"""
inference.py -- everything the demo needs to turn an uploaded image into a prediction.
No Streamlit code in here, so you can also import it from a notebook to test.
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from model_def import ProposedAIDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filenames written by Notebook 2 (Model A) and Notebook 3 (Model B).
CKPT_NAMES = {
    "Model A (MobileNetV3-Small + LFAB)": ("mobilenet", "mobilenet_lfab_model.pth"),
    "Model B (EfficientNet-B0 + LFAB)": ("efficientnet", "efficientnet_lfab_model.pth"),
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Same as Notebook 4's eval_transform (images are resized to 256x256 first, see prepare_image).
EVAL_TRANSFORM = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_models(weights_dir):
    """Load whichever checkpoints exist in weights_dir. Returns ({label: model}, [missing labels])."""
    models, missing = {}, []
    for label, (backbone, fname) in CKPT_NAMES.items():
        path = os.path.join(weights_dir, fname)
        if not os.path.isfile(path):
            missing.append(f"{label}  ->  {path}")
            continue
        m = ProposedAIDetector(embedding_dim=128, pretrained=False, backbone_type=backbone)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        models[label] = m.to(DEVICE).eval()
    return models, missing


def prepare_image(file_or_path):
    """Mimic Notebook 1's preprocessing so the demo sees what the models saw in training:
    RGB -> 256x256 bilinear resize -> (JPEG inputs only) re-save at PIL's default quality ->
    then the 224 center crop + ImageNet normalisation done by EVAL_TRANSFORM.

    Returns (pil_256, tensor_1x3x224x224)."""
    img = Image.open(file_or_path)
    was_jpeg = (img.format or "").upper() in ("JPEG", "MPO")
    img = img.convert("RGB").resize((256, 256), Image.BILINEAR)
    if was_jpeg:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")  # PIL default quality, same as Notebook 1's .save(dest)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img, EVAL_TRANSFORM(img).unsqueeze(0)


@torch.no_grad()
def predict_prob(model, tensor):
    """P(AI-generated) for one preprocessed image (sigmoid of the model's logit)."""
    logits, _ = model(tensor.to(DEVICE))
    return float(torch.sigmoid(logits).item())


class _LogitOnly(nn.Module):
    """Grad-CAM needs a model that returns a single tensor, not (logit, embedding)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


def gradcam_overlay(model, tensor):
    """Grad-CAM heatmap for the 'AI-generated' output, blended onto the 224x224 crop (uint8 RGB).
    Same target layer as Notebook 4: the last backbone block."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    x = tensor.clone().to(DEVICE).requires_grad_(True)
    with torch.enable_grad():
        cam = GradCAM(model=_LogitOnly(model).eval(), target_layers=[model.backbone[-1]])
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
    rgb = tensor[0].permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    return show_cam_on_image(rgb, heat, use_rgb=True, image_weight=0.5)
