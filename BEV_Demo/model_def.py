"""
model_def.py -- architecture copied verbatim from Notebook 4 (V13), so the state_dicts saved by
Notebooks 2 and 3 load without any key mismatch. Only change: make_radial_band_masks() defaults
to device='cpu' instead of 'cuda' (LFAB always passes the device explicitly anyway).

If you ever change ProposedAIDetector / LFAB in the notebooks, re-copy them here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def make_radial_band_masks(H, Wf, num_bands=6, device='cpu'):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(Wf), indexing='ij')
    yy = torch.minimum(yy, H - yy)
    radius = torch.sqrt(yy.float()**2 + xx.float()**2)
    max_r = radius.max()
    band_edges = torch.linspace(0, max_r, num_bands + 1)
    masks = []
    for i in range(num_bands):
        m = (radius >= band_edges[i]) & (radius <= band_edges[i + 1])
        masks.append(m.to(device))
    return masks


class LFAB(nn.Module):
    """Learnable Frequency Attention Block (v2).

    Changes vs V11, each driven by the V11 diagnostics (Notebook 4):
      * Sits on a HIGHER-RESOLUTION intermediate feature map (see
        ProposedAIDetector.lfab_after_block) instead of the final 7x7 map. On 7x7 the rFFT grid is
        only 7x4, so band 0 held a single DC bin and bands 1-5 lumped everything else together.
      * The per-image band_gate offset is BOUNDED (GATE_RANGE * tanh). In V11 it saturated at
        offsets of about +/-8 and gave the same result for every image.
      * No final ReLU. At an intermediate stage the features come from a linear bottleneck and can
        be negative, so the block is a plain residual (exactly the identity when alpha = 0).
      * alpha initialises at 0.1 (V11: 0.4) so the pretrained features are only lightly perturbed at
        the start of training.
    """
    ALPHA_INIT = 0.1
    GATE_RANGE = 1.0   # max magnitude of the per-image gate offset, in logit space

    def __init__(self, channels, num_bands=6):
        super(LFAB, self).__init__()
        self.channels = channels
        self.num_bands = num_bands
        self.last_band_weights = None  # most recent forward pass's per-band gains, for XAI diagnostics
        self._mask_cache = {}

        # Global learned prior per radial frequency band (init 0 -> sigmoid(0) = 0.5 neutral gain).
        # Kept as a plain Parameter because the SHAP-over-bands diagnostic and Finding 2's fix
        # both read/reset it directly by name.
        self.band_weights = nn.Parameter(torch.zeros(num_bands))

        # Learnable gate on how much the frequency branch contributes to the residual sum.
        self.alpha = nn.Parameter(torch.tensor(self.ALPHA_INIT))

        # Content-conditioned band gate: a small per-image offset on top of band_weights, predicted
        # from that image's own pooled features and bounded by GATE_RANGE (see gate_offset).
        # Final layer is zero-init'd so training starts at the global-only behaviour (offset = 0).
        gate_hidden = max(channels // 16, 8)
        self.band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, num_bands),
        )
        nn.init.zeros_(self.band_gate[-1].weight)
        nn.init.zeros_(self.band_gate[-1].bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def gate_offset(self, x):
        """Bounded per-image, per-band offset added to band_weights before the sigmoid."""
        return self.GATE_RANGE * torch.tanh(self.band_gate(x))

    def _get_band_masks(self, H, Wf, device):
        key = (H, Wf, str(device))  # include device so a CPU-cached mask never gets reused after .to('cuda')
        if key not in self._mask_cache:
            self._mask_cache[key] = make_radial_band_masks(H, Wf, self.num_bands, device)
        return self._mask_cache[key]

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape
        x_fp32 = x.to(torch.float32)

        fft_feat = torch.fft.rfft2(x_fp32, norm="ortho")
        Wf = fft_feat.shape[-1]

        band_masks = self._get_band_masks(H, Wf, x.device)      # list of (H, Wf) bool masks

        # Per-sample gain = global prior + this image's bounded predicted offset.
        per_sample_delta = self.gate_offset(x)                                     # (B, num_bands), in [-GATE_RANGE, GATE_RANGE]
        gains = torch.sigmoid(self.band_weights.unsqueeze(0) + per_sample_delta)  # (B, num_bands) in (0, 1)
        self.last_band_weights = gains.detach().mean(dim=0)  # batch-average gain per band, for diagnostics

        gain_map = torch.zeros(B, H, Wf, device=x.device, dtype=torch.float32)
        for i, mask in enumerate(band_masks):
            gain_map = gain_map + gains[:, i].view(B, 1, 1) * mask.to(torch.float32)
        gain_map = gain_map.unsqueeze(1)  # (B, 1, H, Wf) -- broadcasts over the channel dim

        fft_weighted = fft_feat * gain_map.to(torch.complex64)
        spatial_freq = torch.fft.irfft2(fft_weighted, s=(H, W), norm="ortho")

        # Alpha-gated residual fusion (no ReLU -- see class docstring).
        freq_out = self.fusion(spatial_freq.to(x.dtype))
        return residual + self.alpha * freq_out


class ProposedAIDetector(nn.Module):
    # features[1] of MobileNetV3-Small and features[2] of EfficientNet-B0 both output a 56x56 map.
    LFAB_BLOCK_DEFAULTS = {'mobilenet': 1, 'efficientnet': 2}

    def __init__(self, embedding_dim=128, pretrained=False, backbone_type='mobilenet', dropout_p=0.2, use_lfab=True,
                 lfab_after_block=None):
        super(ProposedAIDetector, self).__init__()
        self.backbone_type = backbone_type
        self.use_lfab = use_lfab
        # LFAB v3: the block is applied after backbone.features[lfab_after_block], on a 56x56 map at
        # 224px input (V11 used the final 7x7 map, V12 a 28x28 map). None -> per-backbone default below.

        if backbone_type == 'mobilenet':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights).features
            self.in_channels = 576
        elif backbone_type == 'efficientnet':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights).features
            self.in_channels = 1280
        else:
            raise ValueError("Unsupported backbone. Use 'mobilenet' or 'efficientnet'.")

        self.lfab_after_block = self.LFAB_BLOCK_DEFAULTS[backbone_type] if lfab_after_block is None else lfab_after_block

        if pretrained and weights is not None:
            print(f"[ProposedAIDetector] Loaded ImageNet-pretrained weights for '{backbone_type}' backbone: {weights}")
        else:
            print(f"[ProposedAIDetector] '{backbone_type}' backbone initialized with RANDOM weights (pretrained=False).")

        # Probe the feature-map shape at the LFAB insertion point (eval mode + no_grad so BatchNorm
        # running statistics are not touched), so LFAB is always built with the right channel count.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            probe = self.backbone[:self.lfab_after_block + 1](torch.zeros(1, 3, 224, 224))
        self.backbone.train(was_training)
        self.lfab_feature_shape = tuple(probe.shape[1:])   # (C, H, W)

        self.lfab = LFAB(channels=self.lfab_feature_shape[0])
        self.pool = nn.AdaptiveAvgPool2d(1)

        if backbone_type == 'mobilenet':
            self.embedding_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(256, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.embedding_head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_p),
                nn.Linear(self.in_channels, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True)
            )

        self.final_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Run the backbone block by block so LFAB can sit in the middle of it.
        for i, block in enumerate(self.backbone):
            x = block(x)
            if self.use_lfab and i == self.lfab_after_block:
                x = self.lfab(x)

        pooled = self.pool(x)
        raw_embed = self.embedding_head(pooled)
        norm_embed = F.normalize(raw_embed, p=2, dim=1)
        output = self.final_head(raw_embed)   # classify on the UN-normalized embedding — norm_embed
                                                # stays reserved for the contrastive loss only, so the
                                                # classifier is no longer capped to what fits on a unit sphere

        return output, norm_embed
