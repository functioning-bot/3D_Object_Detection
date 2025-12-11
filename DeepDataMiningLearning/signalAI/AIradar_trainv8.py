"""
Multi-config deep learning radar detection with config-rotating batches
and gradient accumulation.

All shapes below assume:
  - Nc: number of chirps (slow-time / Doppler dimension)
  - Ns: samples per chirp (fast-time / range dimension)
  - D: Doppler bins  (len(velocity_axis))
  - R: Range bins    (len(range_axis))
"""

import os
import math
import numpy as np
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# IMPORT YOUR EXISTING RADAR SIM / CFAR / VISUALIZATION CODE HERE
# ------------------------------------------------------------------
from AIradar_datasetv8 import (
    AIRadarDataset,
    evaluate_dataset_metrics,
    _plot_2d_rdm,
    _plot_3d_rdm,
    RADAR_CONFIGS,
)

from scipy.ndimage import maximum_filter


# ==============================================================
# Losses for segmentation-like detection map
# ==============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Inputs:
      inputs: [B, H, W] (logits)
      targets: [B, H, W] (0 or 1)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks, good for shape alignment and class imbalance.
    Inputs:
      inputs: [B, H, W] (logits)
      targets: [B, H, W] (0 or 1)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)   # [B,H,W]
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combines BCE, Focal, and Dice losses.
    Inputs:
      inputs: [B, H, W] (logits)
      targets: [B, H, W] (0 or 1)
    """
    def __init__(self, w_bce=0.5, w_focal=1.0, w_dice=1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        loss = 0
        if self.w_bce > 0:
            loss += self.w_bce * self.bce(inputs, targets)
        if self.w_focal > 0:
            loss += self.w_focal * self.focal(inputs, targets)
        if self.w_dice > 0:
            loss += self.w_dice * self.dice(inputs, targets)
        return loss


# ==============================================================
# U-Net backbone for 2D RD maps
# ==============================================================

class DoubleConv(nn.Module):
    """
    (conv => BN => ReLU) * 2
    Input:  [B, C_in, H, W]
    Output: [B, C_out, H, W]
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then DoubleConv.
    Input:  [B, C_in, H, W]
    Output: [B, C_out, H/2, W/2]
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then DoubleConv.
    Inputs:
      x1: upsampled feature    [B, C_up, H', W']
      x2: skip connection      [B, C_skip, H, W]
    Output: [B, C_out, H, W]
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, 2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if necessary (due to odd sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution to get desired number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNetRadar2D(nn.Module):
    """
    U-Net for 2D Range-Doppler maps.

    Input:
      x: [B, C_in_total, D, R]
         C_in_total = (3 RD channels) + (meta channels)

    Output:
      logits: [B, 1, D, R] (per-bin detection logits)
    """
    def __init__(self, n_channels=3, n_meta_channels=0, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels + n_meta_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(self.n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        # Initialize bias for sparse positives (~1% prior)
        with torch.no_grad():
            self.outc.conv.bias.data.fill_(-4.6)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ==============================================================
# RadarTimeNetV6: time-domain -> RD map + detection, config-conditioned
# (Option A: RD size matches dataset RD size exactly, driven by axes)
# ==============================================================

class RadarTimeNetV6(nn.Module):
    """
    RadarTimeNetV6

    Input x shapes supported:
      - [B, Nc, Ns, 2]        (single Rx)
      - [B, Rx, Nc, Ns, 2]    (multi Rx)

    meta: [B, meta_dim] radar configuration embedding.

    rd_shape: (D,R) = (len(velocity_axis), len(range_axis)).

    Pipeline:
      1) Demodulate in time domain.
      2) Range FFT:   [Nc,Ns] -> [Nc,Nr] with Nr <= Ns (no zero-padding).
      3) Doppler FFT: [Nc,Nr] -> [D,Nr] with D <= Nc (no zero-padding).
      4) Aggregate RX channels.
      5) Build RD channels: Real, Imag, log-Magnitude.
      6) U-Net over RD map (no resizing).
      7) Output detection logits [B,1,D,R] with same (D,R) as dataset.
    """
    def __init__(
        self,
        meta_dim: int = 6,
        num_rx: int = 1,
    ):
        super().__init__()
        self.num_rx = num_rx
        self.meta_dim = meta_dim

        # Learnable 2x2 demodulation matrix applied on [I,Q]
        self.demod_weights = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))

        # Meta embedding: [B,meta_dim] -> [B,meta_out_channels,D,R]
        self.meta_hidden_dim = 32
        self.meta_out_channels = 16
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, self.meta_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.meta_hidden_dim, self.meta_out_channels),
            nn.ReLU(inplace=True),
        )

        self.unet = UNetRadar2D(
            n_channels=3,
            n_meta_channels=self.meta_out_channels,
            n_classes=1,
            bilinear=True,
        )

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.demod_weights.data = torch.tensor(
                [[1.0, 0.0],
                 [0.0, -1.0]],
                dtype=torch.float32,
            )

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x to shape [B, Rx, Nc, Ns, 2].

        Accepts:
          - [B, Nc, Ns, 2]
          - [B, Rx, Nc, Ns, 2]
        """
        if x.dim() == 4 and x.shape[-1] == 2:
            # [B,Nc,Ns,2] -> [B,1,Nc,Ns,2]
            x = x.unsqueeze(1)
        elif x.dim() == 5 and x.shape[-1] == 2:
            pass  # Already [B,Rx,Nc,Ns,2]
        else:
            raise ValueError(f"Unexpected input shape for RadarTimeNetV6: {x.shape}")
        return x

    def demodulate(self, rx_signal: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable 2x2 demodulation on IQ.

        Input:
          rx_signal: [B,Rx,Nc,Ns,2]
        Output:
          same shape
        """
        flat = rx_signal.reshape(-1, 2)          # [B*Rx*Nc*Ns, 2]
        demod_flat = flat @ self.demod_weights   # [B*Rx*Nc*Ns, 2]
        return demod_flat.view_as(rx_signal)

    def apply_range_fft(self, x: torch.Tensor, out_bins: Optional[int] = None) -> torch.Tensor:
        """
        Range FFT (fast-time).

        Input:
          x: [B,Rx,Nc,Ns,2]

        Output:
          spec: [B,Rx,Nc,Nr,2]
            Nr <= Ns, with no zero-padding.
        """
        B, Rx, Nc, Ns, _ = x.shape
        x_reshaped = x.reshape(B * Rx * Nc, Ns, 2)   # [B*Rx*Nc,Ns,2]
        real, imag = x_reshaped[..., 0], x_reshaped[..., 1]
        complex_input = torch.complex(real, imag)    # [B*Rx*Nc,Ns]

        fft_n = Ns   # *** no zero-padding ***
        complex_output = torch.fft.fft(complex_input, n=fft_n, dim=1)

        if out_bins is None:
            valid_bins = fft_n
        else:
            if out_bins > fft_n:
                raise RuntimeError(
                    f"Requested range bins {out_bins} > Ns={fft_n}; "
                    "dataset RD axis length and raw Ns are inconsistent."
                )
            valid_bins = out_bins

        complex_output = complex_output[:, :valid_bins]      # [B*Rx*Nc,Nr]

        spec = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        return spec.view(B, Rx, Nc, valid_bins, 2)           # [B,Rx,Nc,Nr,2]

    def apply_doppler_fft(self, x: torch.Tensor, out_bins: Optional[int] = None) -> torch.Tensor:
        """
        Doppler FFT (slow-time).

        Input:
          x: [B,Rx,Nc,Nr,2]

        Output:
          spec: [B,Rx,D,Nr,2]
            D <= Nc, with no zero-padding.
        """
        B, Rx, Nc, Nr, _ = x.shape
        x_transposed = x.permute(0, 1, 3, 2, 4)          # [B,Rx,Nr,Nc,2]
        x_reshaped = x_transposed.reshape(B * Rx * Nr, Nc, 2)

        real, imag = x_reshaped[..., 0], x_reshaped[..., 1]
        complex_input = torch.complex(real, imag)        # [B*Rx*Nr,Nc]

        fft_n = Nc   # *** no zero-padding ***
        complex_output = torch.fft.fft(complex_input, n=fft_n, dim=1)
        complex_output = torch.fft.fftshift(complex_output, dim=1)

        if out_bins is None:
            valid_bins = fft_n
        else:
            if out_bins > fft_n:
                raise RuntimeError(
                    f"Requested Doppler bins {out_bins} > Nc={fft_n}; "
                    "dataset RD axis length and raw Nc are inconsistent."
                )
            # Slice from center after fftshift to keep symmetric interval
            center = fft_n // 2
            half = out_bins // 2
            start = max(0, center - half)
            end = start + out_bins
            if end > fft_n:
                end = fft_n
                start = end - out_bins
            complex_output = complex_output[:, start:end]
            valid_bins = out_bins

        spec = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        spec = spec.view(B, Rx, Nr, valid_bins, 2).permute(0, 1, 3, 2, 4)  # [B,Rx,D,Nr,2]
        return spec

    def _meta_to_map(self, meta: torch.Tensor, spatial_size) -> torch.Tensor:
        """
        meta: [B,meta_dim]
        spatial_size: (D,R)

        Output:
          meta_map: [B,meta_out_channels,D,R]
        """
        B = meta.size(0)
        feat = self.meta_mlp(meta)              # [B,Cm]
        feat = feat[:, :, None, None]          # [B,Cm,1,1]
        feat = feat.expand(
            B, self.meta_out_channels, spatial_size[0], spatial_size[1]
        )
        return feat

    def forward(
        self,
        x: torch.Tensor,
        meta: torch.Tensor,
        rd_shape: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x:
          - [B, Nc, Ns, 2] or [B, Rx, Nc, Ns, 2]

        meta:
          - [B, meta_dim]

        rd_shape:
          - (D_target, R_target) = (len(velocity_axis), len(range_axis))

        Returns:
          dict with:
            'rd_map_db':        [B, D, R]     (un-normalized dB RD map)
            'detection_logits': [B, 1, D, R] (logits, same (D,R) as dataset)
        """
        x = self._normalize_input_shape(x)     # -> [B,Rx,Nc,Ns,2]
        B, Rx, Nc, Ns, _ = x.shape

        if rd_shape is not None:
            D_target, R_target = rd_shape
        else:
            D_target, R_target = None, None

        # 1. Demodulation
        x = self.demodulate(x)                 # [B,Rx,Nc,Ns,2]

        # 2. Range FFT -> set Nr to match dataset R if provided
        x = self.apply_range_fft(
            x, out_bins=R_target
        )                                      # [B,Rx,Nc,Nr,2]

        # 3. Doppler FFT -> set D to match dataset D if provided
        x = self.apply_doppler_fft(
            x, out_bins=D_target
        )                                      # [B,Rx,D,Nr,2]

        # 4. Combine RX antennas
        x = x.mean(dim=1)                      # [B,D,R,2]
        D, R = x.shape[1], x.shape[2]

        # 5. Build RD channels
        x_real = x[..., 0]                     # [B,D,R]
        x_imag = x[..., 1]                     # [B,D,R]
        x_mag = torch.norm(x, dim=-1)          # [B,D,R]
        x_log = 20 * torch.log10(x_mag + 1e-6) # [B,D,R]
        x_log_norm = (x_log + 50.0) / 100.0    # rough scaling into ~[0,1]

        x_in = torch.stack([x_real, x_imag, x_log_norm], dim=1)  # [B,3,D,R]

        # 6. Meta feature map (same spatial size as RD)
        if meta.dim() == 1:
            meta = meta.unsqueeze(0)           # [B,meta_dim]
        meta_map = self._meta_to_map(meta, x_in.shape[2:])   # [B,Cm,D,R]

        x_unet_in = torch.cat([x_in, meta_map], dim=1)       # [B,3+Cm,D,R]

        # 7. U-Net (no resizing, output same D,R)
        det_logits = self.unet(x_unet_in)                    # [B,1,D,R]

        return {
            "rd_map_db": x_log,                 # [B,D,R]
            "detection_logits": det_logits,     # [B,1,D,R]
        }


# ==============================================================
# Meta vector (config embedding) from AIRadarDataset
# ==============================================================

def build_meta_vector_from_dataset(
    ds: AIRadarDataset, config_index: int, num_configs: int
) -> np.ndarray:
    """
    Create a normalized radar meta vector for a given dataset/config.

    Fields (example):
      [fc_GHz/100, B_MHz/2000, T_us/1000, fs_MHz/100, R_max/300, config_id_norm]

    Output:
      meta: [meta_dim] float32
    """
    fc_ghz = ds.fc / 1e9
    B_mhz = ds.B / 1e6
    T_us = ds.T * 1e6
    fs_mhz = ds.fs / 1e6
    R_max = ds.R_max

    cfg_norm = config_index / max(1, (num_configs - 1)) if num_configs > 1 else 0.0

    meta = np.array([
        fc_ghz / 100.0,
        B_mhz / 2000.0,
        T_us / 1000.0,
        fs_mhz / 100.0,
        R_max / 300.0,
        cfg_norm,
    ], dtype=np.float32)
    return meta


# ==============================================================
# Convert detection probability map to detection list (for metrics)
# ==============================================================

def logits_to_detections(
    prob_map: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    threshold: float = 0.5,
    nms_kernel_size: int = 3,
) -> List[Dict]:
    """
    Convert probability map [D,R] to list of detection dicts compatible with CFAR metrics.

    prob_map: [D,R], values in [0,1]
    """
    D, R = prob_map.shape
    mask = prob_map >= threshold

    if nms_kernel_size > 1:
        local_max = maximum_filter(prob_map, size=nms_kernel_size)
        mask &= (prob_map == local_max)

    doppler_idxs, range_idxs = np.where(mask)
    dets = []

    for d_idx, r_idx in zip(doppler_idxs, range_idxs):
        if not (0 <= r_idx < len(range_axis) and 0 <= d_idx < len(velocity_axis)):
            continue

        dets.append({
            "range_idx": int(r_idx),
            "doppler_idx": int(d_idx),
            "range_m": float(range_axis[r_idx]),
            "velocity_mps": float(velocity_axis[d_idx]),
            "angle_deg": None,
            "magnitude": float(prob_map[d_idx, r_idx]),
        })

    return dets


# ==============================================================
# Training with config-rotating batches + gradient accumulation
# ==============================================================

def custom_collate_fn(batch):
    """
    Custom collate function to handle dictionary elements that cannot be stacked directly
    (e.g., variable length lists, metadata dicts, numpy arrays).
    """
    elem = batch[0]
    collated = {}
    
    for key in elem:
        if key in ['target_info', 'cfar_detections', 'range_axis', 'velocity_axis']:
            collated[key] = [d[key] for d in batch]  # keep as list
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], np.ndarray):
            collated[key] = torch.stack(
                [torch.from_numpy(d[key]) for d in batch]
            )
        else:
            collated[key] = [d[key] for d in batch]
            
    return collated


def _fix_target_mask_orientation(
    target_mask: torch.Tensor,
    ds: AIRadarDataset,
) -> torch.Tensor:
    """
    Ensure target_mask is [B,D,R,1] where:
      D = len(ds.velocity_axis)
      R = len(ds.range_axis)

    If mask is [B,R,D,1], transpose to [B,D,R,1].
    """
    B, d0, r0, c = target_mask.shape
    D_axis = len(ds.velocity_axis)
    R_axis = len(ds.range_axis)

    if d0 == D_axis and r0 == R_axis:
        # Already [B,D,R,1]
        return target_mask
    elif d0 == R_axis and r0 == D_axis:
        # Probably [B,R,D,1] -> transpose
        return target_mask.permute(0, 2, 1, 3).contiguous()
    else:
        raise RuntimeError(
            f"target_mask shape {target_mask.shape} does not match "
            f"axes lengths D={D_axis}, R={R_axis} (either [B,D,R,1] or [B,R,D,1])."
        )


def train_one_epoch_config_rotation(
    model: nn.Module,
    optimizer,
    criterion,
    device,
    epoch: int,
    config_names: List[str],
    batch_size: int,
    num_batches_per_config: int,
    base_data_dir: str,
    train_datasets: Optional[Dict[str, AIRadarDataset]] = None,
    regenerate_each_epoch: bool = True,
):
    """
    One training epoch with config-rotating batches and gradient accumulation.

    If `train_datasets` is provided and `regenerate_each_epoch=False`,
    those datasets are reused (no regeneration each epoch).
    Otherwise, fresh AIRadarDataset instances are constructed each epoch.
    """
    model.train()
    num_configs = len(config_names)
    accum_steps = num_configs

    # ----------------------------------------------------------
    # 1. Build per-config datasets, loaders, and meta vectors
    # ----------------------------------------------------------
    config_specs = []
    for cfg_idx, cfg_name in enumerate(config_names):

        if (train_datasets is not None) and (not regenerate_each_epoch):
            ds = train_datasets[cfg_name]
            num_samples = len(ds)
            cfg_dir = getattr(
                ds, "save_path",
                os.path.join(base_data_dir, f"{cfg_name}_train_reused")
            )
        else:
            cfg_dir = os.path.join(
                base_data_dir, f"{cfg_name}_train_epoch{epoch}"
            )
            num_samples = batch_size * num_batches_per_config
            ds = AIRadarDataset(
                num_samples=num_samples,
                config_name=cfg_name,
                save_path=cfg_dir,
                drawfig=False,
                apply_realistic_effects=True,
                clutter_intensity=0.3,
            )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )

        iterator = iter(loader)

        meta_vec = build_meta_vector_from_dataset(ds, cfg_idx, num_configs)
        meta_base = torch.from_numpy(meta_vec).float().to(device)  # [meta_dim]

        D_axis = len(ds.velocity_axis)
        R_axis = len(ds.range_axis)

        config_specs.append({
            "name": cfg_name,
            "dataset": ds,
            "loader": loader,
            "iterator": iterator,
            "meta_base": meta_base,
            "D_axis": D_axis,
            "R_axis": R_axis,
        })

        print(
            f"[Epoch {epoch}] Config {cfg_name}: {num_samples} samples, "
            f"{len(loader)} batches (expected ~{num_batches_per_config})"
        )

    optimizer.zero_grad()
    running_loss = 0.0   # track true loss (before /accum_steps)
    n_samples_total = 0

    # ----------------------------------------------------------
    # 2. Rotate across configs with gradient accumulation
    # ----------------------------------------------------------
    for step_idx in range(num_batches_per_config):
        for spec in config_specs:
            try:
                batch = next(spec["iterator"])
            except StopIteration:
                continue

            ds = spec["dataset"]
            D_axis = spec["D_axis"]
            R_axis = spec["R_axis"]

            # time_domain: [B,Nc,Ns,2] or [B,Rx,Nc,Ns,2]
            time_domain = batch["time_domain"].to(device)

            # target_mask: may be [B,D,R,1] or [B,R,D,1]
            target_mask = batch["target_mask"].to(device)
            target_mask = _fix_target_mask_orientation(target_mask, ds)
            target = target_mask.squeeze(-1)   # [B,D,R]

            # meta per sample: [B,meta_dim], all rows = meta_base
            B_cur = time_domain.shape[0]
            meta = spec["meta_base"].unsqueeze(0).repeat(B_cur, 1)

            rd_shape = (D_axis, R_axis)
            out = model(time_domain, meta, rd_shape=rd_shape)

            logits = out["detection_logits"].squeeze(1)  # [B,D,R]

            if logits.shape != target.shape:
                raise RuntimeError(
                    f"Shape mismatch between logits {logits.shape} "
                    f"and target {target.shape} under Option A."
                )

            # Compute true loss (unscaled) for reporting
            raw_loss = criterion(logits, target)         # scalar

            # Scale for gradient accumulation
            loss = raw_loss / float(accum_steps)
            loss.backward()

            running_loss += raw_loss.item() * B_cur
            n_samples_total += B_cur

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / max(1, n_samples_total)
    print(f"[Epoch {epoch}] Train Loss (true): {epoch_loss:.6f}")
    return epoch_loss


# ==============================================================
# Deep model evaluation on a single config dataset
# ==============================================================

@torch.no_grad()
def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: AIRadarDataset,
    device,
    config_index: int,
    num_configs: int,
    name: str,
    out_dir: str,
    threshold: float = 0.5,
    max_vis_samples: int = 3,
):
    """
    Evaluate deep model on one AIRadarDataset and compare with GT.

    Option A + orientation fix:
      - RD shape is (D_axis,R_axis) = (len(velocity_axis), len(range_axis)).
      - We do NOT rely on target_mask for rd_shape; instead we use dataset axes.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    all_tp = all_fp = all_fn = 0
    all_range_errors = []
    all_vel_errors = []

    meta_vec = build_meta_vector_from_dataset(dataset, config_index, num_configs)
    meta_base = torch.from_numpy(meta_vec).float().to(device)  # [meta_dim]

    D_axis = len(dataset.velocity_axis)
    R_axis = len(dataset.range_axis)
    rd_shape = (D_axis, R_axis)

    vis_count = 0
    name_slug = "".join(
        ch for ch in name.lower().replace(" ", "_")
        if ch.isalnum() or ch in ["_"]
    )

    for i in range(len(dataset)):
        sample = dataset[i]

        # time_domain: [Nc,Ns,2] or [Rx,Nc,Ns,2] -> [1,...]
        time_domain = sample["time_domain"].unsqueeze(0).to(device)

        targets = sample["target_info"]["targets"]

        meta = meta_base.unsqueeze(0)               # [1,meta_dim]
        out = model(time_domain, meta, rd_shape=rd_shape)

        logits = out["detection_logits"][0, 0].cpu().numpy()  # [D,R]
        rd_map_db = out["rd_map_db"][0].cpu().numpy()         # [D,R]

        prob_t = torch.sigmoid(torch.from_numpy(logits))
        prob = prob_t.numpy()

        # --- DEBUG: prob stats & number of detections before NMS for the first sample
        if i == 0:
            num_over_05 = int((prob >= threshold).sum())
            num_over_02 = int((prob >= 0.2).sum())
            print(
                f"[DEBUG {name}] sample 0 prob stats: "
                f"min={prob.min():.4f}, max={prob.max():.4f}, "
                f"mean={prob.mean():.4f}, "
                f"#(>=0.5)={num_over_05}, #(>=0.2)={num_over_02}"
            )

        dl_dets = logits_to_detections(
            prob,
            dataset.range_axis,     # length R_axis
            dataset.velocity_axis,  # length D_axis
            threshold=threshold,
            nms_kernel_size=3,
        )

        if i == 0:
            print(f"[DEBUG {name}] sample 0 #detections after NMS@{threshold}: {len(dl_dets)}")

        metrics, matched_pairs, unmatched_targets, unmatched_detections = \
            dataset._evaluate_metrics(targets, dl_dets)

        all_tp += metrics["tp"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]
        if metrics["mean_range_error"] > 0:
            all_range_errors.append(metrics["mean_range_error"])
        if metrics["mean_velocity_error"] > 0:
            all_vel_errors.append(metrics["mean_velocity_error"])

        # Visualization for a few samples
        if vis_count < max_vis_samples:
            vis_count += 1
            rdm_norm = rd_map_db - np.max(rd_map_db)

            save_path_2d = os.path.join(
                out_dir, f"{name_slug}_sample_{i}_DL_2d.png"
            )
            _plot_2d_rdm(
                dataset,
                rdm_norm,
                i,
                metrics,
                matched_pairs,
                unmatched_targets,
                unmatched_detections,
                save_path_2d,
            )

            save_path_3d = os.path.join(
                out_dir, f"{name_slug}_sample_{i}_DL_3d.png"
            )
            _plot_3d_rdm(
                dataset,
                rdm_norm,
                i,
                targets,
                dl_dets,
                save_path_3d,
            )

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    mean_range_err = np.mean(all_range_errors) if all_range_errors else 0.0
    mean_vel_err = np.mean(all_vel_errors) if all_vel_errors else 0.0

    print(f"\n--- Deep Model Results ({name}) ---")
    print(f"Samples: {len(dataset)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print(f"Mean Range Error: {mean_range_err:.4f} m")
    print(f"Mean Vel Error:   {mean_vel_err:.4f} m/s")
    print(f"DL eval visualizations saved to: {out_dir}")
    print("-" * 30)


# ==============================================================
# MAIN: multi-config training + CFAR vs DL comparison
# ==============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Radar configurations to use in training/eval
    train_config_names = ["config1", "config2", "config_cn0566"]
    val_config_names = train_config_names

    # Training "grid":
    batch_size = 4                # per-config mini-batch size
    num_batches_per_config = 50   # mini-batches per config per epoch
    num_epochs = 10

    # Option: reuse generated training datasets across epochs
    reuse_train_datasets = True

    base_data_dir = "data/dl_generalization_v3"

    # ----------------------------------------------------------
    # Build training datasets (if we choose to reuse)
    # ----------------------------------------------------------
    train_datasets = None
    if reuse_train_datasets:
        train_datasets = {}
        for cfg_name in train_config_names:
            cfg_dir = os.path.join(base_data_dir, f"{cfg_name}_train_reuse")
            num_samples = batch_size * num_batches_per_config
            ds = AIRadarDataset(
                num_samples=num_samples,
                config_name=cfg_name,
                save_path=cfg_dir,
                drawfig=False,
                apply_realistic_effects=True,
                clutter_intensity=0.3,
            )
            train_datasets[cfg_name] = ds
            print(
                f"Train dataset for {cfg_name}: {len(ds)} samples "
                f"(reused every epoch)"
            )

    # ----------------------------------------------------------
    # Build validation datasets (fixed across epochs)
    # ----------------------------------------------------------
    per_config_val_samples = 100
    val_datasets = {}
    for cfg_name in val_config_names:
        cfg_dir = os.path.join(base_data_dir, f"{cfg_name}_val")
        ds = AIRadarDataset(
            num_samples=per_config_val_samples,
            config_name=cfg_name,
            save_path=cfg_dir,
            drawfig=False,
            apply_realistic_effects=True,
            clutter_intensity=0.3,
        )
        val_datasets[cfg_name] = ds
        print(f"Validation dataset for {cfg_name}: {len(ds)} samples")

    # ----------------------------------------------------------
    # Model + optimizer + loss
    # ----------------------------------------------------------
    meta_dim = 6
    model = RadarTimeNetV6(
        meta_dim=meta_dim,
        num_rx=1,
    ).to(device)

    criterion = CombinedLoss(w_bce=0.5, w_focal=1.0, w_dice=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------------------------------------
    # Training loop: (optionally) reuse datasets + config-rotating batches
    # ----------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        train_one_epoch_config_rotation(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config_names=train_config_names,
            batch_size=batch_size,
            num_batches_per_config=num_batches_per_config,
            base_data_dir=base_data_dir,
            train_datasets=train_datasets,
            regenerate_each_epoch=not reuse_train_datasets,
        )

        # ----------------------------------------------------------
        # CFAR vs DeepModel comparison per config
        # ----------------------------------------------------------
        num_configs = len(val_datasets)
        for cfg_idx, cfg_name in enumerate(val_config_names):
            ds = val_datasets[cfg_name]
            print("\n" + "=" * 60)
            print(f"Config: {cfg_name} ({ds.config.get('name','')})")
            print("=" * 60)

            # CFAR baseline
            evaluate_dataset_metrics(ds, f"CFAR {cfg_name}")

            # Deep model
            out_dir = os.path.join(base_data_dir, f"{cfg_name}_DL_eval")
            evaluate_model_on_dataset(
                model=model,
                dataset=ds,
                device=device,
                config_index=cfg_idx,
                num_configs=num_configs,
                name=f"DeepModel {cfg_name}",
                out_dir=out_dir,
                threshold=0.5,
                max_vis_samples=3,
            )


if __name__ == "__main__":
    main()