# train_skeleton.py
import os
import math
import random
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# ---------- STFT / audio utils ----------
class STFTHelper:
    def __init__(self, sr=32000, n_fft=2048, hop_length=512, win_length=None, device='cpu'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length or n_fft
        self.window = torch.hann_window(self.win).to(device)
        self.device = device

    def stft(self, w):
        # w: (B, L)
        return torch.stft(w, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, return_complex=True, center=True, pad_mode='reflect')

    def istft(self, complex_spec, length=None):
        return torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop,
                           win_length=self.win, window=self.window, length=length)

    def magphase(self, complex_spec):
        mag = complex_spec.abs()
        phase = torch.angle(complex_spec)
        return mag, phase

    def complex_from_mag_phase(self, mag, phase):
        # phase in radians
        return mag * torch.exp(1j * phase)

    def log_compress(self, mag, C=1.0):
        return torch.log1p(C * mag)

    def log_decompress(self, logmag, C=1.0):
        return (torch.exp(logmag) - 1.0) / C

# ---------- Dataset ----------
class SeparationDataset(torch.utils.data.Dataset):
    """
    Expected directory layout (example):
      data/
        mixtures/*.wav
        sources/track1/*.wav  (or one folder per source type)
        sources/track2/*.wav
        ...
    Or you can provide pairs of file paths in a manifest.
    This skeleton shows how to compute masks from source/mix STFTs.
    """
    def __init__(self, manifest, stft_helper: STFTHelper, n_sources=3, sample_rate=32000, segment_len=None):
        """
        manifest: list of dicts {'mix': path, 'sources': [path1, path2, ...]}
        """
        self.manifest = manifest
        self.stft = stft_helper
        self.n_sources = n_sources
        self.sample_rate = sample_rate
        self.segment_len = segment_len  # in samples, or None to use full

    def __len__(self):
        return len(self.manifest)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # to mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wav

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        mix_wav = self._load_audio(entry['mix'])

        # optional cropping
        if self.segment_len and mix_wav.shape[0] > self.segment_len:
            start = random.randint(0, mix_wav.shape[0] - self.segment_len)
            mix_wav = mix_wav[start:start + self.segment_len]

        mix_spec = self.stft.stft(mix_wav.unsqueeze(0))  # (1, F, T) complex
        mix_mag = mix_spec.abs()[0]  # (F,T)
        mix_phase = torch.angle(mix_spec)[0]

        # load sources
        src_mags = []
        for i in range(self.n_sources):
            if i < len(entry['sources']):
                src_wav = self._load_audio(entry['sources'][i])
                if self.segment_len and src_wav.shape[0] > self.segment_len:
                    # align crop same as mix (simplified)
                    pass
                src_spec = self.stft.stft(src_wav.unsqueeze(0))[0]
                src_mags.append(src_spec.abs())
            else:
                # silent if missing
                src_mags.append(torch.zeros_like(mix_mag))

        src_mags = torch.stack(src_mags, dim=0)  # (n_src, F, T)

        # masks: avoid div-by-zero
        eps = 1e-8
        denom = mix_mag.unsqueeze(0).clamp(min=eps)
        masks = src_mags / denom
        masks = masks.clamp(0.0, 1.0)

        # compress
        log_mix = self.stft.log_compress(mix_mag, C=1.0)
        # normalize per-sample (optional)
        log_mix = (log_mix - log_mix.mean()) / (log_mix.std() + 1e-6)
        # similarly normalize masks? masks already [0,1]
        return {
            'log_mix': log_mix,          # (F, T)
            'mix_phase': mix_phase,      # (F, T)
            'masks': masks               # (n_src, F, T)
        }

# ---------- Simple building blocks ----------
def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch)
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))

# ---------- FF-Parser (FFT-based filtering on features) ----------
class FFParser(nn.Module):
    """
    Approximate the paper's FF-Parser:
      - compute 2D FFT of feature map (F x T)
      - build a small network that predicts a real-valued gating map for each frequency/time bin
      - multiply in Fourier domain and inverse FFT
    """
    def __init__(self, channels):
        super().__init__()
        # small conv net that consumes log-magnitude of spectrum and outputs weights
        # input shape: (B, C, H, W); we output (B, C, H, W) weights in (0,1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        # feat: (B, C, H, W) real
        # get complex spectrum
        M = torch.fft.fft2(feat, dim=(-2, -1))  # complex
        # compute magnitude feature to feed net
        mag = torch.log1p(M.abs())
        weights = self.net(mag)  # real in (0,1)
        # broadcast multiply (complex * real)
        M_filtered = M * weights
        out = torch.fft.ifft2(M_filtered, dim=(-2, -1)).real
        return out

# ---------- UNet-like model with dynamic conditional encoding ----------
class SimpleCondUNet(nn.Module):
    def __init__(self, n_sources=3, base_ch=64, timesteps=1000):
        super().__init__()
        self.n_sources = n_sources
        self.base_ch = base_ch
        # two encoders: one for mixture condition (1 channel), one for current xt (n_sources channels)
        self.enc_mix1 = conv_block(1, base_ch)
        self.enc_x1 = conv_block(n_sources, base_ch)

        self.down1 = conv_block(base_ch, base_ch*2)
        self.down2 = conv_block(base_ch*2, base_ch*4)

        self.middle = nn.Sequential(conv_block(base_ch*4, base_ch*4), ResidualBlock(base_ch*4))

        # FF-Parser applied on skip paths
        self.ff1 = FFParser(base_ch*2)
        self.ff2 = FFParser(base_ch*4)

        # decoder
        self.up2 = conv_block(base_ch*8, base_ch*4)
        self.up1 = conv_block(base_ch*4 + base_ch*2, base_ch*2)
        self.out_conv = nn.Sequential(conv_block(base_ch*2, base_ch), nn.Conv2d(base_ch, n_sources, 1))

        # time embedding
        self.time_mlp = nn.Sequential(nn.Linear(timesteps, base_ch*4), nn.SiLU(), nn.Linear(base_ch*4, base_ch*4))

        # a small conv to combine fusion affinity (optionally)
        self.aff_proj = nn.Conv2d(base_ch*4, base_ch*4, 1)

    def forward(self, x_t, cond_logmix, t_emb):
        """
        x_t: (B, n_sources, F, T)  -- noisy masks
        cond_logmix: (B, 1, F, T)
        t_emb: (B, D) or int tensor (we expect we pass embedding vector)
        """
        # encode mix and x separately
        m1 = self.enc_mix1(cond_logmix)    # (B, C, H, W)
        x1 = self.enc_x1(x_t)

        d1_mix = self.down1(m1)
        d1_x = self.down1(x1)

        # dynamic conditional encoding at this scale:
        # use channel-wise GroupNorm as approximation of LayerNorm over channels
        A1 = F.instance_norm(d1_mix) * F.instance_norm(d1_x)  # (B,C,H,W)
        fused1 = d1_mix * A1  # attentive-like fusion
        fused1 = self.ff1(fused1)

        d2_mix = self.down2(fused1)
        d2_x = self.down2(d1_x)  # re-use x path

        A2 = F.instance_norm(d2_mix) * F.instance_norm(d2_x)
        fused2 = d2_mix * A2
        fused2 = self.ff2(fused2)

        mid = self.middle(fused2)

        # time embedding: if t_emb vector, broadcast and add
        if t_emb is not None:
            # expand to spatial dims and add
            b, _, h, w = mid.shape
            tm = t_emb.view(b, -1, 1, 1).expand(-1, -1, h, w)
            mid = mid + self.aff_proj(tm)

        # decoder: upsample + concat skip
        up = torch.cat([F.interpolate(mid, scale_factor=2, mode='bilinear', align_corners=False), fused2], dim=1)
        up = self.up2(up)
        up = torch.cat([F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False), fused1], dim=1)
        up = self.up1(up)
        out = self.out_conv(up)  # (B, n_sources, F, T)
        # predict noise epsilon for x_t
        return out

# ---------- Diffusion helpers ----------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-8, 0.999)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        betas = cosine_beta_schedule(timesteps)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B, C, F, T)
        t: (B,) ints in [0, timesteps-1]
        returns x_t and noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # extract sqrt terms per sample
        b = x0.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
        sqrt_omt = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_omt * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        b = x_t.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
        sqrt_omt = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
        return (x_t - sqrt_omt * eps) / (sqrt_alpha + 1e-8)

    # simplified single-step DDPM posterior mean (not full ancestral sampling)
    def p_sample(self, model, x_t, cond, t):
        # model predicts eps
        eps = model(x_t, cond['log_mix'].unsqueeze(1), cond['t_emb'])  # (B, n_src, F, T)
        # predicted x0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        # compute posterior mean (simplified)
        b = x_t.shape[0]
        beta_t = self.betas[t].view(b, 1, 1, 1)
        alpha_t = self.alphas[t].view(b, 1, 1, 1)
        alpha_cum = self.alphas_cumprod[t].view(b, 1, 1, 1)
        mean = (torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_cum)) * eps)) / (1e-8 + alpha_t)
        # optionally sample noise
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        sigma = torch.sqrt(beta_t)
        x_prev = mean + sigma * noise
        return x_prev, x0_pred

# ---------- Losses ----------
def eps_mse_loss(eps_pred, eps_target):
    return F.mse_loss(eps_pred, eps_target)

def mask_l1_loss(mask_pred, mask_gt):
    return F.l1_loss(mask_pred, mask_gt)

def si_sdr(est, ref, eps=1e-8):
    """
    est, ref: (B, L)
    returns SI-SDR in dB (averaged over batch)
    """
    def pairwise_si_sdr(a, b):
        # a: est, b: ref
        b_energy = torch.sum(b ** 2, dim=1, keepdim=True) + eps
        proj = torch.sum(a * b, dim=1, keepdim=True) * b / b_energy
        noise = a - proj
        ratio = torch.sum(proj ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)
        return 10 * torch.log10(ratio + eps)
    return torch.mean(pairwise_si_sdr(est, ref))

# ---------- Training loop skeleton ----------
def train_one_epoch(model, diffusion, dataloader, optimizer, stft: STFTHelper, device, epoch, config):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        log_mix = batch['log_mix'].to(device)             # (B, F, T)
        masks = batch['masks'].to(device)                 # (B, n_src, F, T)
        # expand dims
        log_mix = log_mix.unsqueeze(1)                    # (B,1,F,T)
        B = masks.shape[0]
        # sample t
        t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
        # create x0 (masks), q_sample -> x_t
        x0 = masks
        x_t, noise = diffusion.q_sample(x0, t)
        # time embedding: simple sinusoidal or one-hot embedding
        t_emb = F.one_hot(t, num_classes=diffusion.timesteps).float().to(device)

        # forward model predicts eps
        eps_pred = model(x_t, log_mix, t_emb)
        loss_eps = eps_mse_loss(eps_pred, noise)

        # optional mask L1 on predicted x0
        x0_pred = diffusion.predict_x0_from_eps(x_t, t, eps_pred).clamp(0.0, 1.0)
        loss_mask = mask_l1_loss(x0_pred, x0)

        # optional SI-SDR: reconstruct waveforms using predicted masks
        loss_sisdr = torch.tensor(0.0, device=device)
        if config.get('use_sisdr', False) and (i % config.get('sisdr_every', 5) == 0):
            # reconstruct waveform from mask applied to mix mag + mix phase
            mix_complex = stft.stft(torch.zeros(B).to(device))  # placeholder - we need mix complex per sample
            # NOTE: for efficiency and simplicity in skeleton we skip per-batch reconstruction here
            pass

        loss = config['w_eps'] * loss_eps + config['w_mask'] * loss_mask  # + w_sisdr * loss_sisdr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % config.get('log_every', 100) == 0:
            print(f"Epoch {epoch} iter {i+1}/{len(dataloader)} loss {total_loss / (i+1):.4f}")

    return total_loss / len(dataloader)

# ---------- Sampling / inference ----------
@torch.no_grad()
def sample_from_model(model, diffusion: Diffusion, log_mix, stft: STFTHelper, device, steps=100):
    """
    log_mix: (1, F, T) normalized same as training; returns list of reconstructed source waveforms
    """
    model.eval()
    B = 1
    n_src = model.n_sources
    # shape of x: same as masks (B, n_src, F, T)
    Fdim, Tdim = log_mix.shape[-2], log_mix.shape[-1]
    x_t = torch.randn((B, n_src, Fdim, Tdim), device=device)
    for step in reversed(range(0, diffusion.timesteps)):
        t = torch.tensor([step], device=device).long()
        t_emb = F.one_hot(t, num_classes=diffusion.timesteps).float().to(device)
        cond = {'log_mix': log_mix.to(device).unsqueeze(0), 't_emb': t_emb}
        x_t, x0_pred = diffusion.p_sample(model, x_t, cond, t)
        # optionally clamp
        x_t = x_t.clamp(0.0, 1.0)
    # final predicted masks:
    masks = x0_pred.clamp(0.0, 1.0)[0]  # (n_src, F, T)
    # decompress logs etc - in our skeleton we assume masks applied to original (unnormalized) mag
    # user should pass original mix mag and phase to reconstruct wav
    return masks.cpu().numpy()

# ---------- Minimal run (example) ----------
if __name__ == "__main__":
    # Configure paths / manifest manually for your dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = STFTHelper(sr=32000, n_fft=2048, hop_length=512, device=device)

    # manifest = [{'mix': 'path/to/mix.wav', 'sources': ['path/to/s1.wav','path/to/s2.wav','path/to/s3.wav']}, ...]
    manifest = []  # TODO: load your dataset manifest here

    dataset = SeparationDataset(manifest, st, n_sources=3, sample_rate=32000, segment_len=32000*5)  # 5s crops 
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    model = SimpleCondUNet(n_sources=3, base_ch=32, timesteps=100).to(device)
    diffusion = Diffusion(timesteps=100, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    config = {'w_eps': 1.0, 'w_mask': 1.0, 'use_sisdr': False, 'log_every': 10}
    for epoch in range(10):
        train_one_epoch(model, diffusion, loader, optimizer, st, device, epoch, config)
        # checkpoint save
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()}, f'ckpt_epoch{epoch}.pt')
