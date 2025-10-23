import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from einops import rearrange, repeat

# Import từ các file gốc (giả định đã có)
# from .base import Base, init_layer, init_bn
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# Placeholder cho các hàm init (copy từ file gốc)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning"""
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()
        self.condition_size = condition_size
        self.modules, _ = self.create_film_modules(
            film_meta=film_meta, 
            ancestor_names=[],
        )
        
    def create_film_modules(self, film_meta, ancestor_names):
        modules = {}
        for module_name, value in film_meta.items():
            if isinstance(value, int):
                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)
                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value, 
                    unique_module_name=unique_module_name,
                )
            elif isinstance(value, dict):
                ancestor_names.append(module_name)
                modules[module_name], _ = self.create_film_modules(
                    film_meta=value, 
                    ancestor_names=ancestor_names,
                )
            ancestor_names.pop()
        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):
        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)
        return layer

    def forward(self, conditions):
        film_dict = self.calculate_film_data(
            conditions=conditions, 
            modules=self.modules,
        )
        return film_dict

    def calculate_film_data(self, conditions, modules):
        film_data = {}
        for module_name, module in modules.items():
            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]
            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)
        return film_data


class MambaBlock(nn.Module):
    """Simplified Mamba block for audio processing"""
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        has_film=True,
        momentum=0.01,
    ):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.has_film = has_film
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(d_model, momentum=momentum)
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + d_state * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        
        # SSM parameters
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.act = nn.SiLU()
        self.init_weights()
        
    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.in_proj)
        init_layer(self.dt_proj)
        init_layer(self.out_proj)
        
    def forward(self, x, film_dict=None):
        """
        x: (B, C, T, F)
        """
        B, C, T, F = x.shape
        
        # Apply FiLM conditioning
        if film_dict is not None and self.has_film:
            beta = film_dict.get('beta1', 0)
            x = self.bn1(x) + beta
        else:
            x = self.bn1(x)
            
        # Reshape for sequence processing
        x = rearrange(x, 'b c t f -> b (t f) c')
        
        # Input projection
        xz = self.in_proj(x)
        x, z, B_state, C_state = torch.split(
            xz, 
            [self.d_inner, self.d_inner, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Convolution along sequence
        x_conv = rearrange(x, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :x_conv.shape[-1]]
        x_conv = self.act(x_conv)
        x = rearrange(x_conv, 'b d l -> b l d')
        
        # Simplified SSM (linear for efficiency)
        dt = self.dt_proj(x)
        y = x * torch.sigmoid(dt) + x * torch.tanh(B_state) * torch.tanh(C_state)
        
        # Gate with z
        y = y * self.act(z)
        
        # Output projection
        y = self.out_proj(y)
        
        # Reshape back
        y = rearrange(y, 'b (t f) c -> b c t f', t=T, f=F)
        
        return y


class ResidualMambaBlock(nn.Module):
    """Residual block with Mamba for temporal modeling"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        momentum=0.01,
        has_film=True,
        use_mamba=True,
    ):
        super(ResidualMambaBlock, self).__init__()
        
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        self.has_film = has_film
        self.use_mamba = use_mamba
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            bias=False,
        )
        
        if use_mamba:
            self.mamba = MambaBlock(
                d_model=out_channels,
                has_film=has_film,
                momentum=momentum,
            )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            bias=False,
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False
            
        self.init_weights()
        
    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut:
            init_layer(self.shortcut)
            
    def forward(self, x, film_dict=None):
        if film_dict is not None and self.has_film:
            b1 = film_dict.get('beta1', 0)
            b2 = film_dict.get('beta2', 0)
        else:
            b1 = b2 = 0
            
        # First conv
        out = self.conv1(F.leaky_relu_(self.bn1(x) + b1, negative_slope=0.01))
        
        # Mamba block for temporal modeling
        if self.use_mamba:
            mamba_dict = film_dict.get('mamba', None) if film_dict is not None else None
            out = self.mamba(out, mamba_dict)
        
        # Second conv
        out = self.conv2(F.leaky_relu_(self.bn2(out) + b2, negative_slope=0.01))
        
        # Residual connection
        if self.is_shortcut:
            return self.shortcut(x) + out
        else:
            return x + out


class EncoderBlockResMamba(nn.Module):
    """Encoder block with Mamba"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        downsample=(2, 2),
        momentum=0.01,
        has_film=True,
        use_mamba=True,
    ):
        super(EncoderBlockResMamba, self).__init__()
        
        self.res_mamba_block = ResidualMambaBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            momentum=momentum,
            has_film=has_film,
            use_mamba=use_mamba,
        )
        self.downsample = downsample
        
    def forward(self, x, film_dict=None):
        encoder = self.res_mamba_block(x, film_dict)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockResMamba(nn.Module):
    """Decoder block with Mamba"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        upsample=(2, 2),
        momentum=0.01,
        has_film=True,
        use_mamba=True,
    ):
        super(DecoderBlockResMamba, self).__init__()
        
        self.stride = upsample
        self.has_film = has_film
        
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        
        self.res_mamba_block = ResidualMambaBlock(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            momentum=momentum,
            has_film=has_film,
            use_mamba=use_mamba,
        )
        
        self.init_weights()
        
    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        
    def forward(self, x, concat_tensor, film_dict=None):
        if film_dict is not None and self.has_film:
            beta = film_dict.get('beta1', 0)
        else:
            beta = 0
            
        # Upsample
        x = self.conv1(F.leaky_relu_(self.bn1(x) + beta, negative_slope=0.01))
        
        # Concatenate with encoder features
        x = torch.cat((x, concat_tensor), dim=1)
        
        # Process with ResidualMambaBlock
        res_mamba_dict = film_dict.get('res_mamba_block', None) if film_dict is not None else None
        x = self.res_mamba_block(x, res_mamba_dict)
        
        return x


class ResMamba_Base(nn.Module):
    """Base ResMamba model for audio source separation"""
    def __init__(self, input_channels, output_channels, target_sources_num):
        super(ResMamba_Base, self).__init__()
        
        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01
        
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.target_sources_num = target_sources_num
        self.K = 3
        self.time_downsample_ratio = 2 ** 5
        
        # STFT/ISTFT
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        
        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)
        
        # Pre-processing
        self.pre_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        # Encoder with Mamba
        self.encoder_block1 = EncoderBlockResMamba(32, 32, downsample=(2, 2), momentum=momentum, has_film=True)
        self.encoder_block2 = EncoderBlockResMamba(32, 64, downsample=(2, 2), momentum=momentum, has_film=True)
        self.encoder_block3 = EncoderBlockResMamba(64, 128, downsample=(2, 2), momentum=momentum, has_film=True)
        self.encoder_block4 = EncoderBlockResMamba(128, 256, downsample=(2, 2), momentum=momentum, has_film=True)
        self.encoder_block5 = EncoderBlockResMamba(256, 384, downsample=(2, 2), momentum=momentum, has_film=True)
        self.encoder_block6 = EncoderBlockResMamba(384, 384, downsample=(1, 2), momentum=momentum, has_film=True)
        
        # Bottleneck
        self.conv_block7 = EncoderBlockResMamba(384, 384, downsample=(1, 1), momentum=momentum, has_film=True)
        
        # Decoder with Mamba
        self.decoder_block1 = DecoderBlockResMamba(384, 384, upsample=(1, 2), momentum=momentum, has_film=True)
        self.decoder_block2 = DecoderBlockResMamba(384, 384, upsample=(2, 2), momentum=momentum, has_film=True)
        self.decoder_block3 = DecoderBlockResMamba(384, 256, upsample=(2, 2), momentum=momentum, has_film=True)
        self.decoder_block4 = DecoderBlockResMamba(256, 128, upsample=(2, 2), momentum=momentum, has_film=True)
        self.decoder_block5 = DecoderBlockResMamba(128, 64, upsample=(2, 2), momentum=momentum, has_film=True)
        self.decoder_block6 = DecoderBlockResMamba(64, 32, upsample=(2, 2), momentum=momentum, has_film=True)
        
        # Post-processing
        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=input_channels * self.K * self.target_sources_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        self.out_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        self.init_weights()
        
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)
        init_layer(self.out_conv)
        
    def wav_to_spectrogram_phase(self, input_waveform):
        """Convert waveform to magnitude and phase spectrograms"""
        # input_waveform: (B, C, T)
        B, C, T = input_waveform.shape
        
        # Reshape to (B*C, T)
        x = input_waveform.reshape(B * C, T)
        
        # STFT
        real, imag = self.stft(x)
        # real, imag: (B*C, T', F)
        
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        cos = real / (mag + 1e-8)
        sin = imag / (mag + 1e-8)
        
        # Reshape back to (B, C, T', F)
        _, T_spec, F = mag.shape
        mag = mag.reshape(B, C, T_spec, F)
        cos = cos.reshape(B, C, T_spec, F)
        sin = sin.reshape(B, C, T_spec, F)
        
        return mag, cos, sin
    
    def feature_maps_to_wav(self, input_tensor, sp, sin_in, cos_in, audio_length, source_mask):
        """Convert feature maps to waveform"""
        B, _, T, F = input_tensor.shape
        
        x = input_tensor.reshape(B, self.target_sources_num, self.input_channels, self.K, T, F)
        
        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        
        out_cos = cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        out_sin = sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        
        shape = (B * self.target_sources_num, self.input_channels, T, F)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)
        
        out_real = self.out_conv(out_real)
        out_imag = self.out_conv(out_imag)
        
        shape = (B * self.target_sources_num * self.output_channels, 1, T, F)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)
        
        x = self.istft(out_real, out_imag, audio_length)
        
        if self.target_sources_num == 1:
            waveform = x.reshape(B, self.output_channels, audio_length)
            waveform = waveform * source_mask[:, :, None]
        else:
            waveform = x.reshape(B, self.target_sources_num, self.output_channels, audio_length)
            waveform = waveform * source_mask[:, :, None, None]
            
        return waveform
    
    def forward(self, mixtures, film_dict, source_mask):
        """Forward pass"""
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag
        
        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # Padding
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0:x.shape[-1] - 1]
        
        # U-Net with Mamba
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict.get('encoder_block1'))
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict.get('encoder_block2'))
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict.get('encoder_block3'))
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict.get('encoder_block4'))
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict.get('encoder_block5'))
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict.get('encoder_block6'))
        x_center, _ = self.conv_block7(x6_pool, film_dict.get('conv_block7'))
        
        x7 = self.decoder_block1(x_center, x6, film_dict.get('decoder_block1'))
        x8 = self.decoder_block2(x7, x5, film_dict.get('decoder_block2'))
        x9 = self.decoder_block3(x8, x4, film_dict.get('decoder_block3'))
        x10 = self.decoder_block4(x9, x3, film_dict.get('decoder_block4'))
        x11 = self.decoder_block5(x10, x2, film_dict.get('decoder_block5'))
        x12 = self.decoder_block6(x11, x1, film_dict.get('decoder_block6'))
        
        x = self.after_conv(x12)
        
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]
        
        audio_length = mixtures.shape[2]
        
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            sp=mag,
            sin_in=sin_in,
            cos_in=cos_in,
            audio_length=audio_length,
            source_mask=source_mask,
        )
        
        return {'waveform': separated_audio, 'source_mask': source_mask}


def get_film_meta(module):
    """Extract FiLM metadata from module hierarchy"""
    film_meta = {}
    
    if hasattr(module, 'has_film'):
        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            if hasattr(module, 'bn2'):
                film_meta['beta2'] = module.bn2.num_features
            if hasattr(module, 'mamba'):
                film_meta['mamba'] = {'beta1': module.mamba.d_model}
    
    for child_name, child_module in module.named_children():
        child_meta = get_film_meta(child_module)
        if len(child_meta) > 0:
            film_meta[child_name] = child_meta
    
    return film_meta


class ResMamba(nn.Module):
    """Complete ResMamba model with FiLM conditioning"""
    def __init__(self, input_channels, output_channels, target_sources_num, label_len):
        super(ResMamba, self).__init__()
        
        self.target_sources_num = target_sources_num
        
        self.base = ResMamba_Base(
            input_channels=input_channels,
            output_channels=output_channels,
            target_sources_num=target_sources_num,
        )
        
        self.film_meta = get_film_meta(module=self.base)
        
        # Label embedding
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        self.film = FiLM(
            film_meta=self.film_meta,
            condition_size=512
        )
        
    def forward(self, input_dict):
        """Forward pass with label conditioning"""
        mixtures = input_dict['mixture']
        label_vector = input_dict['label_vector']
        
        # Create source mask
        with torch.no_grad():
            copylb = label_vector.clone()
            bs, lbl = copylb.shape
            source_mask = copylb.view(bs, self.target_sources_num, lbl // self.target_sources_num).sum(dim=2)
            source_mask = (source_mask != 0).to(label_vector.dtype)
        
        # Get FiLM parameters from label
        conditions = self.label_embedding(label_vector)
        film_dict = self.film(conditions=conditions)
        
        # Forward through base model
        output_dict = self.base(
            mixtures=mixtures,
            film_dict=film_dict,
            source_mask=source_mask
        )
        
        return output_dict
