import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .base import Base, init_layer, init_bn, act
from mamba_ssm.modules.mamba2 import Mamba2
from einops import rearrange, repeat

class FiLMGenerator(nn.Module):
    def __init__(self, context_dim, feature_dim):
        super(FiLMGenerator, self).__init__()

        self.gamma_genertor = nn.Linear(context_dim, feature_dim)

        self.beta_generator = nn.Linear(context_dim, feature_dim)

        self.init_weights()
    def init_weights(self):
        init_layer(self.gamma_genertor)
        init_layer(self.beta_generator)

    def forward(self, context_vector):
        gamma = self.gamma_genertor(context_vector)
        beta = self.beta_generator(context_vector)
        return gamma, beta
    


class ConditionalMamba2DBlock(nn.Module):
    def __init__(self, d_model, context_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.film_generator = FiLMGenerator(context_dim=context_dim, feature_dim=d_model)

    def forward(self, u, context_vector):
        residual = u
        B, C, T, F = u.shape

        # 1. Chuẩn hóa và "Tuần tự hóa"
        # [B, C, T, F] -> [B, C, F, T] -> [B*F, T, C]
        x = u.permute(0, 3, 2, 1).contiguous() # -> [B, F, T, C]
        x = x.view(-1, T, C) # -> [B*F, T, C]

        # LayerNorm hoạt động trên chiều feature cuối cùng (C)
        x = self.norm(x)

        # 2. Xử lý bằng Mamba
        # Mamba hoạt động trên chuỗi thời gian T
        x = self.mamba(x) # -> [B*F, T, C]

        # 3. "Phi tuần tự hóa"
        # [B*F, T, C] -> [B, F, T, C] -> [B, C, T, F]
        x = x.view(B, F, T, C)
        x = x.permute(0, 3, 2, 1).contiguous() # -> [B, C, T, F]

        # 4. Điều kiện hóa bằng FiLM
        gamma, beta = self.film_generator(context_vector)
        # gamma, beta từ [B, C] -> [B, C, 1, 1] để broadcast
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        
        # --- Residual Connection (phần 2: cộng lại) ---
        output = residual + x
        
        return output

class MambaUnet_Base(nn.Module, Base):
    def __init__(self, input_channels, output_channels, target_sources_num, context_dim):
        super(MambaUnet_Base, self).__init__()

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
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

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

        # --- Tái cấu trúc các khối Encoder/Decoder ---
        
        # Lớp conv đầu vào vẫn giữ nguyên
        self.pre_conv = nn.Conv2d(input_channels, 32, kernel_size=(1, 1))
        # --- Encoder ---
        self.enc1 = ConditionalMamba2DBlock(32, context_dim)
        self.down1 = nn.AvgPool2d((2, 2))

        self.enc2 = ConditionalMamba2DBlock(64, context_dim)
        self.pre_enc2_conv = nn.Conv2d(32, 64, kernel_size=(1,1)) # Tăng kênh
        self.down2 = nn.AvgPool2d((2, 2))

        self.enc3 = ConditionalMamba2DBlock(128, context_dim)
        self.pre_enc3_conv = nn.Conv2d(64, 128, kernel_size=(1,1))
        self.down3 = nn.AvgPool2d((2, 2))

        # --- Bottleneck (khối trung tâm) ---
        self.bottleneck = ConditionalMamba2DBlock(128, context_dim) # Giả sử kênh cuối là 384

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.dec1 = ConditionalMamba2DBlock(128 * 2, context_dim) # Gấp đôi kênh vì có skip connection
        self.post_dec1_conv = nn.Conv2d(128 * 2, 64, kernel_size=(1,1)) # Giảm kênh

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        self.dec2 = ConditionalMamba2DBlock(64 * 2, context_dim) # Gấp đôi kênh vì có skip connection
        self.post_dec2_conv = nn.Conv2d(64 * 2, 32, kernel_size=(1,1)) # Giảm kênh

        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))
        self.dec3 = ConditionalMamba2DBlock(32 * 2, context_dim) # Gấp đôi kênh vì có skip connection
        self.post_dec3_conv = nn.Conv2d(32 * 2, 32, kernel_size=(1,1)) # Giảm kênh
        
        
        self.after_conv = nn.Conv2d(32, input_channels * self.K * self.target_sources_num, kernel_size=(1, 1))
        self.out_conv = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1))

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, input_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, input_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, input_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, input_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, input_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, input_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * input_channels
        
        shape = (batch_size * self.target_sources_num, self.input_channels, time_steps, freq_bins)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        out_real = self.out_conv(out_real)
        out_imag = self.out_conv(out_imag)
        # out_real_out_ch, out_real_out_ch: (batch_size * target_sources_num, output_channels, time_steps, freq_bins)

        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        
        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        if self.target_sources_num == 1: # compatible with previous code, TODO: modify
            waveform = x.reshape(
                        batch_size, self.output_channels, audio_length
                    )
            waveform = waveform * source_mask[:, :, None]
        else:
            waveform = x.reshape(
                batch_size, self.target_sources_num, self.output_channels, audio_length
            )
            waveform = waveform * source_mask[:, :, None, None]
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform
    
    def forward(self, mixtures, context_vector, source_mask):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)
    
        x1 = self.pre_conv(x)
        x1_proc = self.enc1(x1, context_vector)
        x1_pool = self.down1(x1_proc)

        x2 = self.pre_enc2_conv(x1_pool)
        x2_proc = self.enc2(x2, context_vector)
        x2_pool = self.down2(x2_proc)

        x3 = self.pre_enc3_conv(x2_pool)
        x3_proc = self.enc3(x3, context_vector)
        x3_pool = self.down3(x3_proc)

        x_center = self.bottleneck(x3_pool, context_vector)

        dec1_up = self.up1(x_center)
        x_skip_connection = x3_proc # From enc3
        dec1_cat = torch.cat((dec1_up, x_skip_connection), dim=1)
        dec1_proc = self.dec1(dec1_cat, context_vector)
        dec1_out = self.post_dec1_conv(dec1_proc)

        dec2_up = self.up2(dec1_out)
        x_skip_connection = x2_proc # from enc2
        dec2_cat = torch.cat((dec2_up, x_skip_connection), dim=1)
        dec2_proc = self.dec2(dec2_cat, context_vector)
        dec2_out = self.post_dec2_conv(dec2_proc)

        dec3_up = self.up3(dec2_out)
        x_skip_connection = x1_proc # from enc1
        dec3_cat = torch.cat((dec3_up, x_skip_connection), dim=1)
        dec3_proc = self.dec3(dec3_cat, context_vector)
        dec3_out = self.post_dec3_conv(dec3_proc)

        x = self.after_conv(dec3_out)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
            source_mask=source_mask,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio, 'source_mask': source_mask}

        return output_dict

class MambaUNet(nn.Module):
    def __init__(self, input_channels, output_channels, target_sources_num, label_len):
        super(MambaUNet, self).__init__()
        self.target_sources_num = target_sources_num
        context_dim = 512
        self.base = MambaUnet_Base(
            input_channels=input_channels, 
            output_channels=output_channels,
            target_sources_num=target_sources_num,
            context_dim=context_dim
        )
        

        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU())

    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        label_vector = input_dict['label_vector']

        with torch.no_grad():
            copylb = label_vector.clone()
            bs, lbl = copylb.shape
            source_mask = copylb.view(bs, self.target_sources_num, lbl//self.target_sources_num).sum(dim=2)
            source_mask = (source_mask!=0).to(label_vector.dtype)
        context_vector = self.label_embedding(label_vector)
    
        output_dict = self.base(
            mixtures=mixtures, 
            context_vector=context_vector,
            source_mask=source_mask #[bs, target_sources_num]
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
                film_dict=film_dict,
            )['waveform']
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np