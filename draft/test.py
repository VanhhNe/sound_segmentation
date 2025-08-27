from src.datamodules.dataset.s5.dataset_diffusion import DatasetDifussion
import torch
from torch.utils.data import DataLoader
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from src.utils import LABELS

import warnings
warnings.filterwarnings("ignore", message=".*return_complex.*")

def main():
    dummy_config = {
        "spatialscaper": {
            "foreground_dir": "data/dev_set/sound_event/train",
            "background_dir": "data/dev_set/noise/train",
            "rir_dir": "data/dev_set/room_ir/train",
            "duration": 10.0,
            "sr": 32000,
            "max_event_overlap": 3,
            "ref_db": -50,
            "return_dry": True,
            "return_wet": False,
            "return_ir": False,
            "return_background": False,
            "ref_channel": 0,
            "spatialize_direct_path_time_ms": [6, 50]
        },
        "snr_range": [5, 20],
        "nevent_range": [1, 3],
        "inteference_snr_range": [0, 15],
        "ninterference_range": [1, 2],
        "dataset_length": 50000,
        "shuffle_label": False,
        }

    dataset = DatasetDifussion(
        config=dummy_config,
        n_sources=3,
        label_set="dcase2025t4",
        return_dry=True,        # có thêm dry sources
        label_vector_mode="stack",
        checking=True           # trả về đầy đủ output để kiểm tra
    )

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, collate_fn=dataset.collate_fn)

    print("Full labels:", LABELS['dcase2025t4'])
    sample = dataloader.dataset[1]
    mixture = sample['mixture']
    dry_sources = sample['dry_sources']
    labels = sample['label_vector']
    dry_sources_spec = sample['dry_sources_spec']
    mixture_spec = sample['mixture_spec']
    stack_labels = []
    for label in labels:
        stack_labels.append(LABELS['dcase2025t4'][label.argmax().item()] if label.sum().item() > 0 else 'silence')
        print("Label: ", stack_labels[-1])
    print("Sample labels:", labels)
    print("Sample dry_sources shape:", dry_sources.shape)
    
    N, C, T = dry_sources.shape
    time_axis = np.linspace(0, T / 32000, T)

    fig, axes = plt.subplots(N+1, 1, figsize=(12, 2*N), sharex=True)

    fig2, axes2 = plt.subplots(N+1, 1, figsize=(12, 2*N), sharex=True)

    if N == 1:
        axes = [axes]

    fig1, axes1 = plt.subplots(N+1, 1, figsize=(10, 2*(N+1)))

    for i in range(N):
        waveform = dry_sources[i, 0].cpu().numpy()   # lấy kênh 0
        axes1[i].plot(time_axis, waveform)
        axes1[i].set_title(f"Source {i+1}: {stack_labels[i]}")
        axes1[i].set_ylabel("Amplitude")

    # Mixture
    waveform = mixture[0].cpu().numpy()
    axes1[N].plot(time_axis, waveform)
    axes1[N].set_title("Mixture")
    axes1[N].set_ylabel("Amplitude")
    axes1[N].set_xlabel("Time (s)")

    fig1.tight_layout()
    fig1.savefig("dry_sources_plot.png")
    print("Saved waveform plot to dry_sources_plot.png")


    # --- Figure 2: Spectrograms ---
    fig2, axes2 = plt.subplots(N+1, 1, figsize=(10, 2*(N+1)))

    for i in range(N):
        spectrogram = dry_sources_spec[i].cpu().numpy()
        print(spectrogram.shape)  # (freq, time)
        axes2[i].imshow(10 * np.log10(spectrogram + 1e-10),
                        aspect='auto', origin='lower')
        axes2[i].set_title(f"Spectrogram Source {i+1}: {stack_labels[i]}")
        axes2[i].set_ylabel("Frequency Bin")

    # Mixture
    spectrogram =mixture_spec[0].cpu().numpy()
    axes2[N].imshow(10 * np.log10(spectrogram + 1e-10),
                    aspect='auto', origin='lower')
    axes2[N].set_title("Spectrogram Mixture")
    axes2[N].set_ylabel("Frequency Bin")
    axes2[N].set_xlabel("Time Frame")

    fig2.tight_layout()
    fig2.savefig("dry_sources_spectrogram.png")
    print("Saved spectrogram plot to dry_sources_spectrogram.png")

    plt.close(fig1)
    plt.close(fig2)

if __name__ == "__main__":
    main()