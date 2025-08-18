from src.datamodules.dataset.s5.dataset_diffusion import DatasetDifussion
import torch
from torch.utils.data import DataLoader

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
        label_vector_mode="multihot",
        checking=True           # trả về đầy đủ output để kiểm tra
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)


    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}")
        print("Mixture shape:", batch["mixture"].shape)
        print("Label vector shape:", batch["label_vector"].shape)
        print("Labels:", batch["label"])
        if "dry_sources" in batch:
            print("Dry sources shape:", batch["dry_sources"].shape)
        if "spatialscaper" in batch:
            print("Spatialscaper keys:", batch["spatialscaper"][0].keys())
        if i == 2:  # chỉ in 2 batch đầu
            break

if __name__ == "__main__":
    main()