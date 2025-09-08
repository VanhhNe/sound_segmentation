
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from src.diffusion import dist_util, logger
from src.diffusion.resample import create_named_schedule_sampler
from src.datamodules.dataset.s5.dataset_diffusion import DatasetDifussion
from src.diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from pathlib import Path
from src.diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms
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
        label_vector_mode="stack",
        checking=True           # trả về đầy đủ output để kiểm tra
    )

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, collate_fn=dataset.collate_fn)

    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=dataset,
        dataloader=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
