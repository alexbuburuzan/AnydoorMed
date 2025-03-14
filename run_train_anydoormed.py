import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.vindr_mammo import VindrMammoDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--logger_freq', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--sd_locked', action='store_true')
parser.add_argument('--only_mid_control', action='store_true')
parser.add_argument('--n_gpus', type=int, default=2)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
logger_freq = args.logger_freq
learning_rate = args.learning_rate
sd_locked = args.sd_locked
only_mid_control = args.only_mid_control
n_gpus = args.n_gpus
accumulate_grad_batches = args.accumulate_grad_batches
num_workers = args.num_workers

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

resume_path = 'checkpoints/epoch=1-step=8687.ckpt'

# import pdb; pdb.set_trace()

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset = VindrMammoDataset(**DConf.Train.VindrMammoDataset)

# The ratio of each dataset is adjusted by setting the __len__ 
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=n_gpus, strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, dataloader)
