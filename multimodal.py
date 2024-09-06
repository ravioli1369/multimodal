import os
import torch
import train
import ml4gw
import train.priors
import train.data.datasets.flow
import train.data.waveforms.generator.cbc
import ml4gw.waveforms
import ml4gw.distributions
import ml4gw.transforms
import ml4gw.dataloading
from pathlib import Path

dec = ml4gw.distributions.Cosine()
psi = torch.distributions.Uniform(0, 3.14)
phi = torch.distributions.Uniform(-3.14, 3.14)
waveform_sampler = train.data.waveforms.generator.cbc.FrequencyDomainCBCGenerator(
    approximant=ml4gw.waveforms.IMRPhenomD(),
    f_min=20,
    f_max=300,
    waveform_arguments={"f_ref": 40},
    num_test_waveforms=1,
    num_val_waveforms=1,
    parameter_sampler=train.priors.cbc_prior,
    num_fit_params=100000,
    inference_params=[
        "chirp_mass",
        "mass_ratio",
        "distance",
        "phic",
        "inclination",
        "dec",
        "psi",
        "phi",
    ],
    sample_rate=2048,
    dec=dec,
    psi=psi,
    phi=phi,
    duration=4,
)

datamodule = train.data.datasets.flow.FlowDataset(
    data_dir=Path(os.environ["AMPLFI_DATADIR"]),
    inference_params=[
        "chirp_mass",
        "mass_ratio",
        "distance",
        "phic",
        "inclination",
        "dec",
        "psi",
        "phi",
    ],
    highpass=25,
    sample_rate=2048,
    kernel_length=3,
    fduration=1,
    psd_length=10,
    fftlength=2,
    batches_per_epoch=200,
    batch_size=512,
    ifos=["H1", "L1"],
    min_valid_duration=10000.0,
    waveform_sampler=waveform_sampler,
)

# world_size, rank = datamodule.get_world_size_and_rank()
# background_fname = [datamodule.train_fnames[0]]
# [background] = datamodule.load_background(background_fname)
# cross, plus, parameters = datamodule.waveform_sampler.get_val_waveforms(
#     rank, world_size
# )
# background[0], background[1] = background[0][: len(cross)], background[1][: len(cross)]
# datamodule._logger = datamodule.get_logger(world_size, rank)
# datamodule.build_transforms("fit")


# datamodule.inject(background, cross, plus, parameters)
world_size, rank = datamodule.get_world_size_and_rank()
[background] = datamodule.load_background([datamodule.train_fnames[0]])
datamodule._logger = datamodule.get_logger(world_size, rank)
datamodule.build_transforms("fit")
dataset = ml4gw.dataloading.InMemoryDataset(
    background,
    kernel_size=int(datamodule.hparams.sample_rate * datamodule.sample_length),
    batch_size=datamodule.hparams.batch_size,
    coincident=False,
    batches_per_epoch=datamodule.hparams.batches_per_epoch,
    shuffle=True,
)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
[X] = next(iter(dataloader))
X.to("cuda")

cross, plus, parameters = datamodule.waveform_sampler.sample(X)
