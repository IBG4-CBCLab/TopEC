# TopEC: Enzyme function prediction from enzyme (pocket) structure

[TopEC: Improved classification of enzyme function by a localized 3D protein descriptor and 3D Graph Neural Networks](https://www.biorxiv.org/content/10.1101/2024.01.31.578271v1)

TopEC is an enzyme function prediction tool which uses graph neural networks to predict the enzyme class according to [International Union of Biochemistry and Molecular Biology (IUBMB)](https://doi.org/10.1093/nar/gkn582) nomenclature. 

This work is created in cooperation with the [HelmholtzAI consultants @ Helmholtz Munich](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-helmholtz-munich/index.html). A big thanks to Marie Piraud and Erinc Merdivan for helping us realize this project. 

Using TopEC you can predict enzyme function from different representations of proteins. We offer three methods to design the graph input at atomic and residue resolution. See the usage section for more details.

![alt text][logo]

[logo]: figure/method_overview_GH_version.png "Method overview"

# Table of Contents
- [General Information](#general-information)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

# General Information

TopEC classifies enzymes using binding site information. The work was developed in cooperation with HelmholtzAI consultants @ Helmholtz Munich. This repository uses [pytorch](https://pytorch.org/get-started/locally/), [pytorch-lightning](https://lightning.ai/pytorch-lightning), [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [hydra](https://hydra.cc/docs/intro/) with neural network models [SchNet](https://doi.org/10.1063/1.5019779) and [DimeNet++](https://doi.org/10.48550/arXiv.2011.14115)

The goal is to test enzyme classification ([list of enzyme classes](https://www.enzyme-database.org/)) from structure using only binding sites. With the hypothesis that the area where binding occurs should be sufficient for enzymatic function prediction. 

As input we use the protein structure (.pdb files) and binding site location coordinates. Around these binding site coordinates we construct our localized 3D descriptor for use in the graph neural networks. 

We show the two implemented networks SchNet and DimeNet++ on two approaches:

- 1) The residue based approach. Here we construct a graph only for the C_{alpha} positions the protein's amino acid.

- 2) The atom based approach. Here we construct graphs for each atom in the protein such that the network learns from a full atomistic view. 


# Installation

This configuration is tested on a compute node within the [JUWELS-Booster supercomputer](https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html#hardware-configuration-of-the-system-name-booster-module). Installing this on a cluster or server where you have access to a GPU with atleast 40Gb VRAM (Nvidia A100 or newer models) is highly recommended. 
The setup is tested with python3.9 and pytorch2.1 using CUDA12.1

### miniconda
Install miniconda from [here](https://docs.conda.io/projects/miniconda/en/latest/)

Execute the following commands to create and activate the conda environment.
```
conda create --file environment.yaml
conda activate topec
```

### python virtual environment
As the code is tested with python3.9 it is recommended to create your python venv on 3.9.

```
python3.9 -m venv topec_venv

source topec_venv/bin/activate

pip install -r requirements.txt
```

### manual
Install the following list of packages.

* [pytorch](https://pytorch.org/get-started/previous-versions/)
* [lightning](https://lightning.ai/pytorch-lightning)
* [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* [torch-cluster](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels)
* [hydra-core](https://hydra.cc/docs/intro/)
* [hydra_colorlog](https://hydra.cc/docs/plugins/colorlog/)
* [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [h5py](https://docs.h5py.org/en/stable/build.html)
* [biopython](https://biopython.org/wiki/Download)
* [rich](https://github.com/Textualize/rich)
* [matplotlib](https://matplotlib.org/stable/)
* [dotenv](https://github.com/theskumar/python-dotenv)
* [wandb](https://docs.wandb.ai/quickstart)

# Usage

We use hydra to parse configuration files in the ``configs/`` folder. Generally you do not need to make any changes except to check your paths are set correctly. We note down the most important configuration files you might want to change if you want to run your own experiments. For a detailed explanation to work with configuration files see [here](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/). 

```
configs/
├── callbacks           <- Controls the early stopping and metrics configuration.
├── datamodule          <- Contains a configuration file for every data set we trained and tested. 
├── experiment          <- Contains an experiment file for each setup we trained and tested.
├── model               <- Contains the configuration settings for each model
└── trainer             <- Contains configuration for the trainer.
create_dataset.yaml
test.yaml
train.yaml
```

* ### Dataset creation

Obtain the PDB files [here](https://fz-juelich.sciebo.de/s/7cOPiXC0iqlh3c9).

We are working on uploading a single .h5 containing all the data that can be immediatly used to train your networks.

Make sure the paths in ``configs/create_dataset.yaml`` are pointing towards the folder you store the pdb structures.
Then execute from command line:

```
python create_h5dataset.py
```

This takes a while as it needs to process many .pdb files.

* ### Tracking Experiments

For tracking your experiments with wandb follow [this](https://docs.wandb.ai/quickstart) quickstart guide for more information

* ### Resuming from checkpoints

To resume a run from checkpoint:

```
python train.py experiment=<experiment_01> ++trainer.ckpt_path=/path/to/checkpoint
```

* ### Running

To run execute:

```
python train.py experiment=<experiment_01>
```

This will run the training according to the parameters described in ``configs/experiment/experiment_01.yaml``. If you want to overwrite specific settings you can do this from the command line too.

E.g. here we overwrite the batch_size as defined in the datamodule configuration file:

```
python train.py experiment=<experiment_01> ++datamodule.batch_size=64
```

* ### Single / Multi-GPU

Depending on the number of GPUs on your system you want to make changes to the trainer. If you are running on a single GPU system you can simply run the code using:

```
python train.py experiment=<experiment_01> trainer=default
```

If you are running on multiple GPU's or multiple node's make sure to change `config/trainer/ddp.yaml`. Change `gpus: 4` and `num_nodes: 1` to reflect your setup. Alternatively you can overwrite this on the command line. For example if you want to run on two nodes which each have 8 gpus (2x8 GPUs total):

```
python train.py experiment=<experiment_01> ++trainer.num_nodes=2 ++trainer.gpus=8
```

* ### Running on a slurm cluster
Example slurm submission scripts:
* `train.sbatch`
* `test.sbatch`

If you are running on a slurm cluster and submit many jobs with different parameters:
* `sweep_parameters.sh`

# License
![license][logo_license]

[logo_license]: figure/by-nc-nd.eu.png "License"
See LICENSE.MD. Upon publication of the paper we will switch to a CC-BY-NC-SA license.
