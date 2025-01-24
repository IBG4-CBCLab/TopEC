# TopEC: Enzyme function prediction from enzyme (pocket) structure

[TopEC: Improved classification of enzyme function by a localized 3D protein descriptor and 3D Graph Neural Networks](https://www.biorxiv.org/content/10.1101/2024.01.31.578271v1)

TopEC is an enzyme function prediction tool which uses graph neural networks to predict the enzyme class according to [International Union of Biochemistry and Molecular Biology (IUBMB)](https://doi.org/10.1093/nar/gkn582) nomenclature. 

This work is created in cooperation with the [HelmholtzAI consultants @ Helmholtz Munich](https://www.helmholtz.ai/themenmenue/our-research/consultant-teams/helmholtz-ai-consultants-helmholtz-munich/index.html). A big thanks to Marie Piraud and Erinc Merdivan for helping us realize this project. 

Using TopEC you can predict enzyme function from different representations of proteins. We offer three methods to design the graph input at atomic and residue resolution. See the usage section for more details.

![alt text][logo]

[logo]: figure/method_overview_GH_version.png "Method overview"

# Table of Contents
- [General Information](#general-information)
- [Requirements](#requirements)
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

# Requirements

This configuration is tested on compute nodes within the [JUWELS-Booster supercomputer](https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html#hardware-configuration-of-the-system-name-booster-module) and [JURECA supercomputer](https://apps.fz-juelich.de/jsc/hps/jureca/index.html). Installing this on a cluster or server where you have access to a GPU with atleast 40Gb VRAM (Nvidia A100 or newer models) is highly recommended. 
The setup is tested with python3.9 and pytorch2.1 using CUDA12.1

All required software can be installed with pip (see [Installation](#installation)). Alternatively you can manually install the following packages:

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
* [pycm](https://www.pycm.io/)


# Installation

### python virtual environment
As the code is tested with python3.9 it is recommended to create your python venv on 3.9.

```
python3.9 -m venv topec_venv

pip install -r requirements.txt

source topec_venv/bin/activate

```

### Dataset creation

Obtain the PDB files [here](https://fz-juelich.sciebo.de/s/7cOPiXC0iqlh3c9).

Make sure the paths in ``configs/create_dataset.yaml`` are pointing towards the folder you store the pdb structures.
Then execute from command line:

```
python create_h5dataset.py
```

Using a compute node with 48 cores can do this in roughly 5 hours. Using a single core the dataset creation can take up to a day. The dataset creation code takes into account experimental and computationally generated structures. 

**Alternatively you can obtain the H5 file [here](https://fz-juelich.sciebo.de/s/zvnTIm0TdJmPwdd) and skip the creation of the h5 file**

If you want more flexibility over the dataset creation for your own experiments you can alter the create_h5dataset.yaml in the configs and run:
```
python run_dataset_create.py
```

# Usage

We use hydra to parse configuration files in the ``configs/`` folder. Generally you do not need to make any changes except to check your paths are set correctly. We note down the most important configuration files you might want to change if you want to run your own experiments. For a detailed explanation to work with configuration files see [here](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/). 

```
configs/
├── callbacks           <- Controls the early stopping and metrics configuration.
├── datamodule          <- Contains a configuration file for every dataset we trained and tested. 
├── experiment          <- Contains an experiment file for each setup we trained and tested.
├── model               <- Contains the configuration settings for each model
└── trainer             <- Contains configuration for the trainer.
create_dataset.yaml
test.yaml
train.yaml
```

### Tracking Experiments

For tracking your experiments with wandb follow [this](https://docs.wandb.ai/quickstart) quickstart guide for more information.
By default the runs are stored locally in the `./wandb` folder.

### Resuming from checkpoints

To resume a run from checkpoint:

```
python train.py experiment=<experiment_01> ++trainer.ckpt_path=/path/to/checkpoint
```

### Binding site prediction

To train TopEC we need to know the locations of the binding site. TopEC expects as input a tuple ``(x,y,z)`` for the binding locations. 

We added an example script to generate binding sites with p2rank. Alternatively you can use any binding site prediction tool.
First obtain p2rank [here](https://github.com/rdk/p2rank) and follow the instructions for predicting binding sites.

We recommend creating a single text file with a list of PDB file locations e.g. `proteins.txt`
Then run P2Rank with:
```
prank predict -threads 8 -o ./output_folder -c alphafold proteins.txt
```

This will create the output folder with a file for each protein. To merge the results run:
```
python concatenate_binding_sites.py -f /path/to/folder -o output.csv
```
This will create a generated CSV with identifier, binding center tuple and rank of the predicted binding site: ``identifier | (x, y, z) | rank``.




### Training

To run execute:

```
python train.py experiment=<experiment_01>
```

This will run the training according to the parameters described in ``configs/experiment/experiment_01.yaml``. A log folder will be generated under ``logs/`` containing the network checkpoints.

If you want to overwrite specific settings you can do this from the command line too.

E.g. here we overwrite the batch_size as defined in the datamodule configuration file:

```
python train.py experiment=<experiment_01> ++datamodule.batch_size=64
```

Using a JUWELS-BOOSTER compute node with 4x A100 (40GB) we can perform a single training epoch on roughly 

### Single / Multi-GPU

Depending on the number of GPUs on your system you want to make changes to the trainer. If you are running on a single GPU system you can simply run the code using:

```
python train.py experiment=<experiment_01> trainer=default
```

If you are running on multiple GPU's or multiple node's make sure to change `config/trainer/ddp.yaml`. Change `gpus: 4` and `num_nodes: 1` to reflect your setup. Alternatively you can overwrite this on the command line. For example if you want to run on two nodes which each have 8 gpus (2x8 GPUs total):

```
python train.py experiment=<experiment_01> ++trainer.num_nodes=2 ++trainer.gpus=8
```

### Running on a slurm cluster
Example slurm submission scripts:
* `train.sbatch`
* `test.sbatch`

If you are running on a slurm cluster and submit many jobs with different parameters:
* `sweep_parameters.sh`

### Testing

To test the network execute the following:

```
python test.py trainer=test experiment=<experiment_file> ckpt_path=/path/to/checkpoint
```

This will generate the evaluation reports under ``logs/evaluations`` containing a PyCM report for the test and validation sets. Furthermore, we automatically generate pr curves and realibility diagrams for the tested network.

### Expanding TopEC to other classification problems.

As we treat enzyme function prediction as a classification problem, we can use the TopEC framework to develop novel classification tools.
Currently we read in the enzyme classes from the CSV files in ``./data/csv/``. For record keeping these files contain more information than strictly necessary to train deep learning models with TopEC. 

In the core, TopEC expects three columns to be present. The first is `enzyme_name` which refers to the name of the data object in the h5. Usually this is the UniProtAC or PDB identifier. Secondly we need to know the predicted binding site center, in the column `centers` we put the binding site center as a tuple of X, Y, Z coordinates. The last column needed is called `hierarchical`. Here we put a separate class for enzyme hierarchy, e.g. EC: 1.1.1.1 is class 0, EC: 1.1.1.100 is class 1. One could change this to, for example, a encoding of cell locations, to train a classifier for where in the cell a protein would be present.

### Using different graph models within the TopEC framework

Using TopEC its quite simple using different graph models. An example of a new graph model is shown in `src/models/components/ginnet.py`. Here we have an implementation of GINNet, a graph neural network for graph interactions. Adding a new model is as simple as creating a new class based on the `torch.nn.Module` module. The class only needs to contain a `forward(x, pos, edge_index, batch)` function taking the input of the graph embedding, node positions, possible edge indexes and the batch argument. Then we can add any tunable layer in the forward function, as long as we return the resulting graph embedding and output of the network.


# License
CC-BY-NC-SA
See LICENSE.MD.
