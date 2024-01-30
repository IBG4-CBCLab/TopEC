# TopEC: Enzyme function prediction from enzyme (pocket) structure

[Associated Paper](https://www.google.com)

TopEC is an enzyme function prediction tool which uses graph neural networks to predict the enzyme class according to [International Union of Biochemistry and Molecular Biology (IUBMB)](https://doi.org/10.1093/nar/gkn582) nomenclature. 

This work is created in cooperation with the HelmholtzAI consultants @ Helmholtz Munich. A big thanks to Marie Piraud and Erinc Merdivan for helping us realize this project. 

Using TopEC you can predict enzyme function from different representations of proteins. See the usage section for more details.

![alt text][logo]

[logo]: figure/method_overview_GH_version.png "Method overview"

# Table of Contents
- [General Information][#general-information]
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

# General Information

Pytorch Repo for enzyme classification using binding site information for the Juelich Enzyme Prediction Voucher
Uses pytorch, pytorch-lightning, pytorch-geometric, hydra and tensorboard

Task is to classify enzymes according to EC (https://en.wikipedia.org/wiki/Enzyme_Commission_number) using only binding site information instead of full enzyme. 

As input we use the protein structure (.pdb files) and binding site location coordinates. Around these binding site coordinates we construct our localized 3D descriptor for use in the graph neural networks. 

We show the two implemented networks SchNet and DimeNet++ on two approaches:

- 1) The residue based approach. Here we construct a graph only for the C_{alpha} positions the protein's amino acid.

- 2) The atom based approach. Here we construct graphs for each atom in the protein such that the network learns from a full atomistic view. 


# Installation
Install pytorch-geometric following instructions below:

https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
Install

Run
```
$ python -m pip install -e .
```
# Usage

# License


