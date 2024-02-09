import logging
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GraphDataLoader
import torch_geometric.transforms as T
from src.datamodules.components.enzyme_dataset import EnzymeDataset

log = logging.getLogger(__name__)


class EnzymeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_file: str = "../../data/h5/all_enzymes/",
        ff19SB: str = "../preprocessing/ff19SB_map.csv",
        resolution: str = "residue",
        train_csv: str = "",
        val_csv: str = "",
        test_csv: str = "",
        num_atoms: int = 21,
        box_mode: str = "count",
        cut_arg: int = 10,
        cutoff: float = 10.0,
        oversample: bool = False,
        translate_num: float = 0.05,
        test_translate_num: float = 0.,
        batch_size: int = 64,
        num_workers: int = 3,
        pin_memory: bool = True,
        **kwargs,
    ):
        """Datamodule for graph models.

        Attributes:
        
            root_dir: Root directory where the dataset is saved. Defaults to "".
            data_dir (str, optional): File path where entire dataset and labels.pt are saved, each entry in pt file is dense numpy array of point cloud coordinates and corresponding features. Defaults to "../../data/processed/".
            train_csv, val_csv, test_csv (str, optional): csv files to use. Defaults to "".
            num_atoms (int): Number of atoms/residues in dataset 
            level (str): If 'residue' only "CA" will be used to represent the residue and if "atom" all atoms in given radius will be used
            box_mode (str): How to cut the pocket of enzyme. 'distance' will use all atoms within given distance and 'count' will cut all atoms expanding from pocket center until the "cut_arg" is reached
            useHierarchical (bool): If True uses hierarchical EC commision number. Defaults to True.
            cut_arg (int): Argument to be used with box_mode it can refer to radius of pocket or number of atoms of pocket.
            cutoff (float): Radius neighbourhood of a node. It is used to construct radius graph.
            oversample (bool): Calculate weights per class. Overrepresented classes get lower weights
            translate_num (float, optional): Augmentation for graph data. Each node in graph is randomly translated in x, y, z direction by amount between 0,translate_num            
            test_translate_num (float, optional): Augmentation for "TEST" graph data. Each node in graph is randomly translated in x, y, z direction by amount between 0,translate_num            
            batch_size (int, optional): Defaults to 64.
            num_workers (int, optional): Dataloader parameter. Defaults to 3.
 
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ff19SB = ff19SB
        self.resolution = resolution
        self.h5_file = h5_file

        self.translate_num = translate_num
        self.test_translate_num = test_translate_num
        self.cut_arg = cut_arg
        self.cutoff = cutoff
        self.box_mode = box_mode
            
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        train_class_data = pd.read_csv(self.train_csv, sep=",")
        
        nSamples = (train_class_data.uniq_designation.value_counts().sort_index().tolist())
        
        if oversample:
            self.normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        else:
            self.normedWeights = [1 for x in nSamples]

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = EnzymeDataset(
            h5_file = self.h5_file,
            resolution = self.resolution,
            ff19SB = self.ff19SB,
            binding_site_csv = self.train_csv,
            box_mode = self.box_mode,
            cut_arg = self.cut_arg,
            transform = T.Compose([T.RandomTranslate(self.translate_num),T.RadiusGraph(self.cutoff),T.RemoveIsolatedNodes()]) 
        )
        self.val_dataset = EnzymeDataset(
            h5_file = self.h5_file,
            resolution = self.resolution,
            ff19SB = self.ff19SB,
            binding_site_csv = self.val_csv,
            box_mode = self.box_mode,
            cut_arg = self.cut_arg,
            transform = T.Compose([T.RadiusGraph(self.cutoff),T.RemoveIsolatedNodes()]) 
        )
        self.test_dataset = EnzymeDataset(
            h5_file = self.h5_file,
            resolution = self.resolution,
            ff19SB = self.ff19SB,
            binding_site_csv = self.test_csv,
            box_mode = self.box_mode,
            cut_arg = self.cut_arg,
            transform = T.Compose([T.RadiusGraph(self.cutoff),T.RemoveIsolatedNodes()]) 
        )
        
        self.augmented_test_dataset = EnzymeDataset(
            h5_file = self.h5_file,
            resolution = self.resolution,
            ff19SB = self.ff19SB,
            binding_site_csv = self.test_csv,
            box_mode = self.box_mode,
            cut_arg = self.cut_arg,
            transform = T.Compose([T.RandomTranslate(self.test_translate_num),T.RadiusGraph(self.cutoff),T.RemoveIsolatedNodes()]) 
        )

        log.info(
            f"Binding Site Cutting with : {self.box_mode} cut_arg: {self.cut_arg} with {self.resolution}"
        )
        log.info("====================")
        
        log.info(f"Train Dataset:{self.train_csv}")
        log.info("------")
        log.info(f"Number of graphs: {len(self.train_dataset)}")
        log.info("====================")
        
        log.info(f"Validation Dataset:{self.val_csv}")
        log.info("------")
        log.info(f"Number of graphs: {len(self.val_dataset)}")
        log.info("====================")
        
        log.info(f"Test Dataset:{self.test_csv}")
        log.info("------")
        log.info(f"Number of graphs: {len(self.test_dataset)}")
        log.info("====================")
        
    def train_dataloader(self):
        return GraphDataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            
        )

    def val_dataloader(self):
        val_loader = GraphDataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

        return val_loader

    def test_dataloader(self):
        
        if self.test_translate_num > 0:
            
            test_loader = GraphDataLoader(
                self.augmented_test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            
            test_loader = GraphDataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

        return test_loader

    def predict_dataloader(self):

        val_loader = GraphDataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        if self.test_translate_num > 0:
            
            test_loader = GraphDataLoader(
                self.augmented_test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            
            test_loader = GraphDataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

        loaders = [val_loader, test_loader]

        return loaders