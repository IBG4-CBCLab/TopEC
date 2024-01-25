import ast
import logging
from builtins import bool
from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset


from src.datamodules.components.utils import count_cut, no_cut, radial_cut, random_cut

log = logging.getLogger(__name__)


class EnzymeDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "",
        binding_site_csv: str = "",
        num_atoms: int = 21,
        level: str = "residue",
        box_mode: str = "",
        useHierarchical: bool = True,
        cut_arg: int = 10,
        transform=None,
        pre_transform=None,
    ):

        """Binding site dataset using Pytorch geometric dataset.
        Data(pos, y, z) is read from a .pt file. Pos has coordinates for each atom in the binding site, 
        y is label of the binding site and z is the atom type.

        Args
        ----------

            root (str, optional):  Root directory where the dataset is saved. Defaults to "".

            data_dir (str):
                File path where entire dataset and labels.pt are saved, each entry in pt file
                    is dense numpy array of point cloud coordinates and corresponding features

            binding_site_csv (str): 
                CSV file containing all binding site information. 
        
            num_atoms (int): Number of atoms/residues in dataset 
        
            level (str): If 'residue' only "CA" will be used to represent the residue and if "atom"
                all atoms in given radius will be used
            
            box_mode (str): How to cut the pocket of enzyme. 'distance' will use all atoms within given distance 
                and 'count' will cut all atoms expanding from pocket center until the "cut_arg" is reached
            
            useHierarchical (bool): If True uses hierarchical EC commision number. Defaults to True.
            
            cut_arg (int): Argument to be used with box_mode it can refer to radius of pocket or number of atoms of pocket.
            
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.(default: :obj:`None`)

            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
        """

        super().__init__(None, transform, pre_transform)

        self.meta_data = pd.read_csv(binding_site_csv, sep=",")
        self.meta_data.centers = self.meta_data.centers.apply(ast.literal_eval)  # safety check to make sure the data type is a tupple for further processing

        if box_mode == "distance":
            self.cutter = radial_cut
        elif box_mode == "count":
            self.cutter = count_cut
        elif box_mode == "random":
            self.cutter = random_cut
        else:
            self.cutter = no_cut
            log.info("no region cutter selected, loading full enzyme")
        
        self.box_mode = box_mode
        self.data_dir = data_dir
        self.cut_arg = cut_arg
        self.num_atoms = num_atoms
        self.level = level
        self.useHierarchical = useHierarchical


        if self.useHierarchical:
            log.info("Using hierarchical class")
            log.info("Class Ratios")
            log.info(self.meta_data["hierarchical"].value_counts())
            self.labels = torch.load(self.data_dir + "/labels_hierarchical.pt")
            self.dictionary_map = pd.Series(
                self.meta_data["hierarchical"].values,
                index=self.meta_data["uniq_designation"],
            ).to_dict()
        else:
            log.info("Using main class")
            log.info("Class Ratios")
            log.info(self.meta_data["mainclass"].value_counts())
            self.labels = torch.load(self.data_dir + "/labels_main.pt")

    @property
    def raw_file_names(self):
        """Reads form dataframe the binding sites name that are going to be processed

        Returns:
            List: List of raw file names of which binding site is going to be processed.
        """

        return self.meta_data["enzyme_name"].astype(str).tolist()
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        """Reads form dataframe the binding sites name that are going to be processed

        Returns:
            List: List of raw file names of which binding site is going to be processed.
        """

        return self.meta_data["enzyme_name"].astype(str).tolist()
        # return ['data_1.pt', 'data_2.pt', ...]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        if self.level == "residue":
            enzyme_coords_features = torch.load(
                self.data_dir
                + "dataset_residue/"
                + self.processed_file_names[idx]
                + ".pt"
            )
        elif self.level == "hydrogen":
            enzyme_coords_features = torch.load(
                self.data_dir
                + "dataset_atom_hydrogens/"
                + self.processed_file_names[idx]
                + ".pt"
            )
        else:
            enzyme_coords_features = torch.load(
                self.data_dir
                + "dataset_atom/"
                + self.processed_file_names[idx]
                + ".pt"
            )


        if self.useHierarchical:
            label = self.dictionary_map[self.labels[self.processed_file_names[idx]]]
        else:
            label = self.labels[self.processed_file_names[idx]]

        center = self.meta_data[
            self.meta_data["enzyme_name"] == self.processed_file_names[idx]
        ]["centers"].to_numpy()[0]

        enzyme_name = self.raw_file_names[idx]

        binding_site_coords_features = self.cutter(
            center, enzyme_coords_features, self.cut_arg, enzyme_name
        )

        # if binding_site_coords_features.shape[0] < self.cut_arg or binding_site_coords_features.shape[0] > 1000:
        #     log.info(
        #         f"Issues with {enzyme_name} {self.cut_arg} with {binding_site_coords_features.shape[0]} atoms/residues in graph"
        #     )

        pos = torch.tensor(
            binding_site_coords_features[:, :3],
            requires_grad=False,
            dtype=torch.float32,
        )
        y = torch.tensor([label], requires_grad=False, dtype=torch.int64)
        x = torch.tensor(
            binding_site_coords_features[:, 3], requires_grad=False, dtype=torch.int64
        )
        x = F.one_hot(x, num_classes=self.num_atoms)
        x = x.float()
        
        # log.info(f"data shape {x.shape}")
        # log.info(f"data shape {y.shape}")

        data = Data(pos=pos, y=y, x=x, enzyme_name=enzyme_name)
        
        #if self.translate_num > 0.01:
        #    data = RandomTranslate(self.translate_num)(data)
        #data.edge_index = radius_graph(data.pos, r=self.cutoff)
        #data = RemoveIsolatedNodes()(data)

        return data