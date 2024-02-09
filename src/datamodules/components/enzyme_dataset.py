import ast
import logging
from builtins import bool
from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
import h5py
import numpy as np

from src.datamodules.components.utils import count_cut, no_cut, radial_cut, random_cut

log = logging.getLogger(__name__)

res_map = {'ALA':0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19,'UNK':20}
res1int = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}


class EnzymeDataset(Dataset):
    def __init__(
        self,
        h5_file: str = "",
        resolution: str = "residue",
        ff19SB: str = "",
        binding_site_csv: str = "",
        box_mode: str = "",
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

            binding_site_csv (str): 
                CSV file containing all binding site information. 
                           
            box_mode (str): How to cut the pocket of enzyme. 'distance' will use all atoms within given distance 
                and 'count' will cut all atoms expanding from pocket center until the "cut_arg" is reached
            
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
        
        self.h5_file = h5_file
        self.resolution = resolution

        self.ff19SB = ff19SB
        self.atom_map = pd.read_csv(self.ff19SB, index_col='TYPE')

        self.box_mode = box_mode
        self.cut_arg = cut_arg

        self.labels = pd.Series(self.meta_data['mainclass'].values - 1, index=self.meta_data['enzyme_name']).to_dict()
        self.centers = pd.Series(self.meta_data['centers'].values, index=self.meta_data['enzyme_name']).to_dict()


        # if self.useHierarchical:
        #     log.info("Using hierarchical class")
        #     log.info("Class Ratios")
        #     log.info(self.meta_data["hierarchical"].value_counts())
        #     self.labels = torch.load(self.data_dir + "/labels_hierarchical.pt")
        #     self.dictionary_map = pd.Series(
        #         self.meta_data["hierarchical"].values,
        #         index=self.meta_data["uniq_designation"],
        #     ).to_dict()
        # else:
        #     log.info("Using main class")
        #     log.info("Class Ratios")
        #     log.info(self.meta_data["mainclass"].value_counts())
        #     self.labels = torch.load(self.data_dir + "/labels_main.pt")

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

    def open_hdf5(self, h5file, identifier):
        with h5py.File(h5file, 'r') as file:
            for chain in file[f'{identifier}']['structure']['0'].keys():
                amino_types = file[f'{identifier}']['structure']['0'][f'{chain}']['residues']['seq1'][()]
                atom_amino_id = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['atom_amino_id'][()] #size: (n_atom,)
                atom_pos = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['xyz'][()]
                atom_names = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['type'][()].astype('U13') #decodes b'S'

        if self.resolution == 'residue':
            try:
                ca_indices = atom_amino_id[np.where(atom_names == 'CA')] - 1
                ca_pos = atom_pos[atom_names == 'CA']
                ca_pos = ca_pos.reshape(-1,3) 
                mapped_integers = np.array([res1int[char] for enum, char in enumerate(amino_types.decode('utf-8')) if enum in ca_indices])
                if len(mapped_integers) != len(ca_indices):
                    features = np.column_stack((ca_pos, mapped_integers[ca_indices]))
                else:
                    features = np.column_stack((ca_pos, mapped_integers))
            except (IndexError, ValueError) as e:
                print(identifier)
            
            # print('shapes:', ca_pos.shape, mapped_integers.shape)
            # print('lengths:', len(amino_types.decode('utf-8')), len(np.unique(atom_amino_id)))
            # features = np.column_stack((ca_pos, mapped_integers))
            
        if self.resolution == 'atom':
            atom_types = []
            for num in np.unique(atom_amino_id):
                #find indices per AA to find atoms
                indices = np.where(atom_amino_id == num)
                atoms = atom_names.astype('U13')[indices]
                #obtain AA name from numerical pos
                aacid = amino_types.decode('utf-8')[num-1]
                #map atom nums from ff19SB
                atom_types.extend(self.atom_map[f'{aacid}'].loc[atoms].tolist())

            assert len(atom_pos) == len(atom_types)
            features = np.column_stack((atom_pos, atom_types))

        return features


    def get(self, idx):
        identifier = self.processed_file_names[idx]

        enzyme_coords_features = self.open_hdf5(self.h5_file, identifier)

        label = self.labels[identifier]

        center = self.centers[identifier]

        binding_site_coords_features = self.cutter(
            center, enzyme_coords_features, self.cut_arg, identifier
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
        if self.resolution == 'atom':
            x = F.one_hot(x, num_classes=31)
        elif self.resolution == 'residue':
            x = F.one_hot(x, num_classes=21)
        x = x.float()
        
        # log.info(f"data shape {x.shape}")
        # log.info(f"data shape {y.shape}")

        data = Data(pos=pos, y=y, x=x, enzyme_name=identifier)
        
        #if self.translate_num > 0.01:
        #    data = RandomTranslate(self.translate_num)(data)
        #data.edge_index = radius_graph(data.pos, r=self.cutoff)
        #data = RemoveIsolatedNodes()(data)

        return data