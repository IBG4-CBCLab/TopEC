import ast
import logging
import pandas as pd
import h5py
import numpy as np
from collections import Counter

# from builtins import bool
# from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

from src.datamodules.components.utils import count_cut, no_cut, radial_cut, random_cut

log = logging.getLogger(__name__)

res_map = {'ALA':0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19,'UNK':20}
res1int = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20, 'B':20, 'U':20}


class EnzymeDataset(Dataset):
    def __init__(
        self,
        h5_file: str = "",
        resolution: str = "residue",
        ff19SB: str = "",
        binding_site_csv: str = "",
        box_mode: str = "",
        cut_arg: int = 100,
        transform=None,
        useHierarchical: bool = True,
        pre_transform=None,
    ):

        """Binding site dataset using Pytorch geometric dataset.
        Data(pos, y, z) is read from a .h5 file. Pos has coordinates for each atom in the binding site, 
        y is label of the binding site and z is the atom type.

        Args
        ----------
        h5_file (str): Path to the HDF5 file containing enzyme structures.
        resolution (str): Resolution of the dataset, either 'residue' or 'atom'.
        ff19SB (str): Path to the CSV file containing atom mapping information.
        binding_site_csv (str): Path to the CSV file containing binding site information.
        box_mode (str): How to cut the pocket of enzyme. 'distance' will use all atoms within given distance 
            and 'count' will cut all atoms expanding from pocket center until the "cut_arg" is reached.
        cut_arg (int): Argument to be used with box_mode, referring to radius of pocket or number of atoms of pocket.
        transform (callable, optional): A function/transform that takes in a torch_geometric.data.Data object and returns 
            a transformed version.
        useHierarchical (bool, optional): Indicates whether to use hierarchical classification.
        pre_transform (callable, optional): A function/transform that takes in a torch_geometric.data.Data object and returns 
            a transformed version.
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

        self.useHierarchical = useHierarchical
        self.centers = pd.Series(self.meta_data['centers'].values, index=self.meta_data['enzyme_name']).to_dict()

        if self.useHierarchical:
            log.info("Using hierarchical class")
            log.info("Class Ratios")
            log.info(self.meta_data["hierarchical"].value_counts())
            self.labels = pd.Series(self.meta_data["hierarchical"].values, index=self.meta_data['enzyme_name']).to_dict()
        else:
            log.info("Using main class")
            log.info("Class Ratios")
            log.info(self.meta_data["mainclass"].value_counts())
            self.labels = pd.Series(self.meta_data['mainclass'].values - 1, index=self.meta_data['enzyme_name']).to_dict()

    @property
    def raw_file_names(self):
        """Reads form dataframe the binding sites name that are going to be processed

        Returns:
            List: List of raw file names of which binding site is going to be processed.
        """

        return self.meta_data["enzyme_name"].astype(str).tolist()

    @property
    def processed_file_names(self):
        """Reads form dataframe the binding sites name that are going to be processed

        Returns:
            List: List of raw file names of which binding site is going to be processed.
        """

        return self.meta_data["enzyme_name"].astype(str).tolist()

    def len(self):
        """
        Returns the length of the dataset.

        Returns:
        int: Length of the dataset.
        """
        return len(self.processed_file_names)

    def open_hdf5(self, h5file, identifier):
        """
        Opens the HDF5 file and extracts features for a given identifier.
        
        Args:
        h5file: HDF5 file containing enzyme structures.
        identifier: Identifier for the enzyme.

        Returns:
        np.ndarray: Features of the enzyme.
        """
        with h5py.File(h5file, 'r') as file:
            #loop over chains
            amino_types = []
            atom_amino_id = []
            atom_pos = []
            atom_names = []
            for chain in file[f'{identifier}']['structure']['0'].keys():
                amino_types.extend(file[f'{identifier}']['structure']['0'][f'{chain}']['residues']['seq1'][()].decode('utf-8'))
                atom_amino_id.extend(file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['atom_amino_id'][()]) #size: (n_atom,)
                atom_pos.extend(file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['xyz'][()])
                atom_names.extend(file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['type'][()].astype('U13')) #decodes b'S'
        if self.resolution == 'residue':
            try:
                #remove any disallowed atoms (e.g. DNA, RNA, etc.)
                allowed_atoms = self.atom_map.index.values
                mask_allowed_atoms = np.isin(atom_names, allowed_atoms)
                atom_amino_id = np.array(atom_amino_id)[mask_allowed_atoms]
                atom_pos = np.array(atom_pos)[mask_allowed_atoms]
                atom_names = np.array(atom_names)[mask_allowed_atoms]

                #Create counters for each amino acid and convert to long form
                counts = Counter(atom_amino_id)
                sorted_counts = dict(sorted(counts.items()))
                long_amino_types = ''.join([amino_types[enum] * count for enum, count in enumerate(sorted_counts.values())])

                # for enum, key in enumerate(sorted_counts.keys()):
                #     long_amino_types += amino_types[enum] * sorted_counts[key]

                # assert len(long_amino_types) == len(atom_amino_id)
                long_amino_types = np.array(list(long_amino_types))

                ca_positions = atom_pos[np.array(atom_names) == 'CA']
                ca_amino_ids = long_amino_types[np.array(atom_names) == 'CA']
                ca_amino_ids = np.array([res1int[char] for char in ca_amino_ids])
                features = np.column_stack((ca_positions, ca_amino_ids))
            except (IndexError, ValueError, KeyError) as e:
                log.info(f"Error with {identifier}")
            
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
        """
        Gets the data for a given index.
        
        Args:
        idx: Index of the data.

        Returns:
        torch_geometric.data.Data: Data for the given index.
        """
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
        
        # if self.translate_num > 0.01:
        #    data = RandomTranslate(self.translate_num)(data)
        # data.edge_index = radius_graph(data.pos, r=self.cutoff)
        # data = RemoveIsolatedNodes()(data)

        return data