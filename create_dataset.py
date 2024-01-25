import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src import utils
import multiprocessing
from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import os
import numpy as np
import torch
log = logging.getLogger(__name__)


def create_pdb_datasets( 
    input_path: str = "", 
    enzyme_csv: str = "",
    output_path: str = "", 
    weight_path: str = "",
    ):

    """ 
    pdb files for each enzyme binding sites are read. After for each binding site coordinate information and atom-type is saved as .pt file. 
    Dataset is separated temporally to train, val and test set.
    Also Train, Val, Test csv files are saved to given path.
    PNG file with class distribution is also saved to given path.

    Args:
        input_path (str, optional): Folder where all PDBs are.
        enzyme_csv (str, optional): CSV file with all information about enzymes.
        output_path (str, optional): Processed enzyme files will be saved in this directory as ".pt" file. 
        weight_path (str, optional): Chemical Weights File. Defaults to "".
    """

    parser = PDBParser(QUIET=True)
    res_map = {'ALA':0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19,'UNK':20}

    log.info(f"Reading All Enzyme Information from CSV file <{enzyme_csv}>")
    all_data = pd.read_csv(enzyme_csv, sep=",")

    log.info(f"Number of Files in CSV: <{all_data.shape[0]}>")

    log.info(f"Processing enzymes from <{input_path}>")

    if not os.path.exists(output_path):
        log.info(f"Cound not find folder at -- <{output_path}>")
        log.info(f"Creating Folder at -- <{output_path}>")
        os.makedirs(output_path)

    if not os.path.exists(output_path+'dataset_residue/'):
        log.info(f"Cound not find folder at -- <{output_path+'dataset_residue/'}>")
        log.info(f"Creating Folder at -- <{output_path+'dataset_residue/'}>")
        os.makedirs(output_path+'dataset_residue/')
    
    if not os.path.exists(output_path+'dataset_atom/'):
        log.info(f"Cound not find folder at -- <{output_path+'dataset_atom/'}>")
        log.info(f"Creating Folder at -- <{output_path+'dataset_atom/'}>")
        os.makedirs(output_path+'dataset_atom/')

    if not os.path.exists(output_path+'dataset_atom_hydrogens/'):
        log.info(f"Cound not find folder at -- <{output_path+'dataset_atom_hydrogens/'}>")
        log.info(f"Creating Folder at -- <{output_path+'dataset_atom_hydrogens/'}>")
        os.makedirs(output_path+'dataset_atom_hydrogens/')

    # load in chemical weights table for atom type
    df_weight = pd.read_csv(weight_path)

    #create parameter list for pool.starmap
    items = [(idx, all_data, parser, res_map, df_weight, input_path, output_path) for idx in range(all_data.shape[0])]

    #multiprocesses the atom parsing with maximum available cpu's. Result is written to a list and later converted to a dictionary
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        label_list = pool.starmap(parse_atoms, items)
     
    label_array = np.array(label_list)

    label_arrays_main = {}
    label_arrays_hierarchical = {}

    for row in label_array:
        label_arrays_main[row[0][0]] = row[1]
        label_arrays_hierarchical[row[0][0]] = row[2]    

    log.info(f"mainclass labels: {label_arrays_main}")
    log.info(f"hierarchical labels: {label_arrays_hierarchical}")

    torch.save(label_arrays_main, output_path + "/labels_main.pt")
    torch.save(label_arrays_hierarchical, output_path + "/labels_hierarchical.pt")

    log.info(f"Saved data to -- <{output_path}>")
   
    num_unique_enzyme = len(all_data["enzyme_name"].unique())
    log.info(f"Total of <{num_unique_enzyme}> unique enzymes and  are found at the Dataset")
    log.info(f"Total of <{num_unique_enzyme}> labelled enzymes are found at the Dataset")

def parse_atoms(idx, all_data, parser, res_map, df_weight, input_path, output_path):

    if idx % 500 == 0:
        log.info(f"Processed <{idx}> enzymes from <{input_path}>")

    file_name = input_path + all_data.iloc[idx]['enzyme_name'] + ".pdb"
    log.info(f"PDB file processing at -- <{file_name}>")
    structure = parser.get_structure(all_data.iloc[idx]['enzyme_name'], file_name)

    features_res = []
    features_atom = []
    features_atom_hydrogens = []

    for model in structure:
        for chain in model:
            for residue in chain:

                #remove UNK residues
                if residue.get_resname() not in df_weight.columns:
                    log.info(f"PDB file at -- <{file_name}> couldn't process -- <{residue.get_resname()}>")
                    continue

                for atom in residue:
                    #if node_type == 'res':
                        atom_id = atom.get_id()

                        #remove DNA atoms
                        if atom_id not in df_weight['TYPE'].unique():
                            log.info(f"PDB file at -- <{file_name}> couldn't process -- <{atom_id}>")
                            continue

                        x, y, z = atom.get_coord()
                        if atom_id == 'CA':
                            if residue.get_resname() in res_map:
                                features_res.append([x, y, z, res_map[residue.get_resname()]])
                            else:
                                features_res.append([x, y, z, 20.0])

                        chemical_weight = df_weight[df_weight['TYPE'] == atom_id][residue.get_resname()].to_numpy()[0]

                        if 'H' not in atom_id:
                            if residue.get_resname() in res_map:
                                features_atom.append([x, y, z, chemical_weight])
                            else:
                                features_atom.append([x, y, z, 30.0])

                        if residue.get_resname() in res_map:
                            features_atom_hydrogens.append([x, y, z, chemical_weight])
                        else:
                            features_atom_hydrogens.append([x, y, z, 50.0])


    data_array_res = np.array(features_res)
    data_array_atom = np.array(features_atom)
    data_array_atom_hydrogens = np.array(features_atom_hydrogens)

    torch.save(data_array_res, output_path +"dataset_residue/"+ all_data.iloc[idx]['enzyme_name'] + ".pt")
    torch.save(data_array_atom, output_path +"dataset_atom/"+ all_data.iloc[idx]['enzyme_name'] + ".pt")
    torch.save(data_array_atom_hydrogens, output_path +"dataset_atom_hydrogens/"+ all_data.iloc[idx]['enzyme_name'] + ".pt")

    return [[all_data.iloc[idx]['enzyme_name']], int(all_data.iloc[idx].mainclass) - 1, int(all_data.iloc[idx].uniq_designation)]


@hydra.main(config_path="configs/", config_name="create_dataset.yaml")
def main(config: DictConfig):


    # Applies optional utilities
    utils.extras(config)

    return create_pdb_datasets(input_path=config.input_path, 
    enzyme_csv=config.enzyme_csv,
    output_path=config.output_path, 
    weight_path=config.weight_path)


if __name__ == "__main__":
    main()
    
    
    
    
