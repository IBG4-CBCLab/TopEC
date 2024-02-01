import h5py
import pandas as pd
import numpy as np
import logging
import glob
from multiprocessing import Pool
import os

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

### Logging ###
'''
Initiate a logging method. Likely can remove instantiating and connect to full experiment log.
Or we can keep it seperate for handling the data
'''
logging.basicConfig(filename='create_dataset.log', level=logging.DEBUG)
log = logging.getLogger(__name__) #for connecting to the logger

### Structure MAPPINGS ###
'''
Structure mapping represent named ligands from the PDB.
This allows us to process PDB information and seperate on biochemical specificity.
'''
 
ion_map = ['0BE', '3CO', '3NI', '4MO', '4PU', '4TI', '6MO', 'AG', 'AL', 'AM', 
        'AR', 'ARS', 'AU', 'AU3', 'BA', 'BR', 'BRO', 'BS3', 'CA', 'CD', 
        'CE', 'CF', 'CL', 'CLO', 'CO', 'CR', 'CS', 'CU', 'CU1', 'CU3', 
        'D8U', 'DUM', 'DY', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2', 
        'FLO', 'GA', 'GD', 'GD3', 'H', 'HG', 'HO', 'HO3', 'IDO', 'IN', 
        'IOD', 'IR', 'IR3', 'K', 'KR', 'LA', 'LI', 'LU', 'MG', 'MN', 
        'MN3', 'MO', 'NA', 'ND', 'NGN', 'NI', 'O', 'OS', 'OS4', 'OX',
        'OXO', 'PB', 'PD', 'PR', 'PT', 'PT4', 'QTR', 'RB', 'RE', 'RH', 
        'RH3', 'RHF', 'RU', 'S', 'SB', 'SE', 'SM', 'SR', 'TA0', 'TB', 
        'TE', 'TH', 'TL', 'U1', 'UNX', 'V', 'W', 'XE', 'Y1', 'YB', 
        'YB2', 'YT3', 'ZCM', 'ZN', 'ZN2', 'ZR']

dna_map = ['DC', 'DG', 'DA', 'DT', 'DU', 'DI', '8Y9', 'EXC', 'F3H', 'F4Q', 'F6U',
       'F6X', 'F73', 'F74', 'F7H', 'F7K', 'F7O', 'F7R', 'F7X', 'J0X', 'J4T',
       'JSP', 'MFO', 'OKN', 'OKQ', 'OKT', 'PDU', 'QCK', 'S6M', 'T0P', 'T64'
       'TCJ', 'TTI', 'U48', 'U7B', 'WC7', '8YN', 'AWC', 'C7R', 'C7S', 'DP',
       'DZ', 'EAN', 'EW3', 'EWC', '2FE', '2FI', '4U3', '6FK', '8PY', 'Y']

rna_map = ['C', 'G', 'A', 'T', 'U', 'I', '16B', '3AU', '50L', '56B', '7OK',
       '7SN', '8AH', '9V9', 'A6A', 'A6C', 'A6G', 'A6U', 'B8H', 'B8K', 'B8Q',
       'B8T', 'B8W', 'B9B', 'B9H', 'CNU', 'E3C', 'E6G', 'E7G', 'GMX', 'I4U',
       'JMH', 'LHH', 'LV2', 'M7A', 'MHG', 'MMX', 'N7X', 'O2Z', 'OYW', 'P4U',
       'P7G', 'QSK', 'QSQ', 'RFJ', 'TC', 'TG', 'U23', 'U4M', 'UY1', 'UY4',
       'VC7', 'XNY', 'YA4']

#implement modified AA. Current problem is that they have same identifiers for modified AA or as free ligand
#current implementation is GAPS for modified AAs.
modified_aa = []

res2single = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V','UNK':'X'}
res2int = {b'ALA':0, b'ARG':1, b'ASN':2, b'ASP':3, b'CYS':4, b'GLN':5, b'GLU':6, b'GLY':7, b'HIS':8, b'ILE':9, b'LEU':10, b'LYS':11, b'MET':12, b'PHE':13, b'PRO':14, b'SER':15, b'THR':16, b'TRP':17, b'TYR':18, b'VAL':19, b'UNK':20}
### Structure MAPPINGS ###

class HDF5File():   
    '''
    Base class for the HDF5 File. Add in basic functionality here and inherit it for each modality class.
    '''
    def print_attrs(name, obj):
        '''
        Helper function for going over the HDF5 tree.
        '''
        log.info(f'{name}')
        print(f'{name}')
        for key, val in obj.attrs.items():
            log.info(f'{key}:{val}')
            print(f'{key}:{val}')
        return

    def print_full_h5_tree(self):
        '''
        Load in h5py file and prints out full tree to log.
        Warning: Will get very long for the full dataset. 
        Might have to incorporate filtering keys for smaller trees. 
        Other possibility is to write to something JSON-like so you can collapse the tree in JSON viewers.
        '''
        with h5py.File(self.h5_file, 'r') as file:
            file.visititems(print_attrs)
        return

    def remove_id(h5_file, identifier):
        '''
        Removes specific record from the HDF5 File.
        '''
        with h5py.File(self.h5_file, 'a') as file:
            del file[f'{identifier}']
        return

class Structure2HDF5(HDF5File):
    '''
    Base class for adding structural and sequence (from struc) data to the HDF5 file.
    '''
    def __init__(self,
                 h5_file, #path to H5File
                 pdbroot, #folder containing the stored PDBs
                 file_type, #Either 'PDB' or 'MMCIF'
                 experimental=False, #True for PDB structures, False for computed structures. False is faster processing.
                 warnings=True, #False / True to turn on PDB/MMCIFParser warnings. 
    ):

        self.h5_file = h5_file
        self.pdbroot = pdbroot
        self.file_type = file_type
        self.warnings = warnings
        self.experimental = experimental

        if not os.path.isfile(h5_file):
            with h5py.File(self.h5_file, 'w') as file:
                file.close()

        #implement metadata loading for global properties
        # self.metadata = metadata_file
        
        #Determine if we are parsing PDB or MMCIF structures.
        if self.file_type == 'PDB':
            self.parser = PDBParser(QUIET=not warnings)
        elif self.file_type == 'MMCIF':
            self.parser = MMCIFParser(QUIET=not warnings)
        else:
            raise ValueError('Please select either "PDB" or "MMCIF" as strucural input.')

    def structure_topology(self, identifier, structure_file):
        '''
        Reads in the structure PDB / MMCIF file and parses the data into a dictionary object to store within the h5 file.

        identifier: protein name
        structure_file: location of the structure file
        experimental: If we use computational or experimental structures. For computational structures we are usually only intersted in the polypeptide chain
        '''
        #open parser
        structure = self.parser.get_structure(identifier, structure_file)
        #each structure can have multiple models (e.g. NMR ensembles, MD trajectories)
        #start a dictionary for all models:
        model_data = {}

        for model in structure:
            model_id = model.get_id()
            chain_data = {}

            #extract per chain residue information.
            resname_per_chain = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
            resnum_per_chain = {chain.id:[residue.get_id()[1] for residue in chain] for chain in structure.get_chains()}

            #each model can have different functional chains. Currently function is not stored. We can add this as a global or chain property later.
            for chain in model:
                chain_id = chain.get_id()

                #only store atom types and sequences for model 0. In NMR ensembles and MD trajectories topology remains the same over frames (excluding some special cases we might have to adept for later).
                if model_id == 0:
                    top_polypeptide = []
                xyz_polypeptide, resid_polypeptide, bfac_polypeptide, occ_polypeptide = [], [], [], []

                if self.experimental == True:
                    if model_id == 0:
                        #atom type
                        top_ions, top_waters, top_ligands, top_dna, top_rna, top_polypeptide = [], [], [], [], [], []    
                    #xyz positions
                    xyz_ions, xyz_waters, xyz_ligands, xyz_dna, xyz_rna, xyz_polypeptide = [], [], [], [], [], []
                    #amino_atom_id
                    resid_ions, resid_waters, resid_ligands, resid_dna, resid_rna, resid_polypeptide = [], [], [], [], [], []
                    #b_factor
                    bfac_ions, bfac_waters, bfac_ligands, bfac_dna, bfac_rna, bfac_polypeptide = [], [], [], [], [], []
                    #occupancy
                    occ_ions, occ_waters, occ_ligands, occ_dna, occ_rna, occ_polypeptide = [], [], [], [], [], []

                #itterate over all residues within a chain
                for residue in chain:
                    res_hetatm = residue.get_id()[0]
                    res_num = residue.get_id()[1]
                    res_name = residue.get_resname()

                    #itterate over all atoms within a residue
                    for atom in residue:
                        atom_name = atom.get_name()
                        atom_amino_id = res_num
                        atom_coord = atom.get_coord()
                        atom_occu = atom.get_occupancy()
                        atom_bfac = atom.get_bfactor()

                        #filter to arrays based on biological structure separation (waters, ions, dna, rna, ligands and the polypeptide chain)
                        if self.experimental == True:
                            if res_name in dna_map:
                                xyz_dna.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_dna.append(atom_bfac)
                                occ_dna.append(atom_occu)
                                resid_dna.append(atom_amino_id)
                                if model_id == 0:
                                    top_dna.append(atom_name)
                            elif res_name in rna_map:
                                xyz_rna.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_rna.append(atom_bfac)
                                resid_rna.append(atom_amino_id)
                                occ_rna.append(atom_occu)
                                if model_id == 0:
                                    top_rna.append(atom_name)
                            elif res_name in ion_map:
                                xyz_ions.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_ions.append(atom_bfac)
                                resid_ions.append(atom_amino_id)
                                occ_ions.append(atom_occu)
                                if model_id == 0:
                                    top_ions.append(atom_name)
                            elif res_hetatm == ' ': #polypeptide
                                xyz_polypeptide.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_polypeptide.append(atom_bfac)
                                resid_polypeptide.append(atom_amino_id)
                                occ_polypeptide.append(atom_occu)
                                if model_id == 0:
                                    top_polypeptide.append(atom_name)
                            elif res_hetatm == 'W':
                                xyz_waters.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_waters.append(atom_bfac)
                                resid_waters.append(atom_amino_id)
                                occ_waters.append(atom_occu)
                                if model_id == 0:
                                    top_waters.append(atom_name)
                            else: #ligands
                                xyz_ligands.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                                bfac_ligands.append(atom_bfac)
                                resid_ligands.append(atom_amino_id)
                                occ_ligands.append(atom_occu)
                                if model_id == 0:
                                    top_ligands.append(atom_name)
                        else:
                            xyz_polypeptide.append((atom_coord[0], atom_coord[1], atom_coord[2]))
                            bfac_polypeptide.append(atom_bfac)
                            occ_polypeptide.append(atom_occu)
                            resid_polypeptide.append(atom_amino_id)
                            if model_id == 0:
                                top_polypeptide.append(atom_name)

                #merge into a nested dictionary such that datapath is e.g: protein/model/chain/polypeptide/xyz
                polypeptide = {'xyz': xyz_polypeptide,
                    'bfac': bfac_polypeptide,
                    'occupancy': occ_polypeptide,}
                residues = {'pos' : resnum_per_chain[chain_id],
                    'seq1' : resname_per_chain[chain_id],}

                #only create the key if it exists.
                if self.experimental == True:
                    if xyz_ligands:
                        ligands = {'xyz': xyz_ligands,
                            'bfac': bfac_ligands,
                            'occupancy': occ_ligands,}
                    if xyz_ions:
                        ions =  {'xyz': xyz_ions,
                            'bfac': bfac_ions,
                            'occupancy': occ_ions,}
                    if xyz_waters:
                        waters = {'xyz': xyz_waters,
                            'bfac': bfac_waters,
                            'occupancy': occ_waters,}
                    if xyz_dna:
                        dna = {'xyz': xyz_dna,
                            'bfac': bfac_dna,
                            'occupancy': occ_dna,}
                    if xyz_rna:
                        rna = {'xyz': xyz_rna,
                            'bfac': bfac_rna,
                            'occupancy': occ_rna,}

                if model_id == 0:
                    possible_keys = {}
                    # polypeptide |= {'type': top_polypeptide, 'atom_amino_id': resid_polypeptide}
                    polypeptide.update({'type': np.array(top_polypeptide, dtype='S'), 'atom_amino_id': resid_polypeptide})
                    if self.experimental == True:
                        #again only add keys if data actually exists.
                        if top_ligands:
                            ligands |= {'type': top_ligands, 'atom_amino_id': resid_ligands}
                            possible_keys['ligands'] = ligands
                        if top_ions:
                            ions |= {'type': top_ions, 'atom_amino_id': resid_ions}
                            possible_keys['ions'] = ions
                        if top_waters:
                            waters |= {'type': top_waters, 'atom_amino_id': resid_waters}
                            possible_keys['waters'] = waters
                        if top_dna:
                            dna |= {'type': top_dna, 'atom_amino_id': resid_dna}
                            possible_keys['dna'] = dna
                        if top_rna:
                            rna |= {'type': top_rna, 'atom_amino_id': resid_rna}
                            possible_keys['rna'] = rna
                        
                #all info per chain
                if self.experimental == True:
                    experimental_data = {}
                    for key, data in possible_keys.items():
                        experimental_data[key] = data
                            
                    chain_data[chain_id] = experimental_data
                    chain_data[chain_id].update({
                        'polypeptide': polypeptide,
                        'residues': residues,
                    })
                else:
                    chain_data[chain_id] = {
                        'polypeptide': polypeptide,
                        'residues': residues,
                    }

            #all info per model
            model_data[model_id] = chain_data

        #we return a dict of the data arrays for flexibility
        return {
            'identifier': identifier,
            'model_data': model_data,
        }

    def add_structure(self, identifier, structure_file, log_structures=False):
        '''
        Adds a single structure to the H5 file.
        '''
        if log_structures:
            log.info(f'Adding {identifier} to h5 file {self.h5_file}')

        pos_data = self.structure_topology(identifier, structure_file)

        with h5py.File(self.h5_file, 'a') as file:
            try:
                file.create_group(f'{identifier}', 'S')
            except: #determine error type for group already exists
                log.info(f'{identifier} already exists in the data structure')
                return
    
            file.create_group(f'{identifier}/structure', 'S')

            for model in pos_data['model_data']:
                file.create_group(f'{identifier}/structure/{model}', 'S')

                for chain in pos_data['model_data'][model]:
                    file.create_group(f'{identifier}/structure/{model}/{chain}')

                    for info_key in pos_data['model_data'][model][chain].keys():
                        file.create_group(f'{identifier}/structure/{model}/{chain}/{info_key}')

                        for data_key in pos_data['model_data'][model][chain][info_key].keys():
                            file[f'{identifier}/structure/{model}/{chain}/{info_key}'].create_dataset(data_key, data=pos_data['model_data'][model][chain][info_key][data_key])

            file.close()
        return

    def add_folder(self, n_cores=1):
        '''
        Adds a complete folder to h5 storage.
        path: base_path where the protein structures are stored.
        n_cores: amount of cores to use when processing the data
        '''
        if self.file_type == 'PDB':
            files= glob.glob1(self.pdbroot, '*.pdb')
        elif self.file_type == 'MMCIF':
            files = glob.glob1(self.pdbroot, '*.cif')
        else:
            raise ValueError('Please select either "PDB" or "MMCIF" as strucural input.')
        
        identifiers = [file[:-4] for file in files]

        prog = 0
        for file, ids in zip(files, identifiers):
            self.add_structure(ids, file)
            prog +=1 

            if prog % 500 == 0:
                log.info(f'Added {prog} structures to {self.h5_file}')
        return
    
    def add_list_of_structures(self, list_identifier, list_structure_file): 
        '''
        Adds a list of structures to the H5 file.
        '''
        with h5py.File(self.h5_file, 'w') as file:
            #loop over all structures
            prog = 0
            for identifier, structure_file in zip(list_identifier, list_structure_file):
                try:
                    file.create_group(f'{identifier}', 'S')
                    pos_data = self.structure_topology(identifier, structure_file)
                except: #determine error type for group already exists
                    log.info(f'{identifier} already exists in the data structure')
                    continue

                file.create_group(f'{identifier}/structure', 'S')

                for model in pos_data['model_data']:
                    file.create_group(f'{identifier}/structure/{model}', 'S')

                    for chain in pos_data['model_data'][model]:
                        file.create_group(f'{identifier}/structure/{model}/{chain}')

                        for info_key in pos_data['model_data'][model][chain].keys():
                            file.create_group(f'{identifier}/structure/{model}/{chain}/{info_key}')

                            for data_key in pos_data['model_data'][model][chain][info_key].keys():
                                file[f'{identifier}/structure/{model}/{chain}/{info_key}'].create_dataset(data_key, data=pos_data['model_data'][model][chain][info_key][data_key])
                prog += 1
                if prog % 500 == 0:
                    log.info(f'Added {prog} structures to {self.h5_file}')
        return        
    
    
    def return_single_structure(self):
        #TODO print info back to pdb file for visual inspection
        return

    def remove_structure(self, identifier):
        '''
        Removes specific structure record from the HDF5 File.
        '''
        with h5py.File(h5_file, 'a') as file:
            del file[f'{identifier}']['structure']

        return