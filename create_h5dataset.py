# import src.preprocessing.hdf5_store_data
from src.preprocessing import hdf5_store_data
import pandas as pd
from multiprocessing import Pool
from omegaconf import DictConfig, OmegaConf
import os
import h5py
import hydra
# from src import utils
# log = logging.getLogger(__name__)

class create_h5files():
    """
    A class for creating HDF5 datasets from a given DataFrame containing enzyme information.
    
    Attributes:
        csv_file (str): The path to the CSV file containing enzyme information.
        pdb_root (str): The root directory containing PDB files.
        h5_root (str): The root directory where HDF5 files will be stored.
        h5_name (str): The name of the HDF5 file to be created.
        n_cpus (int): The number of CPUs for parallel processing.
    
    
    Methods:
        split_information(df, n_splits):
            Splits the DataFrame into 'n_splits' chunks for parallel processing.
        
        construct_single_h5(folder, file, n_split, df):
            Constructs a single HDF5 file from a portion of the DataFrame.
        
        creation_multiprocessing(folder, h5file, main_csv, n_cpus):
            Creates HDF5 datasets using multiprocessing for parallel processing.
        
        create_pdb_datasets(folder, h5file, main_csv, pdb_root, n_cpus=1):
            Creates HDF5 datasets for PDB files, optionally using multiprocessing.
    """
    def __init__(
        self,
        csv_file,
        pdb_root,
        h5_root,
        h5_name,
        n_cpus,
        ):

        self.csv_file = pd.read_csv(csv_file)
        self.pdb_root = pdb_root
        self.h5_root = h5_root
        self.h5_name = h5_name
        self.n_cpus = n_cpus


    def split_information(self):
        """
        Splits the given DataFrame into 'n_cpus' chunks for parallel processing.

        Returns:
            list: A list of DataFrames, each representing a split of the original DataFrame.
        """

        df_shuffle = self.csv_file.sample(frac=1).reset_index(drop=True)

        rows_per_split = len(df_shuffle) // self.n_cpus
        
        split_dfs = []
        
        for i in range(self.n_cpus):
            start_idx = i * rows_per_split
            end_idx = (i + 1) * rows_per_split if i < self.n_cpus - 1 else None
            split_df = df_shuffle.iloc[start_idx:end_idx]
            split_dfs.append(split_df)
            
        return split_dfs
    
    def construct_chunk_h5(self, n_split, df):
        """
        Constructs a single HDF5 file from a chunk of the DataFrame.

        Args:
            n_split (int): The index of the split portion.
            df (DataFrame): The DataFrame containing enzyme information.

        Returns:
            None
        """
        identifiers = df['enzyme_name'].tolist()
        file_locations = [os.path.join(self.pdb_root, f'{enzyme}') + '.pdb' for enzyme in df['enzyme_name']]

        # Sort both lists based on enzyme identifiers
        identifiers_sorted, file_locations_sorted = zip(*sorted(zip(identifiers, file_locations)))

        # Double-check if enzyme identifiers are in the same order
        for identifier, location in zip(identifiers, file_locations):
            assert identifier in location, f"Enzyme identifier {identifier} not found in file location {location}"
        
        structure_obj = hdf5_store_data.Structure2HDF5(os.path.join(self.h5_root, f'{self.h5_name}_{n_split}'), self.pdb_root, file_type='PDB', warnings=False, experimental=False)
        structure_obj.add_list_of_structures(identifiers, file_locations)
    

    def creation_singleprocess(self):
        """
        Constructs a single HDF5 file of the DataFrame.

        Args:
            df (DataFrame): The DataFrame containing enzyme information.

        Returns:
            None
        """
        identifiers = self.csv_file['enzyme_name'].tolist()
        file_locations = [os.path.join(self.pdb_root, enzyme) + '.pdb' for enzyme in self.csv_file ['enzyme_name']]

        structure_obj = hdf5_store_data.Structure2HDF5(os.path.join(self.h5_root, f'{self.h5_name}'), self.pdb_root, file_type='PDB', warnings=False, experimental=False)
        structure_obj.add_list_of_structures(identifiers, file_locations)
    


    def creation_multiprocessing(self):
        """
        Creates HDF5 datasets using multiprocessing for parallel processing.
        Merges the datasets after separate chunk creation. 

        Returns:
            None
        """

        #create multiprocessing pool
        pool = Pool(processes=self.n_cpus)

        #obtain splits from enzymes.csv
        splits = self.split_information()
        n = range(len(splits))

        #construct the h5 files in parallel
        # results = pool.starmap(self.construct_chunk_h5, zip(n, splits))

        #merge the h5 files. This step takes the longest. 
        #Depending on the size of the dataset it can be quicker to create in a single pass.
        with h5py.File(os.path.join(self.h5_root, self.h5_name), mode='w') as h5fw:
            for h5 in n:
                h5fr = h5py.File(os.path.join(self.h5_root, f'{self.h5_name}_{h5}'), 'r')
                for obj in h5fr.keys():
                    if h5fw.__contains__(obj):
                        continue
                    h5fr.copy(obj, h5fw)


    def create_pdb_datasets(self):
        """
        Creates HDF5 datasets for PDB files, optionally using multiprocessing.

        Args:

        Raises:
            ValueError: If an invalid number of CPUs is provided.

        Returns:
            None
        """
        if self.n_cpus < 2:
            self.creation_singleprocess()
        elif self.n_cpus > 1:
            self.creation_multiprocessing()
        else:
            raise ValueError('Found an invalid number of cpus. 1 for single thread processing. N for n-multithread processing')

@hydra.main(config_path="configs/", config_name="create_h5dataset.yaml")
def main(config: DictConfig):

    # Initialize the create_h5files class with configuration parameters
    h5_creator = create_h5files(
        csv_file=config.csv_file,
        pdb_root=config.pdb_root,
        h5_root=config.h5_root,
        h5_name=config.h5_name,
        n_cpus=config.n_cpus
    )

    # Create the PDB dataset
    h5_creator.create_pdb_datasets()

    return "PDB dataset creation completed successfully"


if __name__ == "__main__":
    main()
    
    