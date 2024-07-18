import hydra
from omegaconf import DictConfig, OmegaConf
import logging
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
#dotenv.load_dotenv(override=True)

log = logging.getLogger(__name__)
@hydra.main(config_path="configs/", config_name="config_dataset_create.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.data.prepare_data import create_pdb_datasets

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    #template_utils.extras(config)

    # Pretty print config using Rich library
    #if config.get("print_config"):
    #    utils.print_config(config, resolve=True)
    # Train model
    return create_pdb_datasets(input_path=config.input_path, 
    enzyme_csv=config.enzyme_csv,
    output_path=config.output_path, 
    weight_path=config.weight_path)


if __name__ == "__main__":
    main()
    
    
    
    
