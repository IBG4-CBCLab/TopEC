import os
from typing import List
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger
import numpy as np
from src import utils
log = utils.get_logger(__name__)
from pycm import ConfusionMatrix

import tensorflow as tf
import tensorboard as tb
from torchmetrics.functional import accuracy, f1_score
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def write_results(
    labels,
    preds,
    confidences,
    num_classes,
    pycm_report_dir,
    dataset_type="Validation",
):

    # print(f'labels: {labels}')
    # print(f'len labels: {len(labels)}')
    # print(f'preds: {preds}')
    # print(f'len preds: {len(preds)}')

    cm = ConfusionMatrix(labels, preds)
    log.info(f"{dataset_type} Confusion Matrix is Created")

    cm.save_html(pycm_report_dir + f"{dataset_type}_ACC", False, color=(0, 64, 64))
    log.info(f"{dataset_type} Confusion Matrix is Saved at {pycm_report_dir}")
 
    acc_micro = accuracy(
        torch.tensor(preds),
        torch.tensor(labels),
        num_classes=num_classes,
        average="micro",
    )
    acc_macro = accuracy(
        torch.tensor(preds),
        torch.tensor(labels),
        num_classes=num_classes,
        average="macro",
    )

    f1_micro = f1_score(
        torch.tensor(preds),
        torch.tensor(labels),
        num_classes=num_classes,
        average="micro",
    )
    f1_macro = f1_score(
        torch.tensor(preds),
        torch.tensor(labels),
        num_classes=num_classes,
        average="macro",
    )

    with open(pycm_report_dir + "results.txt", "a+") as f:
        f.write("-----\n")
        f.write(f"{dataset_type} :\n")
        f.write(f"ACC = {acc_micro.item()}\n")
        f.write(f"ACC(Macro)={acc_macro.item()}\n")
        f.write(f"F1={f1_micro.item()}\n")
        f.write(f"F1(MACRO)={f1_macro.item()}\n")

    log.info(
        f"{dataset_type} Metrics are calculated and saved at {pycm_report_dir}.results.txt"
    )

    utils.reliability_diagram(
        np.array(labels),
        np.array(preds),
        np.array(confidences),
        filepath=pycm_report_dir + f"{dataset_type}_realibility.png",
    )
    
    utils.plot_roc_curve(
        np.array(labels),
        np.array(preds),
        np.array(confidences),
        filepath=pycm_report_dir + f"{dataset_type}_roc_curve.png",
    )

    utils.plot_precision_recall_curve(
        np.array(labels),
        np.array(preds),
        np.array(confidences),
        filepath=pycm_report_dir + f"{dataset_type}_pr_curve.png",
    )
    return


def evaluate(config: DictConfig, trainer: Trainer, model: LightningModule, datamodule: LightningDataModule, logger ):

    log.info("////////////////     Starting Testing!     ////////////////////")

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule
    )

    val_labels = []
    val_names = []
    for batch in datamodule.val_dataloader():
        val_labels.extend(batch.y.detach().cpu().numpy().tolist())
        val_names.extend(batch.enzyme_name)

    test_labels = []
    test_names = []
    for batch in datamodule.test_dataloader():
        test_labels.extend(batch.y.detach().cpu().numpy().tolist())
        test_names.extend(batch.enzyme_name)

    log.info(f"Predictions {len(predictions)}")

    # print(f'test_labels: {test_labels}')
    # print(f'len test_labels: {len(test_labels)}')
    # print(f'test_names: {test_names}')
    # print(f'len test_names: {len(test_names)}')

    val_preds = []
    val_confidences = []
    val_conf2d = []

    for prediction, confidence in predictions[0]:
        preds = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        confidences = np.max(confidence.detach().cpu().numpy(), axis=1)
        conf2d = confidence.detach().cpu().numpy()
        val_preds.extend(preds.tolist())
        val_confidences.extend(confidences.tolist())
        val_conf2d.extend(conf2d)

    embedding_labels = [[i, j] for i, j in zip(val_labels, val_names)]


    logger[0].experiment.add_embedding(
        np.vstack(val_conf2d),
        metadata=embedding_labels,
        tag="validation embedding",
        metadata_header=["class", "enzyme"],
    ) 

    test_preds = []
    test_confidences = []
    test_conf2d = []

    for prediction, confidence in predictions[1]:

        preds = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        confidences = np.max(confidence.detach().cpu().numpy(), axis=1)
        conf2d = confidence.detach().cpu().numpy()
        test_preds.extend(preds.tolist())
        test_confidences.extend(confidences.tolist())
        test_conf2d.extend(conf2d)

    # print(f'test_preds: {test_preds}')
    # print(f'len test_preds: {len(test_preds)}')
    # print(f'test_confidences: {test_confidences}')
    # print(f'len test_confidences: {len(test_confidences)}')
    # print(f'test_conf2d: {test_conf2d}')
    # print(f'len test_conf2d: {len(test_conf2d)}')
    # print(f'predictions[1]: {predictions[1]}')
    # print(f'len predictions[1]: {len(predictions[1])}')

    embedding_labels = [[i, j] for i, j in zip(test_labels, test_names)]

    logger[0].experiment.add_embedding(
        np.vstack(test_conf2d),
        metadata=embedding_labels,
        tag="test embedding",
        metadata_header=["class", "enzyme"],
    )

    log.info(f"Validation labels size = {len(val_labels)}")
    log.info(f"Validation preds size= {len(val_preds)}")

    log.info(f"Test labels size = {len(test_labels)}")
    log.info(f"Test preds size = {len(test_preds)}")

    log.info(f"Predictions {len(predictions)}")

    with open(config.pycm_report_dir + 'test_results.csv', 'w') as specific_results:
        for i, j, k, l in zip(test_names, test_labels, test_preds, test_confidences):
            specific_results.write('%s,%s,%s,%s\n'%(i,j,k, l))
            
    write_results(
        val_labels,
        val_preds,
        val_confidences,
        config.model.num_classes,
        config.pycm_report_dir,
        "Validation",
    )
    write_results(
        test_labels,
        test_preds,
        test_confidences,
        config.model.num_classes,
        config.pycm_report_dir,
        "Test",
    )
    log.info("Results are written")
    

def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline. Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    datamodule.setup()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    
    evaluate(config, trainer, model, datamodule, logger)
    
    
    
    
    
