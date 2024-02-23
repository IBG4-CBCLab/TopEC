import logging
import warnings
from typing import List, Sequence
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import Logger

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert len(confidences) == len(pred_labels)
    assert len(confidences) == len(true_labels)
    assert num_bins > 0

    1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts,
        "bins": bins,
        "avg_accuracy": avg_acc,
        "avg_confidence": avg_conf,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
    }


def _reliability_diagram_subplot(
    ax,
    bin_data,
    draw_ece=True,
    draw_bin_importance=False,
    title="Reliability Diagram",
    xlabel="Confidence",
    ylabel="Expected Accuracy",
):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8 * normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = alphas

    gap_plt = ax.bar(
        positions,
        np.abs(accuracies - confidences),
        bottom=np.minimum(accuracies, confidences),
        width=widths,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions,
        0,
        bottom=accuracies,
        width=widths,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=3,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    if draw_ece:
        ece = bin_data["expected_calibration_error"] * 100
        ax.text(
            0.98,
            0.02,
            "ECE=%.2f" % ece,
            color="black",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(
    ax,
    bin_data,
    draw_averages=True,
    title="Examples per bin",
    xlabel="Confidence",
    ylabel="Count",
):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    ax.bar(positions, counts, width=bin_size * 0.9)

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(
            x=bin_data["avg_accuracy"], ls="solid", lw=3, c="black", label="Accuracy"
        )
        conf_plt = ax.axvline(
            x=bin_data["avg_confidence"],
            ls="dotted",
            lw=3,
            c="#444",
            label="Avg. confidence",
        )
        ax.legend(handles=[acc_plt, conf_plt])


def _reliability_diagram_combined(
    bin_data,
    draw_ece,
    draw_bin_importance,
    draw_averages,
    title,
    figsize,
    dpi,
    filepath,
):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(
        ax[0], bin_data, draw_ece, draw_bin_importance, title=title, xlabel=""
    )

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(int)
    ax[1].set_yticklabels(new_ticks)

    fig.savefig(filepath)
    return


def reliability_diagram(
    true_labels,
    pred_labels,
    confidences,
    num_bins=10,
    draw_ece=True,
    draw_bin_importance=False,
    draw_averages=True,
    title="Reliability Diagram",
    figsize=(6, 6),
    dpi=72,
    filepath="",
):
    """Draws a reliability diagram and confidence histogram in a single plot.

    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin.
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence.
    The closer these two lines are together, the better the calibration.

    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _reliability_diagram_combined(
        bin_data,
        draw_ece,
        draw_bin_importance,
        draw_averages,
        title,
        figsize=figsize,
        dpi=dpi,
        filepath=filepath,
    )


def calc_bins(labels_oneh, preds):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(labels_oneh, preds):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels_oneh, preds)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(labels_oneh, preds, filepath):
    ECE, MCE = get_metrics(labels_oneh, preds)
    bins, _, bin_accs, _, _ = calc_bins(labels_oneh, preds)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor="black", color="r", hatch="\\")

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor="black", color="b")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect("equal", adjustable="box")

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color="green", label="ECE = {:.2f}%".format(ECE * 100))
    MCE_patch = mpatches.Patch(color="red", label="MCE = {:.2f}%".format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    # plt.show()

    plt.savefig(filepath, bbox_inches="tight")

def plot_roc_curve(labels, prediction, confidence, filepath):
    y_true = [1 if i == j else 0 for i, j in zip(labels, prediction)]
    y_score = confidence

    #nan fix
    y_score = [y if y == y else 0 for y in y_score]

    #calc roc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    #initiate figure
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    #roc curve
    ax.plot(fpr, tpr, color='blue')

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title('Micro ROC curve')
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    plt.gca().set_aspect('equal')
    plt.savefig(filepath, bbox_inches="tight")


def plot_precision_recall_curve(labels, prediction, confidence, filepath):

    y_true = [1 if i == j else 0 for i, j in zip(labels, prediction)]
    y_score = confidence

    #nan fix
    y_score = [y if y == y else 0 for y in y_score]

    #calc precission, recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    #initiate figure
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    #F1 curves
    for f1 in np.linspace(0.1, 1, 10):
        x=np.linspace(0, 1, 50)
        y = f1 * x / (2 * x - f1)
        ax.plot(x[y > 0.0], y[y > 0.0], c='gainsboro')
    #precision recall curve
    ax.plot(recall, precision, color='blue')

    #add F1 curve tags
    for f1, ypos in zip(np.linspace(0.1, 0.9, 9), [0.05, 0.1, 0.17, 0.27, 0.35, 0.45, 0.6, 0.72, 0.95]):
        ax.text(0.85, ypos, f'F1={f1:.1f}', c='gray', fontsize=14)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title('Micro precision-recall curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.gca().set_aspect('equal')
    plt.savefig(filepath, bbox_inches="tight")