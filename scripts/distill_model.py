# © Copyright Technical University of Denmark
"""
Distill ensemble model (6 models) into one.

Training data is loaded from a pickled dict that contains 
the ensembled probs (average of 6 model outputs)
Contains the following:
    'input_ids': (n_samples, 73) tensor of input ids
    'input_mask': (n_samples, 73) tensor of input mask
    'pos_probs': (n_samples, 73, 37) tensor of marginal probs
    'type': np array of SP type labels, used for balancing

    alternatively, if distilling emissions instead of marginal
    probs (--use_emissions_loss):
    'emissions': (n_samples, 73, 37) tensor of emissions

Example command:
python3 distill_bert_crf.py --kingdom_as_token --constrain_crf --sp_region_labels --remove_top_layers 1 --weighted_loss

"""
import torch
import pickle
import wandb
import os
import sys
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
import argparse
import numpy as np
import logging
import time


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    return logger


def log_metrics(metrics_dict, split: str, step: int):
    """Convenience function to add prefix to all metrics before logging."""
    wandb.log(
        {
            f"{split.capitalize()} {name.capitalize()}": value
            for name, value in metrics_dict.items()
        },
        step=step,
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    model: torch.nn.Module,
    optimizer,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    pos_probs: torch.Tensor,
    sp_classes: np.ndarray,
    weights: np.ndarray,
    args,
    global_step,
):
    """Cut batches from tensors and process batch-wise
    pos_probs are either the CRF marginals, or the emissions. Specify with args.use_emissions_loss."""
    # process as batches
    all_losses = []
    all_losses_spi = []
    all_losses_spii = []
    all_losses_tat = []
    all_losses_tatlipo = []
    all_losses_spiii = []
    all_losses_other = []
    b_start = 0
    b_end = args.batch_size

    model.train()

    while b_start < len(input_ids):
        ids = input_ids[b_start:b_end, :].to(device)
        true_pos_probs = pos_probs[b_start:b_end, :, :].to(device)
        mask = input_mask[b_start:b_end, :].to(device)
        classes = sp_classes[b_start:b_end]
        weight = torch.Tensor(weights[b_start:b_end]).to(device)

        optimizer.zero_grad()

        global_probs, pred_pos_probs, pos_preds, pred_emissions, mask = model(
            ids, input_mask=mask, return_emissions=True
        )

        # pos_probs: batch_size x seq_len x n_labels
        if args.use_emissions_loss:
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(
                    pred_emissions / args.sm_temperature, dim=-1
                ),
                torch.nn.functional.softmax(
                    true_pos_probs / args.sm_temperature, dim=-1
                ),
                reduction="none",
            )
            loss = loss.sum(dim=-1).mean(dim=-1)
        else:
            loss = torch.nn.functional.kl_div(
                torch.log(pred_pos_probs), true_pos_probs, reduction="none"
            )
            loss = loss.sum(dim=-1).mean(
                dim=-1
            )  # one loss per sample, sum over probability distribution axis
        # weight
        loss_weighted = loss * weight
        loss_weighted.mean().backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        log_metrics(
            {
                "KL Loss": loss.mean().item(),
                "KL Loss weighted": loss_weighted.mean().item(),
            },
            "train",
            global_step,
        )
        global_step += 1

        loss = loss.detach().cpu().numpy()
        all_losses.append(loss)
        all_losses_spi.append(loss[classes == "SP"])
        all_losses_spii.append(loss[classes == "LIPO"])
        all_losses_tat.append(loss[classes == "TAT"])
        all_losses_tatlipo.append(loss[classes == "TATLIPO"])
        all_losses_spiii.append(loss[classes == "PILIN"])
        all_losses_other.append(loss[classes == "NO_SP"])

        b_start = b_start + args.batch_size
        b_end = b_end + args.batch_size

    all_losses = np.concatenate(all_losses).mean()
    all_losses_spi = np.concatenate(all_losses_spi).mean()
    all_losses_spii = np.concatenate(all_losses_spii).mean()
    all_losses_tat = np.concatenate(all_losses_tat).mean()
    all_losses_tatlipo = np.concatenate(all_losses_tatlipo).mean()
    all_losses_spiii = np.concatenate(all_losses_spiii).mean()
    all_losses_other = np.concatenate(all_losses_other).mean()
    log_metrics(
        {
            "KL All": all_losses,
            "KL SP": all_losses_spi,
            "KL LIPO": all_losses_spii,
            "KL TAT": all_losses_tat,
            "KL TATLIPO": all_losses_tatlipo,
            "KL PILIN": all_losses_spiii,
            "KL Other": all_losses_other,
        },
        "train",
        global_step,
    )

    return (
        sum(
            [
                all_losses_other,
                all_losses_spi,
                all_losses_spii,
                all_losses_tat,
                all_losses_tatlipo,
                all_losses_spiii,
            ]
        )
        / 6,
        global_step,
    )


def main_training_loop(args: argparse.ArgumentParser):

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = setup_logger()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)

    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())

    # TODO get rid of this dirty fix once wandb works again
    global wandb
    import wandb

    if (
        wandb.run is None and not args.crossval_run
    ):  # Only initialize when there is no run yet (when importing main_training_loop to other scripts)
        wandb.init(dir=args.output_dir, name=args.experiment_name)
    else:
        wandb = DecoyWandb()

    # Set seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        seed = args.random_seed
    else:
        seed = torch.seed()

    logger.info(f"torch seed: {seed}")
    wandb.config.update({"seed": seed})

    logger.info(f"Saving to {args.output_dir}")

    # Here we set up the config as always and load the pretrained checkpoint.

    logger.info(f"Loading pretrained model in {args.resume}")
    config = BertConfig.from_pretrained(args.resume)

    if config.xla_device:
        setattr(config, "xla_device", False)

    setattr(config, "num_labels", args.num_seq_labels)
    setattr(config, "num_global_labels", args.num_global_labels)

    setattr(config, "lm_output_dropout", args.lm_output_dropout)
    setattr(config, "lm_output_position_dropout", args.lm_output_position_dropout)
    setattr(config, "crf_scaling_factor", 1)
    setattr(
        config, "use_large_crf", True
    )  # legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.

    if args.sp_region_labels:
        setattr(config, "use_region_labels", True)

    # hardcoded for full model, 5 classes, 37 tags
    if args.constrain_crf and args.sp_region_labels:
        allowed_transitions = [
            # NO_SP
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (2, 2),  # I-I, I-M, M-M, M-O, M-I, O-M, O-O
            # SPI
            # 3 N, 4 H, 5 C, 6 I, 7M, 8 O
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (5, 5),
            (5, 8),
            (8, 8),
            (8, 7),
            (7, 7),
            (7, 6),
            (6, 6),
            (6, 7),
            (7, 8),
            # SPII
            # 9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
            (9, 9),
            (9, 10),
            (10, 10),
            (10, 11),
            (11, 11),
            (11, 12),
            (12, 15),
            (15, 15),
            (15, 14),
            (14, 14),
            (14, 13),
            (13, 13),
            (13, 14),
            (14, 15),
            # TAT
            # 16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
            (16, 16),
            (16, 17),
            (17, 17),
            (17, 16),
            (16, 18),
            (18, 18),
            (18, 19),
            (19, 19),
            (19, 22),
            (22, 22),
            (22, 21),
            (21, 21),
            (21, 20),
            (20, 20),
            (20, 21),
            (21, 22),
            # TATLIPO
            # 23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
            (23, 23),
            (23, 24),
            (24, 24),
            (24, 23),
            (23, 25),
            (25, 25),
            (25, 26),
            (26, 26),
            (26, 27),
            (27, 30),
            (30, 30),
            (30, 29),
            (29, 29),
            (29, 28),
            (28, 28),
            (28, 29),
            (29, 30),
            # PILIN
            # 31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
            (31, 31),
            (31, 32),
            (32, 32),
            (32, 33),
            (33, 33),
            (33, 36),
            (36, 36),
            (36, 35),
            (35, 35),
            (35, 34),
            (34, 34),
            (34, 35),
            (35, 36),
        ]
        allowed_starts = [0, 2, 3, 9, 16, 23, 31]
        allowed_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]

        setattr(config, "allowed_crf_transitions", allowed_transitions)
        setattr(config, "allowed_crf_starts", allowed_starts)
        setattr(config, "allowed_crf_ends", allowed_ends)

    # setattr(config, 'gradient_checkpointing', True) #hardcoded when working with 256aa data
    if args.kingdom_as_token:
        setattr(
            config, "kingdom_id_as_token", True
        )  # model needs to know that token at pos 1 needs to be removed for CRF

    if args.global_label_as_input:
        setattr(config, "type_id_as_token", True)

    if args.remove_top_layers > 0:
        # num_hidden_layers if bert
        n_layers = (
            config.num_hidden_layers
            if args.model_architecture == "bert_prottrans"
            else config.n_layer
        )
        if args.remove_top_layers > n_layers:
            logger.warning(f"Trying to remove more layers than there are: {n_layers}")
            args.remove_top_layers = n_layers

        setattr(
            config,
            "num_hidden_layers"
            if args.model_architecture == "bert_prottrans"
            else "n_layer",
            n_layers - args.remove_top_layers,
        )

    model = BertSequenceTaggingCRF.from_pretrained(args.resume, config=config)

    if args.kingdom_as_token:
        logger.info(
            "Using kingdom IDs as word in sequence, extending embedding layer of pretrained model."
        )
        tokenizer = ProteinBertTokenizer.from_pretrained(
            "resources/vocab_with_kingdom", do_lower_case=False
        )
        model.resize_token_embeddings(tokenizer.tokenizer.vocab_size)

    ## Setup data

    data_dict = pickle.load(open(args.data, "rb"))
    all_input_ids = data_dict["input_ids"]
    all_input_masks = data_dict["input_mask"]
    all_pos_probs = (
        data_dict["pos_probs"]
        if not args.use_emissions_loss
        else data_dict["emissions"]
    )
    all_sp_classes = data_dict["type"]
    logger.info("loaded pickled data")

    ## Setup weights
    if args.weighted_loss:
        values, counts = np.unique(all_sp_classes, return_counts=True)
        count_dict = dict(zip(values, counts))
        weights = np.array([1.0 / count_dict[i] for i in all_sp_classes]) * len(
            all_sp_classes
        )
    else:
        weights = np.ones_like(all_sp_classes)

    # shuffe everything
    rand_idx = np.random.permutation(len(all_input_ids))
    all_input_ids = all_input_ids[rand_idx]
    all_input_masks = all_input_masks[rand_idx]
    all_pos_probs = all_pos_probs[rand_idx]
    all_sp_classes = all_sp_classes[rand_idx]
    weights = weights[rand_idx]

    # set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update(model.config.to_dict())

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    if args.optimizer == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )

    model.to(device)

    logger.info("Model set up!")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} trainable parameters")

    logger.info(f"Running model on {device}, not using nvidia apex")

    best_divergence = 1000000000000000
    global_step = 0

    for epoch in range(1, args.epochs + 1):

        epoch_kl_divergence, global_step = train(
            model,
            optimizer,
            all_input_ids,
            all_input_masks,
            all_pos_probs,
            all_sp_classes,
            weights,
            args,
            global_step,
        )
        if epoch_kl_divergence < best_divergence:
            best_divergence = epoch_kl_divergence
            model.save_pretrained(args.output_dir)
            logger.info(
                f"New best model with loss {epoch_kl_divergence}, training step {global_step}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train CRF on top of Pfam Bert")
    parser.add_argument(
        "--data",
        type=str,
        default="sp_prediction_experiments/ensemble_probs_for_distillation.pkl",
        help="location of the data corpus. Expects test, train and valid .fasta",
    )

    # args relating to training strategy.
    parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=8000, help="upper epoch limit")

    parser.add_argument(
        "--batch_size", type=int, default=80, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1.2e-6,
        help="weight decay applied to all weights",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer to use (sgd, adam)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f'{time.strftime("%Y-%m-%d",time.gmtime())}_awd_lstm_lm_pretraining',
        help="path to save logs and trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="Rostlab/prot_bert",
        help="path of model to resume (directory containing .bin and config.json",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Distill-Bert-CRF",
        help="experiment name for logging",
    )
    parser.add_argument(
        "--crossval_run",
        action="store_true",
        help="override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.",
    )
    parser.add_argument(
        "--log_all_final_metrics",
        action="store_true",
        help="log all final test/val metrics to w&b",
    )

    parser.add_argument("--num_seq_labels", type=int, default=37)
    parser.add_argument("--num_global_labels", type=int, default=6)
    parser.add_argument(
        "--global_label_as_input",
        action="store_true",
        help="Add the global label to the input sequence (only predict CS given a known label)",
    )

    parser.add_argument(
        "--weighted_loss", action="store_true", help="Balance loss for all classes"
    )
    parser.add_argument(
        "--use_emissions_loss",
        action="store_true",
        help='Use emissions instead of marginal probs. Need specify a pkl file as data that has "emissions"',
    )
    parser.add_argument(
        "--sm_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature when training on emissions",
    )

    parser.add_argument(
        "--lm_output_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output",
    )
    parser.add_argument(
        "--lm_output_position_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output, drops full hidden states from sequence",
    )
    parser.add_argument(
        "--random_seed", type=int, default=None, help="random seed for torch."
    )

    # args for model architecture
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="bert_prottrans",
        help="which model architecture the checkpoint is for",
    )
    parser.add_argument(
        "--remove_top_layers",
        type=int,
        default=0,
        help="How many layers to remove from the top of the LM.",
    )
    parser.add_argument(
        "--kingdom_as_token",
        action="store_true",
        help="Kingdom ID is first token in the sequence",
    )
    parser.add_argument(
        "--sp_region_labels",
        action="store_true",
        help="Use labels for n,h,c regions of SPs.",
    )
    parser.add_argument(
        "--constrain_crf",
        action="store_true",
        help="Constrain the transitions of the region-tagging CRF.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = "_".join([args.experiment_name, time_stamp])

    args.output_dir = os.path.join(args.output_dir, full_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    main_training_loop(args)
