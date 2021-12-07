# © Copyright Technical University of Denmark
"""
Utils to calculate metrics for SP tagging models.
Last version had the shortcoming of rerunning subsets of the data when getting metrics per kingdom/one-vs-all mccs.
Try to avoid that.
"""

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score
from typing import List, Dict, Union, Tuple
import sys
from .region_similarity import class_aware_cosine_similarities, get_region_lengths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGNALP_VOCAB = [
    "S",
    "I",
    "M",
    "O",
    "T",
    "L",
]  # NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}
REVERSE_GLOBAL_LABEL_DICT = {
    0: "NO_SP",
    1: "SP",
    2: "LIPO",
    3: "TAT",
    4: "TATLIPO",
    5: "PILIN",
}


def run_data(model, dataloader):
    """run all the data of a DataLoader, concatenate and return outputs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    for i, batch in enumerate(dataloader):

        if hasattr(dataloader.dataset, "kingdom_ids") and hasattr(
            dataloader.dataset, "sample_weights"
        ):
            (
                data,
                targets,
                input_mask,
                global_targets,
                sample_weights,
                kingdom_ids,
            ) = batch
            kingdom_ids = kingdom_ids.to(device)
            sample_weights = sample_weights.to(device)
        elif hasattr(dataloader.dataset, "sample_weights") and not hasattr(
            dataloader.dataset, "kingdom_ids"
        ):
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None
        elif hasattr(dataloader.dataset, "kingdom_ids"):
            data, targets, input_mask, global_targets, kingdom_ids = batch
        else:
            data, targets, input_mask, global_targets, sample_weights = batch
            kingdom_ids = None

        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        kingdom_ids = kingdom_ids.to(device)
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                data,
                global_targets=global_targets,
                targets=targets,
                input_mask=input_mask,
                kingdom_ids=kingdom_ids,
            )

        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_targets, all_global_probs, all_targets, all_pos_preds


def run_data_regioncrfdataset(model, dataloader):
    """run all the data of a DataLoader, concatenate and return outputs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_input_ids = []
    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_cs = []

    total_loss = 0
    for i, batch in enumerate(dataloader):

        if hasattr(dataloader.dataset, "kingdom_ids") and hasattr(
            dataloader.dataset, "sample_weights"
        ):
            (
                data,
                targets,
                input_mask,
                global_targets,
                cleavage_sites,
                sample_weights,
                kingdom_ids,
            ) = batch
            kingdom_ids = kingdom_ids.to(device)
            sample_weights = sample_weights.to(device)
        elif hasattr(dataloader.dataset, "sample_weights") and not hasattr(
            dataloader.dataset, "kingdom_ids"
        ):
            (
                data,
                targets,
                input_mask,
                global_targets,
                cleavage_sites,
                sample_weights,
            ) = batch
            kingdom_ids = None
        else:
            (
                data,
                targets,
                input_mask,
                global_targets,
                cleavage_sites,
                sample_weights,
            ) = batch
            kingdom_ids = None

        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(
                data, input_mask=input_mask, kingdom_ids=kingdom_ids
            )

        all_input_ids.append(data.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_cs.append(cleavage_sites)

    all_input_ids = np.concatenate(all_input_ids)
    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_cs = np.concatenate(all_cs)

    return (
        all_input_ids,
        all_global_targets,
        all_global_probs,
        all_targets,
        all_pos_preds,
        all_cs,
    )


def run_data_array(model, sequence_data_array, batch_size=40):
    """run all the data of a np.array, concatenate and return outputs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_global_probs = []
    all_pos_preds = []

    total_loss = 0
    b_start = 0
    b_end = batch_size
    while b_start < len(sequence_data_array):

        data = sequence_data_array[b_start:b_end, :]
        data = torch.tensor(data)
        data = data.to(device)
        # global_targets = global_targets.to(device)
        input_mask = None
        with torch.no_grad():
            global_probs, pos_probs, pos_preds = model(
                data, input_mask=torch.ones_like(data)
            )

        # total_loss += loss.item()
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size

    # all_targets = np.concatenate(all_targets)
    # all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)

    return all_global_probs, all_pos_preds


def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens=[0, 4, 5]):
    """Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. -1 if no SP present in sequence.
    """

    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. """
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = -1
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


def find_cs_tag(tagged_seqs: np.ndarray, cs_tokens=[4, 9, 14]):
    """Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels , with "C" tags for cleavage sites
        cs_tokens = label tokens that indicate a cleavage site
    Returns:
        cs_sites: (batch_size) integer array of position that is a CS. -1 if no SP present in sequence.
    #NOTE this is still the same as tagged_seq_to_cs_multiclass. Implement unique-checks later (only one CS per seq)
    """

    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. """
        sp_idx = np.where(np.isin(x, cs_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = -1
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


def compute_mcc(true_labels: np.ndarray, pred_labels: np.ndarray, label_positive=1):
    """Compute MCC. binarize to true/false, according to label specified.
    For MCC1, subset data before calling this function"""
    # binarize everything
    true_labels = [1 if x == label_positive else 0 for x in true_labels]
    pred_labels = [1 if x == label_positive else 0 for x in pred_labels]

    return matthews_corrcoef(true_labels, pred_labels)


# We used precision and recall to assess CS predictions, where precision is defined as the fraction of CS predictions that are correct,
# and recall is the fraction of real SPs that are predicted as the correct SP type and have the correct CS assigned.


def compute_precision(true_CS: np.ndarray, pred_CS: np.ndarray, window_size: float):
    """Ratio of correct CS predictions in window / total number of predicted CS
    Expects preprocessed vectors where other SP classes are already set to -1"""
    true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)]
    pred_positive = pred_CS[(true_CS != -1) & (pred_CS != -1)]
    correct_in_window = (pred_positive >= true_positive - window_size) & (
        pred_positive <= true_positive + window_size
    )
    precision = correct_in_window.sum() / (pred_CS != -1).sum()

    return precision


def compute_recall(true_CS: np.ndarray, pred_CS: np.ndarray, window_size: int) -> float:
    """Ratio of correct CS predictions in window / total number of true CS"""
    true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)]
    pred_positive = pred_CS[(true_CS != -1) & (pred_CS != -1)]

    correct_in_window = (pred_positive >= true_positive - window_size) & (
        pred_positive <= true_positive + window_size
    )
    recall = correct_in_window.sum() / (true_CS != -1).sum()

    return recall


def mask_cs(
    true_CS: np.ndarray,
    pred_CS: np.ndarray,
    true_label: np.ndarray,
    pred_label: np.ndarray,
    label_positive: int = 1,
    label_negative: int = 0,
):
    """Process cleavage site vectors for precision/recall. CS of samples in other classes need to be masked, don't count as predictions/targets.
    - true CS of samples with true_label not in label_positive/label_negative are set to -1
    - predicted CS of samples with pred_label not in label_positive/label_negative are set to -1

    """
    # SignalP5:
    # We used precision and recall to assess CS predictions, where precision is defined as the fraction of CS predictions that are correct,
    # and recall is the fraction of real SPs that are predicted as the correct SP type and have the correct CS assigned.

    true_CS = true_CS.copy()
    pred_CS = pred_CS.copy()

    # set all true_CS for other types to -1: No CS should be predicted there
    true_CS[true_label != label_positive] = -1

    # set all predicted CS for other types to -1
    pred_CS[~np.isin(pred_label, [label_positive, label_negative])] = -1

    return true_CS, pred_CS


def compute_metrics(
    all_global_targets: np.ndarray,
    all_global_preds: np.ndarray,
    all_cs_targets: np.ndarray,
    all_cs_preds: np.ndarray,
    all_kingdom_ids: List[str],
    rev_label_dict: Dict[int, str] = REVERSE_GLOBAL_LABEL_DICT,
):
    """Compute the metrics used in the SignalP 5.0 supplement.
    Returns all metrics as a dict.
    """
    metrics_dict = {}

    for kingdom in np.unique(all_kingdom_ids):
        # subset for the kingdom
        kingdom_targets = all_global_targets[all_kingdom_ids == kingdom]
        kingdom_preds = all_global_preds[all_kingdom_ids == kingdom]
        kingdom_cs_targets = all_cs_targets[all_kingdom_ids == kingdom]
        kingdom_cs_preds = all_cs_preds[all_kingdom_ids == kingdom]

        for sp_type in np.unique(kingdom_targets):
            # subset for each sp type
            if sp_type == 0:
                continue  # skip NO_SP

            # use full data
            metrics_dict[f"{kingdom}_{rev_label_dict[sp_type]}_mcc2"] = compute_mcc(
                kingdom_targets, kingdom_preds, label_positive=sp_type
            )

            # subset for no_sp
            mcc1_idx = np.isin(kingdom_targets, [sp_type, 0])
            metrics_dict[f"{kingdom}_{rev_label_dict[sp_type]}_mcc1"] = compute_mcc(
                kingdom_targets[mcc1_idx],
                kingdom_preds[mcc1_idx],
                label_positive=sp_type,
            )

            for window_size in [0, 1, 2, 3]:

                # mask the cs predictions: set all other type targets to -1 and remove other classes based on global predicted label
                true_CS, pred_CS = mask_cs(
                    kingdom_cs_targets,
                    kingdom_cs_preds,
                    kingdom_targets,
                    kingdom_preds,
                    label_positive=sp_type,
                    label_negative=0,
                )

                metrics_dict[
                    f"{kingdom}_{rev_label_dict[sp_type]}_recall_window_{window_size}"
                ] = compute_recall(true_CS, pred_CS, window_size=window_size)
                # mask predictions for precision: global pred has effect
                metrics_dict[
                    f"{kingdom}_{rev_label_dict[sp_type]}_precision_window_{window_size}"
                ] = compute_precision(true_CS, pred_CS, window_size=window_size)

    return metrics_dict


def compute_metrics_old(
    all_global_targets: np.ndarray,
    all_global_preds: np.ndarray,
    all_cs_targets: np.ndarray,
    all_cs_preds: np.ndarray,
    all_kingdom_ids: List[str],
    rev_label_dict: Dict[int, str] = REVERSE_GLOBAL_LABEL_DICT,
):
    """Compute the metrics used in the SignalP 5.0 supplement.
    Returns all metrics as a dict.
    """
    metrics_dict = {}
    # get simple overall global metrics
    metrics_dict["mcc"] = matthews_corrcoef(all_global_targets, all_global_preds)

    # subset data for each kingdom
    for kingdom in np.unique(all_kingdom_ids):
        kingdom_targets = all_global_targets[all_kingdom_ids == kingdom]
        kingdom_preds = all_global_preds[all_kingdom_ids == kingdom]
        kingdom_cs_targets = all_cs_targets[all_kingdom_ids == kingdom]
        kingdom_cs_preds = all_cs_preds[all_kingdom_ids == kingdom]

        # subset data for each sp type
        for sp_type in np.unique(kingdom_targets):

            if sp_type == 0:
                continue
            # one-vs-all mcc
            targets_mcc2 = kingdom_targets.copy()
            preds_mcc2 = kingdom_preds.copy()
            # make same label for all that are not target type
            targets_mcc2[targets_mcc2 != sp_type] = 0
            preds_mcc2[preds_mcc2 != sp_type] = 0
            metrics_dict[
                f"{kingdom}_{rev_label_dict[sp_type]}_mcc2"
            ] = matthews_corrcoef(targets_mcc2, preds_mcc2)
            # one-vs-no_sp mcc
            targets_mcc1 = kingdom_targets[np.isin(kingdom_targets, [sp_type, 0])]
            preds_mcc1 = kingdom_preds[np.isin(kingdom_targets, [sp_type, 0])]
            metrics_dict[
                f"{kingdom}_{rev_label_dict[sp_type]}_mcc1"
            ] = matthews_corrcoef(targets_mcc1, preds_mcc1)

            # subset current type + type 0 (no sp)
            true_CS, pred_CS = (
                kingdom_cs_targets[np.isin(kingdom_targets, [sp_type, 0])],
                kingdom_cs_preds[np.isin(kingdom_targets, [sp_type, 0])],
            )

            # get CS metrics
            # For the precision calculation, you have a vector with the position of the true and predicted cs.
            # So if the model doesn’t predict a SP, you would have -1 in the pred_CS, I assume.
            # The first thing that you want to do is subset proteins predicted as having a SP (pred_CS != -1) and also having a true SP
            # so we can compare the cleavage sites (true_CS != -1), which I don’t see in your code.
            # What you want to have is true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)],
            # and pred_positive = pred_CS[(true_CS != -1) & (pred_CS != -1)],
            # The precision would be then: (pred_positive == true_positive).sum()/(pred_CS != -1).sum().
            # Note that we divide by the total number of proteins predicted as having an SP.
            # Let me know if I need to explain this better.

            # The first thing that you want to do is subset proteins predicted as having a SP (pred_CS != -1) and also having a true SP
            true_positive = true_CS[(true_CS != -1) & (pred_CS != -1)]
            pred_positive = pred_CS[(true_CS != -1) & (pred_CS != -1)]

            for window_size in [0, 1, 2, 3]:
                correct_in_window = pred_positive == true_positive  # window 0
                correct_in_window = (pred_positive >= true_positive - window_size) & (
                    pred_positive <= true_positive + window_size
                )
                precision = correct_in_window.sum() / (pred_CS != -1).sum()
                recall = correct_in_window.sum() / (true_CS != -1).sum()

                metrics_dict[
                    f"{kingdom}_{rev_label_dict[sp_type]}_recall_window_{window_size}"
                ] = recall
                metrics_dict[
                    f"{kingdom}_{rev_label_dict[sp_type]}_precision_window_{window_size}"
                ] = precision

    return metrics_dict


def get_metrics(
    model,
    data=Union[Tuple[np.ndarray, np.ndarray, np.ndarray], torch.utils.data.DataLoader],
    cs_tagged=False,
    sp_tokens=None,
):

    if sp_tokens is None:
        sp_tokens = [3, 7, 11] if model.use_large_crf else [0, 4, 5]
    print(f"Using SP tokens {sp_tokens} to infer cleavage site.")

    # handle array datasets (small organism test sets)
    if type(data) == tuple:
        data, global_targets, all_cs_targets = data
        all_global_probs, all_pos_preds = run_data_array(model, data)
        kingdom_ids = ["DEFAULT"] * len(
            global_targets
        )  # do not split by kingdoms, make dummy kingdom
    # handle dataloader (signalp data)
    else:
        all_global_targets, all_global_probs, all_targets, all_pos_preds = run_data(
            model, data
        )

        if cs_tagged:
            all_cs_targets = find_cs_tag(all_targets)
        else:
            all_cs_targets = tagged_seq_to_cs_multiclass(
                all_targets, sp_tokens=sp_tokens
            )

        # parse kingdom ids
        parsed = [
            element.lstrip(">").split("|") for element in data.dataset.identifiers
        ]
        acc_ids, kingdom_ids, type_ids, partition_ids = [
            np.array(x) for x in list(zip(*parsed))
        ]

    all_global_preds = all_global_probs.argmax(axis=1)
    if cs_tagged:
        all_cs_preds = find_cs_tag(all_pos_preds)
    else:
        all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens=sp_tokens)
    metrics = compute_metrics(
        all_global_targets,
        all_global_preds,
        all_cs_targets,
        all_cs_preds,
        all_kingdom_ids=kingdom_ids,
    )

    return metrics


def get_region_metrics(all_global_targets, all_pos_preds, all_input_ids, sp_lengths):

    # NOTE skip indices are hardcoded
    # NOTE only report set means, not per kingdom yet

    n_h, h_c = class_aware_cosine_similarities(
        all_pos_preds,
        all_input_ids,
        all_global_targets,
        replace_value=np.nan,
        op_mode="numpy",
    )
    n_lengths, h_lengths, c_lengths = get_region_lengths(
        all_pos_preds, all_global_targets, agg_fn="none"
    )
    n_fracs, h_fracs, c_fracs = get_region_lengths(
        all_pos_preds, all_global_targets, agg_fn="none", sp_lengths=sp_lengths
    )  # normalized by sp length

    metrics_dict = {}
    for label in np.unique(all_global_targets):

        if label == 0 or label == 5:
            continue

        # dirty way to skip invalid regions -> all nan due to masking in class_aware_cosine_similarities, so nanmean is also nan
        cos_nh = np.nanmean(n_h[all_global_targets == label])
        if cos_nh != np.nan:
            metrics_dict[f"Cosine similarity nh {label}"] = cos_nh
        cos_hc = np.nanmean(h_c[all_global_targets == label])
        if cos_hc != np.nan:
            metrics_dict[f"Cosine similarity hc {label}"] = cos_hc

        metrics_dict[f"Average length n {label}"] = n_lengths[
            all_global_targets == label
        ].mean()
        metrics_dict[f"Average length h {label}"] = h_lengths[
            all_global_targets == label
        ].mean()
        metrics_dict[f"Average length c {label}"] = c_lengths[
            all_global_targets == label
        ].mean()

        metrics_dict[f"Average SP fraction n {label}"] = n_fracs[
            all_global_targets == label
        ].mean()
        metrics_dict[f"Average SP fraction h {label}"] = h_fracs[
            all_global_targets == label
        ].mean()
        metrics_dict[f"Average SP fraction c {label}"] = c_fracs[
            all_global_targets == label
        ].mean()
        # w&b can plot histogram heatmaps over time when logging sequences
        metrics_dict[f"Lengths n {label}"] = n_lengths[all_global_targets == label]
        metrics_dict[f"Lengths h {label}"] = h_lengths[all_global_targets == label]
        metrics_dict[f"Lengths c {label}"] = c_lengths[all_global_targets == label]

        metrics_dict[f"SP fraction n {label}"] = n_fracs[all_global_targets == label]
        metrics_dict[f"SP fraction h {label}"] = h_fracs[all_global_targets == label]
        metrics_dict[f"SP fraction c {label}"] = c_fracs[all_global_targets == label]

    return metrics_dict


def get_metrics_multistate(
    model,
    data=Union[Tuple[np.ndarray, np.ndarray, np.ndarray], torch.utils.data.DataLoader],
    rev_label_dict: Dict[int, str] = REVERSE_GLOBAL_LABEL_DICT,
    sp_tokens=None,
    compute_region_metrics=True,
):
    """Works with multi-state CRF and corresponding RegionDataset"""

    if sp_tokens is None:
        sp_tokens = [5, 11, 19, 26, 31]

    print(f"Using SP tokens {sp_tokens} to infer cleavage site.")

    # handle array datasets (small organism test sets)
    if type(data) == tuple:
        data, all_global_targets, all_cs_targets = data
        all_global_probs, all_pos_preds = run_data_array(model, data)
        kingdom_ids = ["DEFAULT"] * len(
            all_global_targets
        )  # do not split by kingdoms, make dummy kingdom for compatibility with same functions
    # handle dataloader (signalp data)
    else:
        (
            all_input_ids,
            all_global_targets,
            all_global_probs,
            all_targets,
            all_pos_preds,
            all_cs,
        ) = run_data_regioncrfdataset(model, data)
        all_cs_targets = all_cs

        # parse kingdom ids
        parsed = [
            element.lstrip(">").split("|") for element in data.dataset.identifiers
        ]
        acc_ids, kingdom_ids, type_ids, partition_ids = [
            np.array(x) for x in list(zip(*parsed))
        ]

    all_global_preds = all_global_probs.argmax(axis=1)
    print(all_global_preds)

    all_cs_preds = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens=sp_tokens)
    metrics = compute_metrics(
        all_global_targets,
        all_global_preds,
        all_cs_targets,
        all_cs_preds,
        all_kingdom_ids=kingdom_ids,
        rev_label_dict=rev_label_dict,
    )
    if compute_region_metrics:
        region_metrics = get_region_metrics(
            all_global_targets, all_pos_preds, all_input_ids, sp_lengths=all_cs
        )
        metrics.update(region_metrics)

    return metrics
