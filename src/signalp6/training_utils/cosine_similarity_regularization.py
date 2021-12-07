# © Copyright Technical University of Denmark
"""
The cosine similarity can be used as a regularization term.

The problem with the code in region_similarity.py is that
our viterbi implementation is not differentiable. So we need to work
with marginal probabilities instead.

Algorithm:
First, we aggregate the probabilities of the different tags into n,h,c
(we sum them up at each position, according to the defined region membership of each CRF label, so that we get 3 numbers)
(n_samples, seq_len, n_labels) -> (n_samples, seq_len, 3)

We then transform these probabilities over the sequence length into 
a pseudo-count for each AA. So for each AA, we sum all positions in the sequence that have it. We do that for n,h and c each.
(n_samples, seq_len, 3) -> (n_samples, 20, 3)

We then have a pseudo-count for n, h and c. Normalize all pseudo-count vectors by their sum to get a pseudo-distribution, and compute cosine similarities of those.
(n_samples, 20, 3) -> (n_samples, 2)

"""
import torch
from typing import List, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_N_TAGS = [3, 9, 16, 23]
DEFAULT_H_TAGS = [4, 10, 18, 25, 33]
DEFAULT_C_TAGS = [5, 19]

AA_SPECIAL_TOKENS = [
    1,
    2,
    3,
    4,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
]  # these are removed before computation to ensure same length with CRF output
AA_IDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

SIGNALP6_REGION_LABEL_MAP = [
    [3, 9, 16, 23],  # n
    [4, 10, 18, 25, 33],  # h
    [5, 19],
]  # c

# SIGNALP6_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3, 'TATLIPO':4, 'PILIN':5}
N_H_IGNORE_CLASSES = [0, 3, 4, 5]
H_C_IGNORE_CLASSES = [0, 2, 4, 5]


def torch_isin(
    element: torch.LongTensor, test_elements: Union[torch.LongTensor, List[int]]
) -> torch.BoolTensor:
    """torch equivalent of np.isin()
    element is True when value at this position is in test_elements"""

    if type(test_elements) == list:
        test_elements = torch.tensor(test_elements).to(device)

    bool_tensor = (element.unsqueeze(-1) == test_elements).any(-1)

    return bool_tensor


def aggregate_probs_per_region(
    probs: torch.Tensor, mask: torch.Tensor, label_region_map: List[List[int]] = None
):
    """Aggregates probabilities for region-tagging CRF output"""
    if mask is None:
        mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

    if label_region_map is None:
        label_region_map = SIGNALP6_REGION_LABEL_MAP

    probs = probs * mask.unsqueeze(-1)  # mask probs

    region_probs = []
    for region in label_region_map:
        # sum indices that belong to a region at each position
        region_sums = probs[:, :, region].sum(-1)
        region_probs.append(region_sums)

    region_probs = torch.stack(region_probs, -1)

    assert region_probs.shape == (probs.shape[0], probs.shape[1], len(label_region_map))
    return region_probs  # n_samples, seq_len, n_regions


def region_probs_to_aa_count(region_probs: torch.Tensor, input_ids=torch.LongTensor):

    # sum positions of each amino acid
    aa_sum_list = []
    for aa in AA_IDS:

        aa_sums = (region_probs * (input_ids == aa).unsqueeze(-1)).sum(
            1
        )  # n_samples, n_regions
        aa_sum_list.append(aa_sums)

    return torch.stack(aa_sum_list, 1)  # n_samples, n_amino_acids, n_regions


def compute_cosine_region_regularization(
    marginal_probs: torch.Tensor,
    input_ids: torch.Tensor,
    global_label_ids: torch.Tensor,
    mask: torch.Tensor = None,
    label_region_map=None,
):
    """
    Computes cosine similarities between pseudo amino acid counts of sp regions.
    """

    assert (
        marginal_probs.shape[:2] == input_ids.shape
    ), "First 2 dimensions of input ids and probs need to agree"
    assert len(global_label_ids) == len(
        input_ids
    ), "same number of global label ids as input ids needed"

    # get sums for each amino acid
    region_probs = aggregate_probs_per_region(marginal_probs, mask, label_region_map)
    aa_counts = region_probs_to_aa_count(region_probs, input_ids)

    # normalize
    aa_freqs = aa_counts / aa_counts.sum(dim=1).unsqueeze(1)

    # distances
    n_h_similarities = torch.nn.functional.cosine_similarity(
        aa_freqs[:, :, 0], aa_freqs[:, :, 1]
    )
    h_c_similarities = torch.nn.functional.cosine_similarity(
        aa_freqs[:, :, 1], aa_freqs[:, :, 2]
    )

    # mask out invalid comparisons
    n_h_masking_indices = torch_isin(global_label_ids, N_H_IGNORE_CLASSES)
    h_c_masking_indices = torch_isin(global_label_ids, H_C_IGNORE_CLASSES)

    n_h_similarities[n_h_masking_indices] = 0
    h_c_similarities[h_c_masking_indices] = 0

    return n_h_similarities, h_c_similarities
