# © Copyright Technical University of Denmark
"""
Compute the cosine similarity between n,h,c regions.
This is used for evaluation.

Note that all the parameters are hardcoded and refer to the Bert-CRF multi-tag setup.

For legacy reasons, this is implemented in pytorch.
(wanted to create a regularization term, ended up doing this differently,
this can still be used for evaluation by itself.)
"""
from typing import Tuple, Union, List
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AA_SPECIAL_TOKENS = [
    0,
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
]  # these are ignored in distance
TAG_PAD_IDX = -1
MAX_LEN_AA_TOKENS = 30  # regular AAs + special AAs
# to remove irrelevant indices after bincount
AA_TOKEN_START_IDX = 5  # first aa token
AA_TOKEN_END_IDX = 24  # last aa token

# SIGNALP6_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3, 'TATLIPO':4, 'PILIN':5}
N_H_IGNORE_CLASSES = [0, 3, 4, 5]
H_C_IGNORE_CLASSES = [0, 2, 4, 5]

DEFAULT_N_TAGS = [3, 9, 16, 23]
DEFAULT_H_TAGS = [4, 10, 18, 25, 33]
DEFAULT_C_TAGS = [5, 19]


def torch_isin(
    element: torch.LongTensor, test_elements: Union[torch.LongTensor, List[int]]
) -> torch.BoolTensor:
    """torch equivalent of np.isin()
    element is True when value at this position is in test_elements"""

    if type(test_elements) == list:
        test_elements = torch.tensor(test_elements).to(device)

    bool_tensor = (element.unsqueeze(-1) == test_elements).any(-1)

    return bool_tensor


def compute_region_cosine_similarity(
    tag_sequences: Union[List[torch.LongTensor], torch.LongTensor],
    aa_sequences: Union[List[torch.LongTensor], torch.LongTensor],
    n_region_tags: List[int] = None,
    h_region_tags: List[int] = None,
    c_region_tags: List[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the cosine similiarities of sequence regions as defined by the tags.
    This function is not aware of ground truth sequence type/global label,
    which means that also e.g. LIPO SPs will have their h-c similarity calculated.
    These need to be replaced with 0/Nan afterwards as needed.

    Inputs:
        tag_sequences: (n_samples, seq_len) tensor
                        or list of (seq_len,) tensors with AA token indices
        aa_sequences: tokenize: (n_samples, seq_len) tensor
                        or list of (seq_len,) tensors with region label indices
        n_region_tags: list of label indices that are considered n region
        h_region_tags: list of label indices that are considered h region
        c_region_tags: list of label indices that are considered c region

    Returns:
        (n_samples,) tensor of n-h similarities
        (n_samples,) tensor of h-c similarities

    """
    if n_region_tags is None:
        n_region_tags = DEFAULT_N_TAGS
    if h_region_tags is None:
        h_region_tags = DEFAULT_H_TAGS
    if c_region_tags is None:
        c_region_tags = DEFAULT_C_TAGS

    # check shape of arrays
    assert len(tag_sequences) == len(
        aa_sequences
    ), "Need same number of tag and amino acid sequences"

    n_h_similarities = []
    h_c_similarities = []

    for i in range(len(tag_sequences)):

        tags = tag_sequences[i]
        aas = aa_sequences[i]

        # Remove special tokens
        aas = aas[~torch_isin(aas, AA_SPECIAL_TOKENS)]
        tags = tags[tags != TAG_PAD_IDX]

        # Preprocess tags and sequences
        tags = tags[1:]  # skip first M
        aas = aas[1:]  # skip first M

        assert len(tags) == len(aas)

        # Whenever we are normalizing, there is a risk that we divide by sum 0,
        # especially at early epochs. Add eps to prevent nan

        # Get n region
        n_idx = torch_isin(tags, n_region_tags)

        n_aas = aas[n_idx]
        n_aa_counts = torch.bincount(n_aas, minlength=MAX_LEN_AA_TOKENS)
        n_aa_counts = n_aa_counts[AA_TOKEN_START_IDX : AA_TOKEN_END_IDX + 1]
        n_aa_freq = n_aa_counts.float() / (n_aa_counts.sum() + 1e-6)

        # Get h region
        h_idx = torch_isin(tags, h_region_tags)

        h_aas = aas[h_idx]
        h_aa_counts = torch.bincount(h_aas, minlength=MAX_LEN_AA_TOKENS)
        h_aa_counts = h_aa_counts[AA_TOKEN_START_IDX : AA_TOKEN_END_IDX + 1]
        h_aa_freq = h_aa_counts.float() / (h_aa_counts.sum() + 1e-6)

        # Get c region
        c_idx = torch_isin(tags, c_region_tags)

        c_aas = aas[c_idx]
        c_aa_counts = torch.bincount(c_aas, minlength=MAX_LEN_AA_TOKENS).float()
        c_aa_counts = c_aa_counts[AA_TOKEN_START_IDX : AA_TOKEN_END_IDX + 1]
        c_aa_freq = c_aa_counts.float() / (c_aa_counts.sum() + 1e-6)

        # Compute n vs h
        n_h_similarity = torch.nn.functional.cosine_similarity(n_aa_freq, h_aa_freq, 0)
        n_h_similarities.append(n_h_similarity)

        # Compute h vs c
        h_c_similarity = torch.nn.functional.cosine_similarity(h_aa_freq, c_aa_freq, 0)
        h_c_similarities.append(h_c_similarity)

    return torch.stack(n_h_similarities), torch.stack(h_c_similarities)


def class_aware_cosine_similarities(
    tag_sequences: Union[List[torch.LongTensor], torch.LongTensor, np.ndarray],
    aa_sequences: Union[List[torch.LongTensor], torch.LongTensor, np.ndarray],
    class_labels: Union[torch.LongTensor, np.ndarray],
    n_region_tags: List[int] = None,
    h_region_tags: List[int] = None,
    c_region_tags: List[int] = None,
    replace_value: float = 0,
    op_mode: str = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for `compute_region_cosine_similiarity`
    Takes care of post-processing kingdoms"""

    assert not (
        op_mode == "torch" and replace_value == np.nan
    ), "Cannot use nan when working in torch"

    if op_mode == "torch":
        n_h_masking_indices = torch_isin(class_labels, N_H_IGNORE_CLASSES)
        h_c_masking_indices = torch_isin(class_labels, H_C_IGNORE_CLASSES)
        n_h_similarities, h_c_similarities = compute_region_cosine_similarity(
            tag_sequences, aa_sequences, n_region_tags, h_region_tags, c_region_tags
        )

    elif op_mode == "numpy":
        tag_sequences = torch.tensor(tag_sequences).to(device)
        aa_sequences = torch.tensor(aa_sequences).to(device)

        n_h_masking_indices = np.isin(class_labels, N_H_IGNORE_CLASSES)
        h_c_masking_indices = np.isin(class_labels, H_C_IGNORE_CLASSES)
        with torch.no_grad():
            n_h_similarities, h_c_similarities = compute_region_cosine_similarity(
                tag_sequences, aa_sequences, n_region_tags, h_region_tags, c_region_tags
            )
            n_h_similarities = n_h_similarities.detach().cpu().numpy()
            h_c_similarities = h_c_similarities.detach().cpu().numpy()

    else:
        raise NotImplementedError("Valid op_modes are  `torch` and `numpy`")

    # mask
    n_h_similarities[n_h_masking_indices] = replace_value
    h_c_similarities[h_c_masking_indices] = replace_value

    return n_h_similarities, h_c_similarities


# SIGNALP6_GLOBAL_LABEL_DICT = {'NO_SP':0, 'SP':1,'LIPO':2, 'TAT':3, 'TATLIPO':4, 'PILIN':5}
NO_N_CLASSES = [0, 5]
NO_H_CLASSES = [0, 5]
NO_C_CLASSES = [0, 2, 4, 5]


def get_region_start_end(tag_sequence, token_ids: List[int]):
    """end_idx defined as last idx that has this value.
    Add +1 when [start, end] indexing."""

    if type(tag_sequence) == np.ndarray:
        tag_sequence = list(tag_sequence)

    start_idx = 1000
    end_idx = -1
    for token_id in token_ids:
        try:
            start = tag_sequence.index(token_id)
            end = len(tag_sequence) - 1 - tag_sequence[::-1].index(token_id)

            start_idx = min(start_idx, start)
            end_idx = max(end_idx, end)
        # .index() throws error when no match is found.
        except ValueError:
            pass

    if start_idx == 1000 or end_idx == -1:
        return 0, 0
    else:
        return start_idx, end_idx


def get_region_lengths(
    tag_sequences: Union[List[np.ndarray], List[List[int]], np.ndarray],
    class_labels: np.ndarray,
    sp_lengths: np.ndarray = None,
    n_region_tags: List[int] = None,
    h_region_tags: List[int] = None,
    c_region_tags: List[int] = None,
    no_n_classes: List[int] = None,
    no_h_classes: List[int] = None,
    no_c_classes: List[int] = None,
    agg_fn="mean",
):
    """Calculate the mean length of each region.
    Only counts tags in a sequence, does not check whether the tags are contiguous. Assume CRF enforces that anyway.
    Inputs:
        tag_sequences: viterbi path of each sequence
        class_labels: global label of each sequence
        sp_lengths: length of sp of each sequence. If not None, used to normalize region lengths by sp length
        {n,h,c}_region_tags : list of tag indices belonging to a region
        no_{n,h,c}_classes  : list of global label indices that don't have the region
        agg_fn : 'mean' - mean over all seqs, 'none' return arrays of length (n_sequences)

    """

    n_region_tags = DEFAULT_N_TAGS if n_region_tags is None else n_region_tags
    h_region_tags = DEFAULT_H_TAGS if h_region_tags is None else h_region_tags
    c_region_tags = DEFAULT_C_TAGS if c_region_tags is None else c_region_tags

    no_n_classes = NO_N_CLASSES if no_n_classes is None else no_n_classes
    no_h_classes = NO_H_CLASSES if no_h_classes is None else no_h_classes
    no_c_classes = NO_C_CLASSES if no_c_classes is None else no_c_classes

    # count number of region tags in each sequence
    if type(tag_sequences) == np.ndarray:

        n_tags = np.isin(tag_sequences, n_region_tags).sum(axis=1)
        h_tags = np.isin(tag_sequences, h_region_tags).sum(axis=1)
        c_tags = np.isin(tag_sequences, c_region_tags).sum(axis=1)

    elif type(tag_sequences) == list:

        n_tags = np.array(
            [np.isin(np.array(x), tag_sequences).sum() for x in tag_sequences]
        )
        h_tags = np.array(
            [np.isin(np.array(x), tag_sequences).sum() for x in tag_sequences]
        )
        c_tags = np.array(
            [np.isin(np.array(x), tag_sequences).sum() for x in tag_sequences]
        )

    computation_mode = "x"
    if computation_mode == "region_borders":
        n_tags = []
        h_tags = []
        c_tags = []
        for seq in tag_sequences:
            start, end = get_region_start_end(seq, n_region_tags)
            n_tags.append(end + 1 - start)
            start, end = get_region_start_end(seq, h_region_tags)
            h_tags.append(end + 1 - start)
            start, end = get_region_start_end(seq, c_region_tags)
            c_tags.append(end + 1 - start)

        n_tags = np.array(n_tags)
        h_tags = np.array(h_tags)
        c_tags = np.array(c_tags)

    if sp_lengths is not None:
        assert len(sp_lengths) == len(
            tag_sequences
        ), "lengths of input arrays don't match"

        n_tags = n_tags / sp_lengths
        h_tags = h_tags / sp_lengths
        c_tags = c_tags / sp_lengths

    # mean
    if agg_fn == "mean":
        # we only apply mask when taking the mean. when no aggregation, return unmasked so indices match.
        n_mask = np.isin(class_labels, no_n_classes)
        h_mask = np.isin(class_labels, no_h_classes)
        c_mask = np.isin(class_labels, no_c_classes)

        n_mean_length = n_tags[~n_mask].mean()
        h_mean_length = h_tags[~h_mask].mean()
        c_mean_length = c_tags[~c_mask].mean()

        return n_mean_length, h_mean_length, c_mean_length

    elif agg_fn == "none":
        return n_tags, h_tags, c_tags

    else:
        raise NotImplementedError(agg_fn)
