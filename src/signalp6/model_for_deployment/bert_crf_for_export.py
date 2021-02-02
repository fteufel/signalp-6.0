"""
This is mostly the same model as in /models/multi_crf_bert.

Here we adapt it to remove uncessary parameters and arguments for the final prediction model,
and make it compatible with ONNX export.

Things that change
 - removed arguments
 - removed conditions that will never be true (kingdom ids, loss)
 - removed kingdom id stuff (comes in via input ids now always)
 - do not compute viterbi path in here. That will be another model on top 
   in the final predictor.
 - always return emissions

Model is traceable in general when we stick to a sequence length of 70. Then all for loops can 
be unrolled (except for viterbi decoding).
Exceptions:

_trim_transformer_output:
is incompatible with tracing, iterates over batch_size. 
moved out of class, jitting doesn't like class methods.
Refactored to make loop-free
"""


import torch
import torch.nn as nn
from .multi_tag_crf_for_probs import CRF as CRF_probs
from .multi_tag_crf_for_viterbi import CRF as CRF_viterbi
from typing import Tuple
from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel
import re


class SequenceDropout(nn.Module):
    """Layer zeroes full hidden states in a sequence of hidden states"""

    def __init__(self, p=0.1, batch_first=True):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x

        if not self.batch_first:
            x = x.transpose(0, 1)
        # make dropout mask
        mask = torch.ones(x.shape[0], x.shape[1], dtype=x.dtype).bernoulli(
            1 - self.dropout
        )  # batch_size x seq_len
        # expand
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 16)
        # multiply
        after_dropout = mask_expanded * x
        if not self.batch_first:
            after_dropout = after_dropout.transpose(0, 1)

        return after_dropout


# @torch.jit.script
def _trim_transformer_output_old(hidden_states, input_mask):
    """Helper function to remove CLS, SEP tokens after passing through transformer"""

    # remove CLS
    hidden_states = hidden_states[:, 1:, :]

    if input_mask is not None:

        input_mask = input_mask[:, 1:]
        # remove SEP - hidden states are padded at end!
        true_seq_lens = input_mask.sum(dim=1) - 1  # -1 for SEP

        mask_list = []
        output_list = []
        for i in range(input_mask.shape[0]):
            mask_list.append(input_mask[i, : true_seq_lens[i]])
            output_list.append(hidden_states[i, : true_seq_lens[i], :])

        mask_out = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
        hidden_out = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True)
    else:
        hidden_out = hidden_states[:, :-1, :]
        mask_out = None

    return hidden_out, mask_out


def _trim_transformer_output(hidden_states, input_mask):
    """Take advantage of multiplications to avoid lists and looping.
    Have: Input: [CLS].....[SEP][PAD]...[PAD]
          Mask:    1  11111  1    1       1
    Aim: remove 2 tokens from each sequence, and if sep is not the last token zero it.
    input_mask is never none.

    The logic of this is not intuitive compared to cutting and looping,
    but its a tensor-only operation without dynamic indexing.
    """
    # remove cls
    hidden_states = hidden_states[:, 1:, :]
    input_mask = input_mask[:, 1:]
    # print(hidden_states.shape)
    # print(input_mask.shape)
    # print('Before processing, cls removed.')

    # shift input mask by one. the last pos will be zero, and all others are shifted to
    # the left by one. In full-length seqs, this makes the [SEP] token 0. In padded seqs.
    # this shifts the first 0 from the first [PAD] to [SEP]
    shifted_input_mask = input_mask[:, 1:]
    # add the zero vector at the end
    zeros = (
        input_mask[:, 1] * 0
    )  # make zero vector directly from input tensor, don't compute shapes
    shifted_input_mask = torch.cat(
        [shifted_input_mask, zeros.unsqueeze(1)], dim=1
    )  # add dummy seq dim and concatenate along seq dim.
    # print('made mask')
    # print(shifted_input_mask.shape)

    # use the new input mask to make SEP tokens 0
    hidden_states = hidden_states * shifted_input_mask.unsqueeze(-1)

    # now the last pos is zero for all, can drop it.
    hidden_out = hidden_states[:, :-1, :]
    # do the same to the shifted mask
    mask_out = shifted_input_mask[:, :-1]

    return hidden_states, mask_out


# This here is to compute global labels from CRF states
SIGNALP6_CLASS_LABEL_MAP = [
    [0, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22],
    [23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
]


class BertSequenceTaggingCRF(BertPreTrainedModel):
    """Sequence tagging and global label prediction model (like SignalP).
    LM output goes through a linear layer with classifier_hidden_size before being projected to num_labels outputs.
    These outputs then either go into the CRF as emissions, or to softmax as direct probabilities.
    config.use_crf controls this.

    Inputs are batch first.
       Loss is sum between global sequence label crossentropy and position wise tags crossentropy.
       Optionally use CRF.

    """

    def __init__(self, config):
        super().__init__(config)

        ## Set up kingdom ID embedding layer if used
        self.use_kingdom_id = (
            config.use_kingdom_id if hasattr(config, "use_kingdom_id") else False
        )

        ## Set up LM and hidden state postprocessing
        self.bert = BertModel(config=config)
        self.lm_output_dropout = nn.Dropout(
            config.lm_output_dropout if hasattr(config, "lm_output_dropout") else 0
        )  # for backwards compatbility
        self.lm_output_position_dropout = SequenceDropout(
            config.lm_output_position_dropout
            if hasattr(config, "lm_output_position_dropout")
            else 0
        )
        self.kingdom_id_as_token = (
            config.kingdom_id_as_token
            if hasattr(config, "kingdom_id_as_token")
            else False
        )  # used for truncating hidden states
        self.type_id_as_token = (
            config.type_id_as_token if hasattr(config, "type_id_as_token") else False
        )

        self.crf_input_length = 70

        ## Hidden states to CRF emissions
        self.outputs_to_emissions = nn.Linear(config.hidden_size, config.num_labels)

        ## Set up CRF
        self.num_global_labels = (
            config.num_global_labels
            if hasattr(config, "num_global_labels")
            else config.num_labels
        )
        self.num_labels = config.num_labels
        self.class_label_mapping = (
            config.class_label_mapping
            if hasattr(config, "class_label_mapping")
            else SIGNALP6_CLASS_LABEL_MAP
        )
        assert (
            len(self.class_label_mapping) == self.num_global_labels
        ), "defined number of classes and class-label mapping do not agree."

        self.allowed_crf_transitions = (
            config.allowed_crf_transitions
            if hasattr(config, "allowed_crf_transitions")
            else None
        )
        self.allowed_crf_starts = (
            config.allowed_crf_starts if hasattr(config, "allowed_crf_starts") else None
        )
        self.allowed_crf_ends = (
            config.allowed_crf_ends if hasattr(config, "allowed_crf_ends") else None
        )

        # This ad hoc flag allows to load two different CRF versions - the fwd pass either being probs or viterbi path
        # ONNX needs the fn to be compiled to be forward()
        # Being able to directly load is more convenient than pulling the weights from the pretrained model and
        # initializing a new CRF with the proper fwd method
        CRF = CRF_viterbi if hasattr(config, "load_viterbi_fwd_crf") else CRF_probs
        self.crf = CRF(
            num_tags=config.num_labels,
            batch_first=True,
            allowed_transitions=self.allowed_crf_transitions,
            allowed_start=self.allowed_crf_starts,
            allowed_end=self.allowed_crf_ends,
        )
        # Legacy, remove this once i completely retire non-mulitstate labeling
        self.sp_region_tagging = (
            config.use_region_labels if hasattr(config, "use_region_labels") else False
        )  # use the right global prob aggregation function
        self.use_large_crf = (
            True  # config.use_large_crf #TODO legacy for get_metrics, no other use.
        )

        ## Loss scaling parameters
        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

        self.init_weights()

    def forward(self, input_ids=None, input_mask=None, force_states=False):
        """Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 kingdom_ids (batch_size) :  [0,1,2,3] for eukarya, gram_positive, gram_negative, archaea
                 input_mask (batch_size, seq_len). binary tensor, 0 at padded positions
                 return_emissions : return the emissions and masks for the CRF. used when averaging viterbi decoding.


        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)
        """

        ## Get LM hidden states
        outputs = self.bert(
            input_ids, attention_mask=input_mask
        )  # Returns tuple. pos 0 is sequence output, rest optional.

        sequence_output = outputs[0]

        ## Remove special tokens
        sequence_output, input_mask = _trim_transformer_output(
            sequence_output, input_mask
        )  # this takes care of CLS and SEP, pad-aware
        if self.kingdom_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None
        if self.type_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None

        ## Trim transformer output to length of targets or to crf_input_length
        sequence_output = sequence_output[:, : self.crf_input_length, :]
        input_mask = (
            input_mask[:, : self.crf_input_length] if input_mask is not None else None
        )

        ## Apply dropouts
        sequence_output = self.lm_output_dropout(sequence_output)

        ## CRF emissions
        prediction_logits = self.outputs_to_emissions(sequence_output)

        ## CRF

        probs = self.crf.compute_marginal_probabilities(
            emissions=prediction_logits, mask=input_mask.byte()
        )

        global_probs = self.compute_global_labels_multistate(probs, input_mask)

        preds = probs.argmax(dim=1)

        # from preds, make initial sequence label vector
        # if force_states:
        #    init_states = self.inital_state_labels_from_global_labels(preds)
        # else:
        #    init_states = None

        outputs = (
            global_probs,
            probs,
            prediction_logits,
        )  # probs, marginal probs, emissions.

        return outputs

    def compute_global_labels(self, probs, mask):
        """Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.
        If num_global_labels is 2, assume we deal with the sp-no sp case.
        TODO refactor, implicit handling of eukarya-only and cs-state cases is hard to keep track of
        """
        # probs = b_size x seq_len x n_states tensor
        # Yes, each SP type will now have 4 labels in the CRF. This means that now you only optimize the CRF loss, nothing else.
        # To get the SP type prediction you have two alternatives. One is to use the Viterbi decoding,
        # if the last position is predicted as SPI-extracellular, then you know it is SPI protein.
        # The other option is what you mention, sum the marginal probabilities, divide by the sequence length and then sum
        # the probability of the labels belonging to each SP type, which will leave you with 4 probabilities.
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        # TODO check unsqueeze ops for division/multiplication broadcasting
        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        # aggregate
        no_sp = global_probs[:, 0:3].sum(dim=1)

        spi = global_probs[:, 3:7].sum(dim=1)

        if self.num_global_labels > 2:
            spii = global_probs[:, 7:11].sum(dim=1)
            tat = global_probs[:, 11:].sum(dim=1)

            # When using extra state for CS, different indexing
            if self.num_labels == 18:
                spi = global_probs[:, 3:8].sum(dim=1)
                spii = global_probs[:, 8:13].sum(dim=1)
                tat = global_probs[:, 13:].sum(dim=1)

            return torch.stack([no_sp, spi, spii, tat], dim=-1)

        else:
            return torch.stack([no_sp, spi], dim=-1)

    def compute_global_labels_multistate(self, probs, mask):
        """Aggregates probabilities for region-tagging CRF output"""
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        global_probs_list = []
        for class_indices in self.class_label_mapping:
            summed_probs = global_probs[:, class_indices].sum(dim=1)
            global_probs_list.append(summed_probs)

        return torch.stack(global_probs_list, dim=-1)

    def predict_global_labels(self, probs, kingdom_ids, weights=None):
        """Given probs from compute_global_labels, get prediction.
        Takes care of summing over SPII and TAT for eukarya, and allows reweighting of probabilities."""

        if self.use_kingdom_id:
            eukarya_idx = torch.where(kingdom_ids == 0)[0]
            summed_sp_probs = probs[eukarya_idx, 1:].sum(dim=1)
            # update probs for eukarya
            probs[eukarya_idx, 1] = summed_sp_probs
            probs[eukarya_idx, 2:] = 0

        # reweight
        if weights is not None:
            probs = (
                probs * weights
            )  # TODO check broadcasting, think should just work, 1 axis agrees
        # predict
        preds = probs.argmax(dim=1)

        return preds

    @staticmethod
    def inital_state_labels_from_global_labels(preds):

        initial_states = torch.zeros_like(preds)
        # update torch.where((testtensor==1) | (testtensor>0))[0] #this syntax would work.
        initial_states[preds == 0] = 0
        initial_states[preds == 1] = 3
        initial_states[preds == 2] = 9
        initial_states[preds == 3] = 16
        initial_states[preds == 4] = 23
        initial_states[preds == 5] = 31

        return initial_states
