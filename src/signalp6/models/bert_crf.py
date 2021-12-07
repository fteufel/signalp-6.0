# © Copyright Technical University of Denmark
import torch
import torch.nn as nn
import sys
from .multi_tag_crf import CRF
from typing import Tuple
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
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


class ProteinBertTokenizer:
    """Wrapper class for Huggingface BertTokenizer.
    implements an encode() method that takes care of
    - putting spaces between AAs
    - prepending the kingdom id token,if kingdom id is provided and vocabulary allows it
    - prepending the label token, if provided and vocabulary allows it. label token is used when
      predicting the CS conditional on the known class.
    """

    def __init__(self, *args, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(*args, **kwargs)

    def encode(self, sequence, kingdom_id=None, label_id=None):
        # Preprocess sequence to ProtTrans format
        sequence = " ".join(sequence)

        prepro = re.sub(r"[UZOB]", "X", sequence)

        if kingdom_id is not None and self.tokenizer.vocab_size > 30:
            prepro = kingdom_id.upper() + " " + prepro
        if (
            label_id is not None and self.tokenizer.vocab_size > 34
        ):  # implies kingdom is also used.
            prepro = (
                label_id.upper().replace("_", "") + " " + prepro
            )  # HF tokenizers split at underscore, can't have taht in vocab
        return self.tokenizer.encode(prepro)

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        return cls(checkpoint, **kwargs)


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

        if self.use_kingdom_id:
            self.kingdom_embedding = nn.Embedding(4, config.kingdom_embed_size)

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

        self.crf_input_length = 70  # TODO make this part of config if needed. Now it's for cases where I don't control that via input data or labels.

        ## Hidden states to CRF emissions
        self.outputs_to_emissions = nn.Linear(
            config.hidden_size
            if self.use_kingdom_id is False
            else config.hidden_size + config.kingdom_embed_size,
            config.num_labels,
        )

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
        self.use_large_crf = True  # legacy for get_metrics, no other use.

        ## Loss scaling parameters
        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        kingdom_ids=None,
        input_mask=None,
        targets=None,
        targets_bitmap=None,
        global_targets=None,
        inputs_embeds=None,
        sample_weights=None,
        return_emissions=False,
        force_states=False,
    ):
        """Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 kingdom_ids (batch_size) :  [0,1,2,3] for eukarya, gram_positive, gram_negative, archaea
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
                 global_targets (batch_size)
                 input_mask (batch_size, seq_len). binary tensor, 0 at padded positions
                 input_embeds: Optional instead of input_ids. Start with embedded sequences instead of token ids.
                 sample_weights (batch_size) float tensor. weight for each sequence to be used in cross-entropy.
                 return_emissions : return the emissions and masks for the CRF. used when averaging viterbi decoding.


        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if targets is not None and targets_bitmap is not None:
            raise ValueError(
                "You cannot specify both targets and targets_bitmap at the same time"
            )

        ## Get LM hidden states
        outputs = self.bert(
            input_ids, attention_mask=input_mask, inputs_embeds=inputs_embeds
        )  # Returns tuple. pos 0 is sequence output, rest optional.

        sequence_output = outputs[0]

        ## Remove special tokens
        sequence_output, input_mask = self._trim_transformer_output(
            sequence_output, input_mask
        )  # this takes care of CLS and SEP, pad-aware
        if self.kingdom_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None
        if self.type_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None

        ## Trim transformer output to length of targets or to crf_input_length
        if targets is not None:
            sequence_output = sequence_output[
                :, : targets.shape[1], :
            ]  # this removes extra residues that don't go to CRF
            input_mask = (
                input_mask[:, : targets.shape[1]] if input_mask is not None else None
            )
        else:
            sequence_output = sequence_output[:, : self.crf_input_length, :]
            input_mask = (
                input_mask[:, : self.crf_input_length]
                if input_mask is not None
                else None
            )

        ## Apply dropouts
        sequence_output = self.lm_output_dropout(sequence_output)

        ## Add kingdom ids
        if self.use_kingdom_id == True:
            ids_emb = self.kingdom_embedding(kingdom_ids)  # batch_size, embed_size
            ids_emb = ids_emb.unsqueeze(1).repeat(
                1, sequence_output.shape[1], 1
            )  # batch_size, seq_len, embed_size
            sequence_output = torch.cat([sequence_output, ids_emb], dim=-1)

        ## CRF emissions
        prediction_logits = self.outputs_to_emissions(sequence_output)

        ## CRF
        if targets is not None:
            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=targets,
                tag_bitmap=None,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        elif targets_bitmap is not None:

            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=None,
                tag_bitmap=targets_bitmap,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        else:
            neg_log_likelihood = 0

        probs = self.crf.compute_marginal_probabilities(
            emissions=prediction_logits, mask=input_mask.byte()
        )

        if self.sp_region_tagging:
            global_probs = self.compute_global_labels_multistate(probs, input_mask)
        else:
            global_probs = self.compute_global_labels(probs, input_mask)

        global_log_probs = torch.log(global_probs)

        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)

        # TODO update init_states generation to new n,h,c states and actually start using it
        # from preds, make initial sequence label vector
        if force_states:
            init_states = self.inital_state_labels_from_global_labels(preds)
        else:
            init_states = None
        viterbi_paths = self.crf.decode(
            emissions=prediction_logits,
            mask=input_mask.byte(),
            init_state_vector=init_states,
        )

        # pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1] * (max_pad_len - len(x)) for x in viterbi_paths]
        pos_preds = torch.tensor(
            pos_preds, device=probs.device
        )  # NOTE convert to tensor just for compatibility with the else case, so always returns same type

        outputs = (global_probs, probs, pos_preds)  # + outputs

        # get the losses
        losses = neg_log_likelihood

        if global_targets is not None:
            loss_fct = nn.NLLLoss(
                ignore_index=-1,
                reduction="none" if sample_weights is not None else "mean",
            )
            global_loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels),
                global_targets.view(-1),
            )

            if sample_weights is not None:
                global_loss = global_loss * sample_weights
                global_loss = global_loss.mean()

            losses = losses + global_loss

        if (
            targets is not None
            or global_targets is not None
            or targets_bitmap is not None
        ):

            outputs = (losses,) + outputs  # loss, global_probs, pos_probs, pos_preds

        ## Return emissions
        if return_emissions:
            outputs = outputs + (
                prediction_logits,
                input_mask,
            )  # (batch_size, seq_len, num_labels)

        return outputs

    @staticmethod
    def _trim_transformer_output(hidden_states, input_mask):
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

    def compute_global_labels(self, probs, mask):
        """Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.
        If num_global_labels is 2, assume we deal with the sp-no sp case.
        """
        # probs = b_size x seq_len x n_states tensor
        # Yes, each SP type will now have 4 labels in the CRF. This means that now you only optimize the CRF loss, nothing else.
        # To get the SP type prediction you have two alternatives. One is to use the Viterbi decoding,
        # if the last position is predicted as SPI-extracellular, then you know it is SPI protein.
        # The other option is what you mention, sum the marginal probabilities, divide by the sequence length and then sum
        # the probability of the labels belonging to each SP type, which will leave you with 4 probabilities.
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

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
            tat = global_probs[:, 11:15].sum(dim=1)
            tat_spi = global_probs[:, 15:19].sum(dim=1)
            spiii = global_probs[:, 19:].sum(dim=1)

            # When using extra state for CS, different indexing

            # if self.num_labels == 18:
            #    spi = global_probs[:, 3:8].sum(dim =1)
            #    spii = global_probs[:, 8:13].sum(dim =1)
            #    tat = global_probs[:,13:].sum(dim =1)

            return torch.stack([no_sp, spi, spii, tat, tat_spi, spiii], dim=-1)

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

        # if self.sp2_only:
        #    no_sp = global_probs[:,0:3].sum(dim=1)
        #    spii = global_probs[:,3:].sum(dim=1)
        #    return torch.stack([no_sp,spii], dim=-1)

        # else:
        #    no_sp = global_probs[:,0:3].sum(dim=1)
        #    spi = global_probs[:,3:9].sum(dim=1)
        #    spii = global_probs[:,9:16].sum(dim=1)
        #    tat = global_probs[:,16:23].sum(dim=1)
        #    lipotat = global_probs[:, 23:30].sum(dim=1)
        #    spiii = global_probs[:,30:].sum(dim=1)

        #    return torch.stack([no_sp,spi,spii,tat,lipotat,spiii], dim=-1)

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
            probs = probs * weights
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
