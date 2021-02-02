"""
Write the whole cross-validated averaged model as a single module
Probably very big, but let's see if it hits the memory limit.

This transformers branch supports scripting of Bert, but could not make it work:
https://github.com/sbrody18/transformers
work around by tracing (fixes sequence length though, so always pad to use the model)
"""
import pandas as pd
import numpy as np
from signalp6.models import ProteinBertTokenizer
from signalp6.training_utils import RegionCRFDataset


from typing import List
import torch
from .multi_tag_crf_for_viterbi import CRF
from .bert_crf_for_export import BertSequenceTaggingCRF


class EnsembleBertCRFModel(torch.nn.Module):
    def __init__(self, bert_checkpoints, crf_checkpoint):
        super().__init__()

        self.berts = torch.nn.ModuleList(
            [BertSequenceTaggingCRF.from_pretrained(ckp) for ckp in bert_checkpoints]
        )

        # self.crf =  CRF()
        # pull CRF config from BertSequenceTaggingCRF config
        self.crf = CRF(
            num_tags=self.berts[0].config.num_labels,
            batch_first=True,
            allowed_transitions=self.berts[0].config.allowed_crf_transitions,
            allowed_start=self.berts[0].config.allowed_crf_starts,
            allowed_end=self.berts[0].config.allowed_crf_ends,
        )

        # get CRF weights from berts and average
        start_transitions = [x.crf.start_transitions for x in self.berts]
        transitions = [x.crf.transitions for x in self.berts]
        end_transitions = [x.crf.end_transitions for x in self.berts]

        start_transitions = torch.stack(start_transitions).mean(dim=0)
        transitions = torch.stack(transitions).mean(dim=0)
        end_transitions = torch.stack(end_transitions).mean(dim=0)

        self.crf.start_transitions.data = start_transitions
        self.crf.transitions.data = transitions
        self.crf.end_transitions.data = end_transitions

        print("Initalized model and averaged weights for viterbi")

    def forward(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        # get global probs, sequence probs and emissions from berts
        futures = [torch.jit.fork(model, input_ids, input_mask) for model in self.berts]
        results = [torch.jit.wait(fut) for fut in futures]
        print("bert fwd passes done")

        # results is list of (global_prob, seq_prob, emissions) tuples
        global_probs, marginal_probs, emissions = zip(*results)

        global_probs_mean = torch.stack(global_probs).mean(dim=0)
        marginal_probs_mean = torch.stack(marginal_probs).mean(dim=0)
        emissions_mean = torch.stack(emissions).mean(dim=0)

        viterbi_paths = self.crf(emissions_mean, input_mask.byte())
        print("viterbi paths done")

        return global_probs_mean, marginal_probs_mean, viterbi_paths
