"""
Had to move model for viterbi decoding into own file because
ONNX requires the fn to be compiled to be forward().
Normally, we access viterbi by calling decode().
Removed superfluous functions and renamed decode forward.
"""
__version__ = "0.7.2"

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ONNX compatible viterbi decoding
# Added type hints where needed to make lists work
# removed reversed()
# removed transition constraining ops
# .item() calls are ok for torchscript, but not ONNX.
# removed those, function now returns List[Tensor] instead of List[List[int]]
@torch.jit.script
def _viterbi_decode_onnx(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    transitions: torch.Tensor,
    include_start_end_transitions: bool,
) -> torch.Tensor:
    # emissions: (seq_length, batch_size, num_tags) logits
    # mask: (seq_length, batch_size)

    # seq_length = mask.size(0)
    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    if include_start_end_transitions:
        score = start_transitions + emissions[0]
    else:
        score = emissions[0]
    history: List[torch.Tensor] = []  # list of tensors, is ok

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag

    for i in range(1, 70):

        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        # TODO this torch.where call is broken, uses
        mask_at_pos = mask[i].unsqueeze(1)
        score = torch.where(mask_at_pos, next_score, score)
        # score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    if include_start_end_transitions:
        score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list: List[torch.Tensor] = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        # best_tags = [best_last_tag.item()]
        best_tags: List[torch.Tensor] = []
        # best_tags.append(best_last_tag.item())
        best_tags.append(best_last_tag)

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        # for hist in reversed(history[:seq_ends[idx]]):
        # reversed op is broken in torchscript, assumes tensor arg. -1 step indexing is also broken.
        history_reversed = history[: seq_ends[idx]].copy()
        history_reversed.reverse()
        for hist in history_reversed:
            best_last_tag = hist[idx][best_tags[-1]]
            # best_tags.append(best_last_tag.item())
            best_tags.append(best_last_tag)

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(torch.stack(best_tags))

    # pad the viterbi paths
    # NOTE this might be incompatible with ONNX.we use it for the jit.script() export, to allow tracing
    # of full model with inlining of viterbi decoding
    pos_preds = torch.nn.utils.rnn.pad_sequence(
        best_tags_list, batch_first=True, padding_value=-1.0
    )
    return pos_preds


# FROM https://github.com/JeppeHallgren/pytorch-crf/blob/master/torchcrf/__init__.py


class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(
        self,
        num_tags: int,
        batch_first: bool = False,
        constrain_every: bool = False,
        allowed_transitions: List[Tuple[int, int]] = None,
        allowed_start: List[int] = None,
        allowed_end: List[int] = None,
        include_start_end_transitions: bool = False,
        init_weight: float = 0.577350,
    ) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.constrain_every = constrain_every
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.transition_constraint = allowed_transitions is not None
        self.init_weight = init_weight
        self.include_start_end_transitions = True
        # self.include_start_end_transitions = include_start_end_transitions
        if self.include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))

            if allowed_transitions is None:  # All transitions are valid.
                constraint_start_mask = torch.empty(num_tags).fill_(1.0)
                constraint_end_mask = torch.empty(num_tags).fill_(1.0)
            else:
                constraint_start_mask = torch.empty(num_tags).fill_(0.0)
                constraint_end_mask = torch.empty(num_tags).fill_(0.0)
                constraint_start_mask[allowed_start] = 1.0
                constraint_end_mask[allowed_end] = 1.0
            self._constraint_start_mask = torch.nn.Parameter(
                constraint_start_mask, requires_grad=False
            )
            self._constraint_end_mask = torch.nn.Parameter(
                constraint_end_mask, requires_grad=False
            )

        # todo: implement constraints. only used in viterby decode.
        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        constraint_mask = None
        if allowed_transitions is None:  # All transitions are valid.
            constraint_mask = torch.empty(num_tags, num_tags).fill_(1.0)
        else:
            constraint_mask = torch.empty(num_tags, num_tags).fill_(0.0)
            for i, j in allowed_transitions:
                constraint_mask[i, j] = 1.0
        #     todo: could maybe be done more pretty than a loop?
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        # nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -self.init_weight, self.init_weight)
        # nn.init.uniform_(self.transitions, -0.577350, 0.577350)

        # from allennlp
        # nn.init.xavier_normal_(self.transitions)

        if self.include_start_end_transitions:
            # nn.init.uniform_(self.start_transitions, -0.577350, 0.577350)
            # nn.init.uniform_(self.end_transitions, -0.577350, 0.577350)
            nn.init.uniform_(
                self.start_transitions, -self.init_weight, self.init_weight
            )
            nn.init.uniform_(self.end_transitions, -self.init_weight, self.init_weight)
            # nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            # nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            # torch.nn.init.normal_(self.start_transitions)
            # torch.nn.init.normal_(self.end_transitions)

        if self.transition_constraint:
            self.do_transition_constraint()

    def do_transition_constraint(self):
        # inf = torch.finfo(self.start_transitions.dtype).min
        inf = torch.as_tensor(-20, dtype=self.transitions.dtype)
        inf_matrix = (
            torch.empty(self.transitions.shape).fill_(inf).to(self.transitions.device)
        )
        self.transitions.data = torch.where(
            self._constraint_mask.byte(), self.transitions, inf_matrix
        )

        if self.include_start_end_transitions:
            inf_vector = (
                torch.empty(self.start_transitions.shape)
                .fill_(inf)
                .to(self.start_transitions.device)
            )
            self.start_transitions.data = torch.where(
                self._constraint_start_mask.byte(), self.start_transitions, inf_vector
            )
            self.end_transitions.data = torch.where(
                self._constraint_end_mask.byte(), self.end_transitions, inf_vector
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    ## made this fn exportable for ONNX
    def forward(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.ByteTensor] = None,
        init_state_vector: Optional[torch.LongTensor] = None,
        forced_steps: int = 2,
        no_mask_label: int = 0,
    ) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        # self._validate(emissions, mask=mask)
        # if mask is None:
        #    mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8) #

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if init_state_vector is not None:
            paths = self._viterbi_decode_force_states(
                emissions, mask, init_state_vector, forced_steps, no_mask_label
            )
        else:
            # paths = self._viterbi_decode(emissions, mask)
            paths = _viterbi_decode_onnx(
                emissions,
                mask,
                self.start_transitions,
                self.end_transitions,
                self.transitions,
                self.include_start_end_transitions,
            )

        print("viterbi done")
        return paths

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags) logits
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        if self.transition_constraint:
            start_transitions_temp = self.start_transitions.data.clone()
            end_transitions_temp = self.end_transitions.data.clone()
            transitions_temp = self.transitions.data.clone()
            self.do_transition_constraint()

        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        if self.include_start_end_transitions:
            score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        if self.transition_constraint:
            self.start_transitions.data = start_transitions_temp
            self.end_transitions.data = end_transitions_temp
            self.transitions.data = transitions_temp

        return best_tags_list

    @staticmethod
    def _log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(
            torch.sum(torch.exp(tensor - broadcast_offset), dim)
        )
        # Add offset back
        return offset + safe_log_sum_exp

    def _viterbi_decode_force_states(
        self,
        emissions: torch.FloatTensor,
        mask: torch.ByteTensor,
        init_state_vector=torch.LongTensor,
        forced_steps: int = 2,
        no_mask_label: int = 0,
    ) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags) logits
        # mask: (seq_length, batch_size)
        # init_state_vector (batch_size) label of initial state of path
        # forced_steps: number of steps for which to override emissions according to init_state_vector
        # no_mask_label label in init_state_vector for which no masking should be performed (NO-SP token, because there it doesn't matter)
        # State forcing works by adding a large constant (100000) to the emissions of the states that shall be forced
        # for forced_stes. Then viterbi decoding continues as usual. There is no theorical guarantee for this to work,
        # but testing it showed that the constant is large enough to force the path through the states when backtracing
        # and it does not cause any overflow.
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        if self.transition_constraint:
            start_transitions_temp = self.start_transitions.data.clone()
            end_transitions_temp = self.end_transitions.data.clone()
            transitions_temp = self.transitions.data.clone()
            self.do_transition_constraint()

        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # create mask - set all emissions except the one for the desired initial label to 0
        init_steps_mask = torch.nn.functional.one_hot(
            init_state_vector, num_classes=self.num_tags
        )  # batch_size, num_classes
        dont_mask_idx = torch.where(init_state_vector == no_mask_label)[
            0
        ]  # NO_SP token, provide as argument.

        init_steps_mask[
            dont_mask_idx
        ] = 1  # (batch_size, num_tags) mask that is 0 at all emissions that should be set to 0

        # it seems that 0 masking is not working - try making wanted emissions large instead of zeroing unwanted
        init_steps_mask[
            dont_mask_idx
        ] = 0  # init_steps_mask is 1 at all emissions that shall be scaled

        # mask initial score
        score = score + init_steps_mask * 10000

        # force first steps
        for i in range(1, forced_steps):
            broadcast_score = score.unsqueeze(2)  # batch_size, num_tags, 1)

            # apply mask to emissions - unwanted emissions get set to 0.
            emissions_fixed = (
                emissions[i] + init_steps_mask * 100000
            )  # batch_size, num_tags

            broadcast_emission = emissions_fixed.unsqueeze(1)  # batch_size, 1, num_tags

            next_score = (
                broadcast_score + self.transitions + broadcast_emission
            )  # batch_size, num_tags, num_tags

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # now we can continue as usual

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(forced_steps, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        if self.include_start_end_transitions:
            score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        if self.transition_constraint:
            self.start_transitions.data = start_transitions_temp
            self.end_transitions.data = end_transitions_temp
            self.transitions.data = transitions_temp

        return best_tags_list
