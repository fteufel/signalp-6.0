__version__ = "0.7.2"

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

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

    def forward(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        tag_bitmap: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise. If none, then tag_bitmap should be provided.
            tag_bitmap (`~torch.LongTensor`): (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise. If none, then tags should be provided.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
            multitag: use multitag method to calculate sequence score
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        # assert (tags is not tag_bitmap), "Provide either tags or tag_bitmap"

        if tag_bitmap is not None:
            self._validate(emissions, tags=tag_bitmap[:, :, 0], mask=mask)
        else:
            self._validate(emissions, tags=tags, mask=mask)

        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tag_bitmap is not None:
                tag_bitmap = tag_bitmap.transpose(0, 1)
            else:
                tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if self.constrain_every:
            self.do_transition_constraint()

        # import torch.nn.functional as F # PyTorch Utilily functions
        # tag_bitmap = F.one_hot(tags,9).to(emissions.device)

        # shape: (batch_size,)
        if tag_bitmap is not None:  # if true then tags is tag_bitmap
            log_numerator = self._compute_seq_score_multi_tag(
                emissions, tag_bitmap, mask
            )
        else:
            log_numerator = self._compute_seq_score(
                emissions, tags, mask
            )  # aka. sequence scores TODO: unary + binary scores
        # shape: (batch_size,)
        log_denominator = self._compute_log_normalizer(
            emissions, mask
        )  # log norm normalization constant
        # shape: (batch_size,)
        llh = log_numerator - log_denominator

        # print(self.transitions)

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.float().sum()

    def decode(
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
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if init_state_vector is not None:
            paths = self._viterbi_decode_force_states(
                emissions, mask, init_state_vector, forced_steps, no_mask_label
            )
        else:
            paths = self._viterbi_decode(emissions, mask)

        return paths

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_seq_score(  # input likelihood
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags) logits
        # tags: (seq_length, batch_size) y labels
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        if self.include_start_end_transitions:
            score = self.start_transitions[tags[0]]
            score += emissions[0, torch.arange(batch_size), tags[0]]
        else:
            score = emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += (
                self.transitions[tags[i - 1], tags[i]] * mask[i]
            )  # TODO binary score

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += (
                emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            )  # TODO unary score

        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        if self.include_start_end_transitions:
            score += self.end_transitions[last_tags]  # an extra score

        return score

    def _compute_seq_score_multi_tag(
        self,
        emissions: torch.Tensor,
        tag_bitmap: torch.LongTensor,
        mask: torch.ByteTensor,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        """Computes the unnormalized score of all tag sequences matching
        tag_bitmap.
        tag_bitmap enables more than one tag to be considered correct at each time
        step. This is useful when an observed output at a given time step is
        consistent with more than one tag, and thus the log likelihood of that
        observation must take into account all possible consistent tags.
        Using one-hot vectors in tag_bitmap gives results identical to
        crf_sequence_score.
        Args:
          emissions: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
              representing all active tags at each index for which to calculate the
              unnormalized score.
          mask: (seq_length, batch_size)
        Returns:
          sequence_scores: A [batch_size] vector of unnormalized sequence scores.
        """
        # torch.as_tensor(float("-inf"))
        # torch.as_tensor(0)
        # inf_matrix = torch.empty(emissions.shape).fill_(torch.as_tensor(float("-inf"))).to(emissions.device)
        inf_matrix = (
            torch.empty(emissions.shape)
            .fill_(torch.finfo(emissions.dtype).min)
            .to(emissions.device)
        )
        # inf_matrix = torch.empty(emissions.shape).fill_(torch.as_tensor(-30.0)).to(emissions.device)
        filtered_inputs = torch.where(tag_bitmap.byte(), emissions, inf_matrix)

        seq_score = self._compute_log_normalizer(filtered_inputs, mask)
        # torch.index_fill(emissions, )

        return seq_score

    def _compute_log_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags) logits
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        if self.include_start_end_transitions:
            score = self.start_transitions + emissions[0]
        else:
            score = emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        if self.include_start_end_transitions:
            score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

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

    def _compute_log_alpha(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor, run_backwards: bool
    ) -> torch.FloatTensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()
        broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
        emissions_broadcast = emissions.unsqueeze(2)
        seq_iterator = range(1, seq_length)

        if run_backwards:
            # running backwards, so transpose
            broadcast_transitions = broadcast_transitions.transpose(
                1, 2
            )  # (1, num_tags, num_tags)
            emissions_broadcast = emissions_broadcast.transpose(2, 3)

            # the starting probability is end_transitions if running backwards
            # (batch_size,tags)
            if self.include_start_end_transitions:
                log_prob = [self.end_transitions.expand(emissions.size(1), -1)]
            else:
                # batch_size = emissions.shape[1]
                # seq_ends = mask.long().sum(dim=0) - 1
                # log_prob = [emissions[seq_ends, torch.arange(batch_size), :]]
                log_prob = [torch.zeros_like(emissions[0])]

            # iterate over the sequence backwards
            seq_iterator = reversed(seq_iterator)
        else:
            # Start transition score and first emission
            # (batch_size, tags)
            if self.include_start_end_transitions:
                log_prob = [emissions[0] + self.start_transitions.view(1, -1)]
            else:
                log_prob = [torch.zeros_like(emissions[0])]
                # log_prob = [emissions[0]]

        for i in seq_iterator:
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Sum current log probability, transition, and emission scores
            score = (
                broadcast_log_prob + broadcast_transitions + emissions_broadcast[i]
            )  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = self._log_sum_exp(score, dim=1)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # copy the prior value
            log_prob.append(
                score * mask[i].unsqueeze(1)
                + log_prob[-1] * (1.0 - mask[i]).unsqueeze(1)
            )

        if run_backwards:
            log_prob.reverse()

        return torch.stack(log_prob)

    def compute_marginal_probabilities(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> torch.FloatTensor:
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        alpha = self._compute_log_alpha(emissions, mask, run_backwards=False)
        beta = self._compute_log_alpha(emissions, mask, run_backwards=True)
        if self.include_start_end_transitions:
            z = torch.logsumexp(alpha[alpha.size(0) - 1] + self.end_transitions, dim=1)
        else:
            z = torch.logsumexp(alpha[alpha.size(0) - 1], dim=1)
        # TODO what is the point of above line ?
        prob = alpha + beta - z.view(1, -1, 1)

        if self.batch_first:
            prob = prob.transpose(0, 1)

        marginals = torch.exp(prob)

        # res = torch.nn.functional.softmax(marginals, dim=-1)
        test = marginals.sum(-1)
        # todo: with include start and end transition, test is 1 as supposed too. But not without. Lacking emissions for end and start tag in this solution.
        return marginals

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
