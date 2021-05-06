from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, cast

import numpy as np

from ..pose import PoseBase
from .metrics import Metric

PENALTY_WEIGHT = 1e6


@dataclass
class TrackingSpec:
    """
    A domain specific language for threshold distance-based style tracking.

    Candidate poses are first filtered using `cand_filter` and then `procedure`
    is executed to decide how to assign the candidates.
    """

    enlarge_scale: float
    prev_frame_buf_size: int
    cand_filter: "CandFilter"
    procedure: "ProcedureNode"


class CandFilter(ABC):
    @abstractmethod
    def accept_pose(self, pose: PoseBase) -> bool:
        ...


@dataclass
class SumConfCandFilter(CandFilter):
    min_score: float

    def accept_pose(self, pose: PoseBase) -> bool:
        score = cast(float, np.sum(pose.all()[:, 2]))
        return score >= self.min_score


@dataclass
class MeanConfCandFilter(CandFilter):
    min_score: float

    def accept_pose(self, pose: PoseBase) -> bool:
        confs = pose.all()[:, 2]
        score = cast(float, np.mean(confs[confs > 0]))
        return score >= self.min_score


class ProcedureNode(ABC):
    @abstractmethod
    def assign(
        self, get_fresh_id, candidates, references, tracking_state,
    ):
        ...


@dataclass
class OrElse(ProcedureNode):
    choices: List[ProcedureNode]

    def assign(
        self, get_fresh_id, candidates, references, tracking_state,
    ):
        for node in self.choices:
            node.assign(get_fresh_id, candidates, references, tracking_state)


@dataclass
class AssignRemaining(ProcedureNode):
    def assign(
        self, get_fresh_id, candidates, references, tracking_state,
    ):
        for tracked_pose in tracking_state.untracked_as_tracked(candidates):
            tracked_pose.track_as(get_fresh_id())
            tracking_state.track(tracked_pose)


@dataclass
class AssignMetMixin:
    metric: Metric
    thresh: Optional[float]


@dataclass
class GreedyAssignMetThresh(ProcedureNode, AssignMetMixin):
    def assign(
        self, get_fresh_id, candidates, references, tracking_state,
    ):
        for cand in tracking_state.untracked_as_tracked(candidates):
            cand_rep = self.metric.preproc_pose(cand)
            min_cost = float("inf")
            min_ref = None
            for ref in references:
                ref_rep = self.metric.preproc_pose(ref)
                cost = self.metric.cmp(cand_rep, ref_rep)
                if cost < min_cost:
                    min_cost = cost
                    min_ref = ref
            if min_ref is not None and (self.thresh is None or cost <= self.thresh):
                assert min_ref.track_id is not None
                cand.track_as(min_ref.track_id)
                tracking_state.track(cand)


@dataclass
class OptAssignMetThresh(ProcedureNode, AssignMetMixin):
    def assign(
        self, get_fresh_id, candidates, references, tracking_state,
    ):
        from scipy.optimize import linear_sum_assignment

        # Get references/candidates as lists
        references_list = list(references)
        if not references_list:
            return
        candidates_as_tracked_list = list(
            tracking_state.untracked_as_tracked(candidates)
        )
        if not candidates_as_tracked_list:
            return

        # Represent them
        refs_proc = self.metric.preproc_poses(references_list)
        assert len(refs_proc) == len(references_list)
        cands_proc = self.metric.preproc_poses(candidates_as_tracked_list)
        assert len(cands_proc) == len(candidates_as_tracked_list)
        cost_mat = np.empty((len(candidates_as_tracked_list), len(references_list)))

        # Build cost matrix
        for cand_idx, cand_proc in enumerate(cands_proc):
            for ref_idx, ref_proc in enumerate(refs_proc):
                cost = self.metric.cmp(cand_proc, ref_proc)
                if self.thresh is not None and cost > self.thresh:
                    cost_mat[cand_idx, ref_idx] = PENALTY_WEIGHT
                else:
                    cost_mat[cand_idx, ref_idx] = cost

        # Solve
        cand_idxs, ref_idxs = linear_sum_assignment(cost_mat)
        for cand_idx, ref_idx in zip(cand_idxs, ref_idxs):
            if cost_mat[cand_idx, ref_idx] == PENALTY_WEIGHT:
                continue
            cand = candidates_as_tracked_list[cand_idx]
            cand.track_as(references_list[ref_idx].track_id)
            tracking_state.track(cand)


@dataclass
class PrevFrameCascade(ProcedureNode):
    inner: ProcedureNode
    length: Optional[int] = None
    skip: Optional[int] = None

    def assign(
        self, get_fresh_id, candidates, reference_buf, tracking_state,
    ):
        for references in reference_buf.hist_iter(self.skip, self.length):
            # Copy references since they can be mutated within inner
            self.inner.assign(get_fresh_id, candidates, references, tracking_state)
