from .metrics import BboxIouMetric, LightTrackPoseMatchMetric, WeightedSumMetric
from .spec import (
    AssignRemaining,
    GreedyAssignMetThresh,
    OptAssignMetThresh,
    OrElse,
    PrevFrameCascade,
    SumConfCandFilter,
    TrackingSpec,
)

CONFS = {
    "lighttrackish": TrackingSpec(
        enlarge_scale=0.2,
        prev_frame_buf_size=5,
        cand_filter=SumConfCandFilter(5),
        procedure=OrElse(
            [
                PrevFrameCascade(
                    OrElse(
                        [
                            GreedyAssignMetThresh(BboxIouMetric(), -0.3,),
                            GreedyAssignMetThresh(LightTrackPoseMatchMetric(), 0.4,),
                        ]
                    )
                ),
                AssignRemaining(),
            ]
        ),
    ),
    "opt_lighttrack": TrackingSpec(
        enlarge_scale=0.2,
        prev_frame_buf_size=5,
        cand_filter=SumConfCandFilter(5),
        procedure=OrElse(
            [
                PrevFrameCascade(
                    OrElse(
                        [
                            OptAssignMetThresh(BboxIouMetric(), -0.3,),
                            OptAssignMetThresh(LightTrackPoseMatchMetric(), 0.4,),
                        ]
                    )
                ),
                AssignRemaining(),
            ]
        ),
    ),
    "deepsortlike": TrackingSpec(
        # No Kalman filter
        enlarge_scale=0.2,
        prev_frame_buf_size=20,
        cand_filter=SumConfCandFilter(5),
        procedure=OrElse(
            [
                PrevFrameCascade(
                    OptAssignMetThresh(
                        WeightedSumMetric(
                            [(1, BboxIouMetric()), (1, LightTrackPoseMatchMetric()),]
                        ),
                        0,
                    ),
                    5,
                ),
                AssignRemaining(),
            ]
        ),
    ),
}
"""
A dictionary of ready-made tracking specs

Attributes:
    "lighttrackish" (TrackingSpec): A LightTrack-like algoirhtm using greedy matching
    "opt_lighttrack" (TrackingSpec): A LightTrack-like algorithm which attempts to make optimal matches
    "deepsortlike" (TrackingSpec): A DeepSort-like algorithm (currently missing Kalman filtering)
"""
