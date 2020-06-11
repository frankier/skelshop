from functools import wraps

import click
from skeldump.bbshotseg import ShotSegStage
from skeldump.bbtrack import TrackStage
from skeldump.csvshotseg import CsvShotSegStage
from skeldump.pipebase import RewindStage
from skeldump.track import man_embed_match, mk_posetrack_gcn_pose_matcher


class Pipeline:
    def __init__(self):
        self.stages = []
        self.metadata = {}

    def add_stage(self, stage, *args, **kwargs):
        self.stages.append((stage, args, kwargs))

    def add_metadata(self, k, v):
        self.metadata[k] = v

    def apply_metadata(self, h5out):
        for k, v in self.metadata.items():
            h5out.attrs[k] = v

    def __call__(self, source):
        pipeline = source
        for stage, args, kwargs in self.stages:
            kwargs["prev"] = pipeline
            pipeline = stage(*args, **kwargs)
        return pipeline


PIPELINE_CONF_OPTIONS = [
    "shot_seg",
    "track",
    "track_enlarge_bbox",
    "track_buf_size",
    "track_min_conf_sum",
    "track_min_iou",
    "track_max_reid_dist",
    "track_reid_embed",
]


ALL_OPTIONS = PIPELINE_CONF_OPTIONS + [
    "pose_matcher_config",
    "shot_csv",
]


def process_options(options, allow_empty, kwargs):
    pipeline = Pipeline()
    # Check argument validity
    if kwargs.get("shot_seg") == "csv" and not kwargs.get("shot_csv"):
        raise click.BadOptionUsage(
            "--shot-csv", "--shot-csv required when --shot-seg=csv",
        )
    if kwargs.get("track_reid_embed") == "posetrack" and not kwargs.get(
        "pose_matcher_config"
    ):
        raise click.BadOptionUsage(
            "--pose-matcher-config",
            "--pose-matcher-config required when --track-reid-embed=posetrack",
        )
    if (not kwargs.get("track") and kwargs.get("shot_seg") != "none") or (
        kwargs.get("track") and kwargs.get("shot_seg") == "none"
    ):
        raise click.UsageError(
            "Cannot perform shot segmentation without tracking or visa-versa",
        )
    # Add metadata
    for k in PIPELINE_CONF_OPTIONS:
        if kwargs.get("track") and k.startswith("track_"):
            continue
        pipeline.add_metadata(k, kwargs.get(k))
    # Add stages
    start_frame = kwargs.get("start_frame", 0)
    if kwargs.get("shot_seg") == "bbskel":
        pipeline.add_stage(RewindStage, 20)
    if kwargs.get("track"):
        if kwargs.get("track_reid_embed") == "posetrack":
            pose_matcher = mk_posetrack_gcn_pose_matcher(
                kwargs.get("pose_matcher_config")
            )
        elif kwargs.get("track_reid_embed") == "manual":
            pose_matcher = man_embed_match
        else:
            pose_matcher = None
        pipeline.add_stage(
            TrackStage,
            pose_matcher=pose_matcher,
            enlarge_scale=kwargs.get("track_enlarge_bbox"),
            queue_len=kwargs.get("track_buf_size"),
            min_conf_sum=kwargs.get("track_min_conf_sum"),
            min_spatial_iou=kwargs.get("track_min_iou"),
            max_pose_distance=kwargs.get("track_max_reid_dist"),
        )
    if kwargs.get("shot_seg") == "bbskel":
        pipeline.add_stage(ShotSegStage)
    elif kwargs.get("shot_seg") == "csv":
        pipeline.add_stage(
            CsvShotSegStage, shot_csv=kwargs.get("shot_csv"), start_frame=start_frame
        )
    if not allow_empty and not pipeline.stages:
        raise click.UsageError("Cannot construct empty pipeline",)
    for option in ALL_OPTIONS:
        del kwargs[option]
    kwargs["pipeline"] = pipeline


def pipeline_options(allow_empty=True):
    def inner(wrapped):
        options = [
            click.option(
                "--shot-seg",
                type=click.Choice(["bbskel", "csv", "none"]),
                default="none",
            ),
            click.option("--track/--no-track", default=False),
            click.option("--track-enlarge-bbox", type=float, default=0.2),
            click.option("--track-buf-size", type=int, default=5),
            click.option("--track-min-conf-sum", type=float, default=5),
            click.option("--track-min-iou", type=float, default=0.3),
            click.option("--track-max-reid-dist", type=float, default=0.3),
            click.option(
                "--track-reid-embed",
                type=click.Choice(["posetrack", "manual", "none"]),
                default="none",
            ),
            click.option(
                "--pose-matcher-config", envvar="POSE_MATCHER_CONFIG", required=True
            ),
            click.option("--shot-csv", type=click.Path(exists=True)),
        ]

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            process_options(options, allow_empty, kwargs)
            wrapped(*args, **kwargs)

        for option in options:
            wrapper = option(wrapper)
        return wrapper

    return inner
