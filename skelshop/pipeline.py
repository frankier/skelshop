from functools import wraps

import click

from skelshop.bbshotseg import ShotSegStage
from skelshop.bbtrack import TrackStage
from skelshop.csvshotseg import CsvShotSegStage
from skelshop.pipebase import RewindStage
from skelshop.track import PoseMatcher
from skelshop.track.confs import CONFS as TRACK_CONFS
from skelshop.track.metrics.lighttrack_pose_match import LightTrackPoseMatchMetric


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
    "track_conf",
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
    if kwargs.get("track") and not kwargs.get("pose_matcher_config"):
        raise click.BadOptionUsage(
            "--pose-matcher-config", "--pose-matcher-config required when --track",
        )
    if (not kwargs.get("track") and kwargs.get("shot_seg") != "none") or (
        kwargs.get("track") and kwargs.get("shot_seg") == "none"
    ):
        raise click.UsageError(
            "Cannot perform shot segmentation without tracking or visa-versa",
        )
    # Add metadata
    for k in PIPELINE_CONF_OPTIONS:
        pipeline.add_metadata(k, kwargs.get(k))
    # Add stages
    start_frame = kwargs.get("start_frame", 0)
    if kwargs.get("shot_seg") != "none":
        pipeline.add_stage(RewindStage, 20)
    if kwargs.get("track"):
        # XXX: A bit ugly just patching this on like so...
        LightTrackPoseMatchMetric.setup(PoseMatcher(kwargs.get("pose_matcher_config")))
        pipeline.add_stage(
            TrackStage, spec=TRACK_CONFS[kwargs.get("track_conf")],
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
            click.option(
                "--track-conf", type=click.Choice(TRACK_CONFS.keys()), default=None
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
