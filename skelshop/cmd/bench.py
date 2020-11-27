import time

import click

from skelshop.io import AsIfSingleShot, ShotSegmentedReader


@click.group()
def bench():
    """
    Commands to benchmark SkelShop's I/O speeds.
    """
    pass


@bench.command()
@click.argument("skels_fn", type=click.Path(exists=True))
def read_shot_seg(skels_fn):
    """
    Benchmark reading a shot segmented skeleton file.
    """
    import h5py

    with h5py.File(skels_fn, "r") as skels_h5:
        begin_time = time.time()
        prev_time = begin_time
        prev_skels = 0
        prev_bundles = 0
        skels_cnt = 0
        bundles_cnt = 0

        def check_time():
            nonlocal prev_time, prev_skels, prev_bundles
            cur_time = time.time()
            if cur_time > prev_time + 30:
                print(
                    "Last 30s:\n"
                    f"Skels/s = {(skels_cnt - prev_skels) / (cur_time - prev_time)}\n"
                    f"Bundles/s = {(bundles_cnt - prev_bundles) / (cur_time - prev_time)}"
                )
                prev_time = cur_time
                prev_skels = skels_cnt
                prev_bundles = bundles_cnt

        for bundle in AsIfSingleShot(ShotSegmentedReader(skels_h5, infinite=False)):
            for skel in bundle:
                skels_cnt += 1
            bundles_cnt += 1
            check_time()
    end_time = time.time()
    print(
        "Total:\n"
        f"Skels/s = {skels_cnt / (end_time - begin_time)}\n"
        f"Bundles/s = {bundles_cnt / (end_time - begin_time)}"
    )
