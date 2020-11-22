# Reading and writing the formats from your own code

There are tools to read skeletons and face embeddings from your own code. Here
is an example of reading from a tracked skeleton file:


    import h5py
    from skelshop.io import ShotSegmentedReader

    with h5py.File("/path/to/my/skels.h5", "r") as skels_f:
        for shot in ShotSegmentedReader(skels_f):
            for skel_id, skel in shot:
                print(skel_id, skel)

## Reference

::: skelshop.io.UnsegmentedWriter

::: skelshop.io.ShotSegmentedWriter

::: skelshop.io.ShotSegmentedReader

::: skelshop.io.UnsegmentedReader

::: skelshop.io.ShotReader

::: skelshop.io.AsIfTracked

::: skelshop.io.AsIfSingleShot
