The tracker is a stage that each frame takes an untracked bundle of poses and
gives them IDs. 

::: skelshop.bbtrack.TrackStage

Internally it uses the following class, which implements a generic approach to
online pose tracking:

::: skelshop.track.PoseTrack

Which is in turn configured using a:

::: skelshop.track.spec.TrackingSpec

Several configurations are given in:

::: skelshop.track.confs.CONFS

There are references to two systems in the name: LightTrack[^1] and
DeepSort[^2]. The configurations are inspired by these, but not exact
implementations.

Looking at the implementations in `skelshop.track.confs.CONFS` is a good
starting point for adding new configurations, or extending the tracking, e.g.
with a new approach to reidentification.

[^1]: Guanghan Ning, Heng Huang (2019) *LightTrack: A Generic Framework for Online Top-Down Human Pose Tracking* [https://arxiv.org/abs/1905.02822](https://arxiv.org/abs/1905.02822)

[^2]: Nicolai Wojke, Alex Bewley, Dietrich Paulus (2017) *Simple Online and Realtime Tracking with a Deep Association Metric* [https://arxiv.org/abs/1703.07402](https://arxiv.org/abs/1703.07402)
