When generating face embeddings, there are a number of choices to make:

| Decision | Options |
|---|---|
| Detection/localisation | Dlib/OpenPose |
| Which frames to embed faces from | All/Only the best |

## What to use for detection/localisation?

In almost all cases, OpenPose based detection and localisation should be used.
The reason is twofold:

 * Skeletons are currently the only entity which can be tracked in OpenPose.
   Associating face embeddings with a skeleton means they remain attached
   assigned to the correct person within each shot.

 * Assuming you need to estimate a skeleton for other reasons, using these
   already estimated keypoints is faster.

## Embed all frames or only the best?

This question is not as clear, given that given a reasonable GPU, embedding
faces is quite fast (at least for dlib), and so the benefit of only embedding
a few keyframes is not as marked as might be expected. The reason for this is
probably that the face embedding itself is fast enough that is comparable to
the video decoding cost, which is necessary in both cases.

So the choice of which to use is more a matter of what the downstream task
needs. In general, for face embedding clustering, more points per shot-person
is not necessarily a good thing and so embedding only the best is usually the
best option (also referred to as producing a *sparse* face embedding dump). The
reason is that the clustering algorithms have memory usage proportional. For
direct identification, it's up to you, and it certainly may be the case that
more accurate identification is possible from a full face embedding dump.

Embedding all frames is done with the [`skelshop face
embedall`](cli.md#embedall) command, while embedding only the
best is done with [`skelshop face
bestcands`](cli.md#bestcands) followed by [`skelshop face
embedselect`](cli.md#embedselect).
