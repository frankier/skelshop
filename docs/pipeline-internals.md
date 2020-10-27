The stage interface is quite simple. Each stage acts as an iterator, typically
yielding some kind of pose bundle. A pose bundle is an iterator of skeletons,
either with ids or not depending on whether it has been tracked.

Each stage inherits from the `PipelineStageBase` abstract base class which
includes also `send_back` to send back events to earlier stages in the
pipeline.

::: skelshop.pipebase.PipelineStageBase

## Events types in use through `send_back`

Currently **`cut`** event is sent back by any shot segmentation stage to the
tracking stage, so that tracking can be reset. Each stage is free to deal with
events as it wishes, e.g. a tracking stage attempting to track across shots
could react differently to this event.

A **`rewind`** event can be sent back so that a `RewindStage` will reverse
a given number of frames in its buffer. Note that you must arrange for
a `RewindStage` to be placed into the pipeline.
