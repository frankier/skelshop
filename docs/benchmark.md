# Benchmarks

Some benchmarks are given here, mainly to help with back of the envelope -type
calculations of what level of resources might be needed for your own pipeline
build from these stages on your own video corpus.

The following benchmarks were run on the following machine:

 * 7 cores of Intel Xeon Gold 5120 CPU @ 2.20GHz
 * 1 P100 with 16GB RAM
 * 32 GB RAM

They were run on the 480p version of [a typical talking heads video on
YouTube](https://www.youtube.com/watch?v=9U4Ha9HQvMo). The video is just over
2 minutes, so use this to get an approximate scaling factor for your own
corpus. All times include the script/Singularity startup time, so this should
be taken into account.

| System | Time (m:s) |
|---|---|
| **Shot Segmentors** | |
| PySceneDetect | 10.4 |
| ffprobe | 10.7 |
| **Pose estimators** | |
| OpenPose Body25 | 4:23.8 |
| OpenPose Body25+Face | 8:53.9 |
| OpenPose Body25+Hands | 10:11.2 |
| OpenPose Body25+Face+Hands | 14:49.7 |
| **Within-shot pose trackers** | (run on output of Body25) |
| lighttrackish | 33.5 |
| opt_lighttrack | 17.7 |
| deepsortlike | 21.8 |
| **Face detection + embedding** | (this based upon embedding all faces found) |
| dlib-hog-face5 | 2:50.1 |
| dlib-hog-face68 | 2:53.1 |
| dlib-cnn-face5 | 0:44.5 |
| dlib-cnn-face68 | 0:43.7 |
| **Embedding faces from existing OpenPose** | (face3 only requires Body25, whereas face68 requires Body25+Face) |
| openpose-face68 | 17.3 |
| openpose-face3 | 21.0 |

## Accuracy comparisons

In many cases, accuracy comparisons can be found on the original websites or
papers of the software providing the different pipeline stages. However, here
are some extra comparisons available elsewhere. Note that these may refer to
old versions:

 * [Dlib on LFW](https://github.com/tahaemara/dlib-model-evaluation)
 * [Shot detection benchmarks](https://github.com/albanie/shot-detection-benchmarks)
 * [Master's thesis analysing failure modes of different face embeddings including dlib's](https://matheo.uliege.be/handle/2268.2/4650)
