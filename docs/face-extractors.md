Before performing the face embedding step required for facial recognition, we
need to extract rotated, non-aspect ratio preserving scaled images of an exact
square size, since this is the type of data the embedders are trained on.

There are currently two major approaches to extracting face *chips*: those
based on dlib's pipeline and those based on OpenPose skeletons. Typically the
latter is recommended if you are planning on extracting these skeleton
keypoints at any point, since it is faster and allows the faces to be
associated with tracked skeletons.

## dlib

The dlib based face chip extractor first runs a face detector and then as show
in [the pipelines overview](pipelines-overview.md)

The following modes are available:

| Mode name  | Face detector | Face keypoint model |
| ------------- | ------------- | ------------- |
| dlib-hog-face5 | HOG | 5-point |
| dlib-hog-face68 | HOG | 68-point |
| dlib-cnn-face5 | CNN | 5-point |
| dlib-cnn-face68 | CNN | 68-point |

The CNN detector is slower than the HOG detector, but can be GPU accelerated.
The 5 point face keypoint model is slightly faster than the 68 keypoint model.
Both keypoint models run on the CPU only.

## OpenPose

The OpenPose based face chip extractor uses keypoints detected by OpenPose. If
your keypoint model (see [meet the pose estimators for a list of keypoint
models](pose-estimators.md) includes the face, you can use the 68 keypoint
model, otherwise you can use the 3 keypoint model which uses only the face
keypoints from BODY_25.

| Mode name  | Face keypoint model |
| ------------- | ------------- |
| openpose-face3 | 3-point |
| openpose-face68 | 68-point |
