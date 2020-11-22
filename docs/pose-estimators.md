## Pose estimators

Currently pose estimation is based on OpenPose.

### OpenPose

OpenPose can produce untracked poses. It can produce a variety of different
keypoint models.

#### Modes

| Mode name  | Keypoints | Body | Hands | Face |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BODY_25  | 25  | Yes | No | No |
| BODY_25_ALL  | 135  | Yes | Yes | Yes |
| BODY_25_HANDS  | 65  | Yes | Yes | No |
| FACE  | 70  | No | No | Yes |
| BODY_25_FACE  | 95  | Yes | No | Yes |
