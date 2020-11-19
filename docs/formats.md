The dump format is a HDF5 file:

```
/ - Contains metadata attributes such as:
    fmt_type = unseg | trackshots
    mode = BODY_25 | BODY_25_ALL | BODY_135
    num_frames
    various version information and command line flag information
    ...
/timeline - Contains shots if trackshots, otherwise if unseg contains
            poses directly.
/timeline/shot0 - A single shot containing poses and with attributes
                  start_frame and end_frame. This interval is closed at
                  the beginning and open and the end, as with Python
                  slices so that num_frames = end_frame - start_frame.
/timeline/shot0/pose0 - A CSR sparse matrix[1] stored as a group.
                        Has start_frame and end_frame. The shape of the
                        matrix is (num_frames, limbs, 3). Each element
                        of the matrix is a (x, y, c) tuple directly from
                        OpenPose.
```

1. [CSR sparse matrix on Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_\(CSR,_CRS_or_Yale_format\))
