<h1><img src="https://raw.githubusercontent.com/frankier/skelshop/master/skelshop/news.png" /> SkelShop</h1>

<p align="center">
<a href="https://gitlab.com/frankier/skelshop/-/commits/master">
  <img alt="pipeline status" src="https://gitlab.com/frankier/skelshop/badges/master/pipeline.svg" />
</a>
<a href="https://hub.docker.com/r/frankierr/skelshop/builds">
  <img alt="DockerHub hosted images" src="https://img.shields.io/docker/pulls/frankierr/skelshop?style=flat" />
</a>
<a href="https://frankier.github.io/skelshop/">
  <img alt="Documentation on GitHub pages" src="https://img.shields.io/badge/Docs-MkDocs-informational" />
</a>
<a href="https://github.com/psf/black">
  <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg" />
</a>
</p>

SkelShop is a toolkit usable either as a suite of command line tools or a
Python library aimed at *offline analysis* of the ``talking heads'' genre of
videos, typically created in a television studio. This genre includes news
and current affairs programmes, variety shows and some documentaries. The
main attributes of this type of material is:
 * Multiple fixed place cameras in a studio with shot changes to different
   cameras and non-studio material.
 * Mainly upper bodies are visible.
 * Frontal shots are the most frequent type.
 * Faces are usually visible and most often frontal.
 * Occlusion is as often by an OSD (On Screen Display) than by some other
   object in the scene.

## Getting started

[See the documentation.](https://frankier.github.io/skelshop/)

## Features

 * Dump OpenPose skeletons to a fast-to-read HDF5 file format
 * A processing pipeline starting with shot segmentation
 * Apply black box person tracking on OpenPose skeletons
 * Draw skeletons over video and...
   * View the result in real time
   * Output the result to another video
 * Convert from some existing JSON based dump formats
 * Embed faces using dlib
   * Using OpenPose's face detection and keypoint estimation
   * Or using dlib's own face detection/keypoint estimation
   * Select best frames for best quality embeddings
 * Dump across heterogeneous HPC environments including GPU and CPU nodes.
 * Identify faces based on embeddings
   * Identify by direct comparison with references
   * Or cluster faces with no reference (also useful re-identitication)
   * Identify faces based on clusters, either with a reference or interactively
   * Quickly build up libraries of references using data from WikiData/WikiMedia commons

Here's a screenshot of the playsticks command:

![Screenshot of the playsticks
command](https://user-images.githubusercontent.com/299380/87277551-2d9f6180-c4eb-11ea-917c-4336ad36a97f.png)

[Find out more about what SkelShop can do in the documentation.](https://frankier.github.io/skelshop/)

## Contributions & Questions

Contributions are welcome! Feel free to use GitHub discussions to ask any
questions. [See also the contributing section of the
documentation.](https://frankier.github.io/skelshop/development/)

## Acknowledgments

Thanks to the authors of all the useful libraries I have used.

Some of the black box tracking code is based
on [this repository](https://github.com/lxy5513/cvToolkit).

Some code to do with clustering and usage of FAISS is based on [this repository](https://github.com/yl-1993/learn-to-cluster/).

[Icon by Adrien Coquet, FR from the Noun Project used under
CC-BY.](https://thenounproject.com/term/news/2673777)
