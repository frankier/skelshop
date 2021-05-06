# Person identification

There are two major approaches to identification:

 1. Direct comparison with one or more references. For each identified
    reference, multiple images may be supplied.
 2. First performing clustering and then either:
   1. Identifying clusters by comparison with one or more references.
   2. Identifying clusters manually by inspection of prototypical images.

## Construction of a face reference library

A face reference library is simply a directory containing a number of
directories, each containing photos containing the face of some individual of
interest. Each directory should have a unique name which will be given used as
a label for that person. All images should be JPEGs containing exactly one
face. For example:

```
myref
├── bill
│   └── billphoto.jpg
└── ben
    └── benphoto.jpeg
```

By convention, we can use Wikidata identifiers to identify people:

```
myref
├── Q230739
│   └── Katie%20Couric%20VF%202012%20Shankbone%202.JPG
├── Q6271597
│   └── Jon%20Snow2.jpg
└── Q929985
    └── Alex%20Trebek%20%28May%2021%2C%202012%29.jpg.jpg
```

SkelShop can help quickly construct this specific type of reference. For
example the above reference can be created by first creating a file
`people.txt` containing:

```
Q230739
Q6271597
Q929985
```

And then simply running:

    $ skelshop iden buildrefs entities people.txt myref wikidata

## Embedding your face library in advance

At your option, you can embed the reference library using [``skelshop iden
embedrefs``](cli.md#embedrefs)`` myref/ myref.h5``. All commands accepting
a reference can then be given the output HDF5 file instead of the input
directory.

## Direct comparison

Once we have [obtained either a full or sparse set of face
embeddings](face-pipeline-configuration-guide.md), we can identify people in
each shot by direct comparison with our reference like so for a sparse
embedding dump [``skelshop iden idsegssparse``](cli.md#idsegssparse)`` myref/
path/to/scenes.csv path/to/sparsefacedump.h5 outputids.csv``. Or like so for
a full embedding dump: [``skelshop iden idsegsfull``](cli.md#idsegsfull)``
myref/ path/to/scenes.csv path/to/fullfacedump.h5 outputids.csv``.

## Making a corpus description file

Up until now, we have been dealing with commands that process either only
a single video at a time, or process information associated with a single
video, such as skeleton dumps, at a time. However, when we perform face
clustering we would like to deal data from multiple videos. To do this, we need
to (usually automatically, e.g. in [Snakemake](snakemake.md)) create a CSV file
for our corpus. Here is an example:

```
video,group,group_typ,faces,segsout,untracked_skels,tracked_skels,bestcands
/path/to/video.mp4,/path/to/scenes,ffprobe|psd,/path/to/facedump.h5,/path/to/clustering.output.csv,/path/to/untracked.skels.h5,/path/to/tracked.skels.h5,/path/to/bestcands.csv
```

Paths in the corpus description file can be absolute or relative. If they are
relative, you must pass in ``--corpus-base`` at the same time as passing in the
CSV file.

## Clustering

The best family of clustering algorithm to use in situations like ours, where
clusters of high density are separated by areas of areas of low density are
density-based clustering algorithms. The classic density-based clustering
algorithm is DBSCAN.

However, the version of DBSCAN included in sklearn has issues with high memory
consumption[^sklearn-dbscan-memory]. sklearn also contains the OPTICS
clustering method, which is capable of producing clusterings identical to
DBSCAN. Sklearn's OPTICS implementation doesn't have memory consumption issues.

By default sklearn will use its own tree based spacial indexes to accelerate
the distance queries underlying DBSCAN like algorithms, however, these indices
are unable to deal well with hundreds of thousands of high dimensional vectors,
like the face embeddings we are dealing with here. To mitigate this problem, it
is possible to first use an library with an accelerated index designed for
dealing with large numbers of high dimensional points to construct a (usually
approximate) K-nearest-neighbours graph and then run clustering on
this[^sklearn-ann].

Running DBSCAN on a K-nearest-neighbours graph is possible, however DBSCAN is
based on radius queries, we we must be careful to pick a high enough K that,
most of the time most of the points within the query radius as in the Knn
graph. As an added advantage, RNN-DBSCAN[^rnn-dbscan] only has a single parameter, K, which
is the theshold of reverse nearest neighbours, but can be though of as the
similar to DBSCAN's minPts - 1, making tuning easier. In our case if we take
f faces per segment then we can choose this parameter as K = nf - 1 where n is
the minimum number of appearances in segments someone should make to be
assigned a cluster rather than treated as noise. 2 is a reasonable value for n,
and so K = 5 is a good value for RNN-DBSCAN assuming you leave f at its
default of 3.

Currently, RNN-DBSCAN with pynndescent is the recommended approach. So the
recommended command is: [``skelshop iden clus fixed``](cli.md#fixed)``
--proto-out protos --model-out model.pkl --ann-lib pynndescent --algorithm
rnn-dbscan --knn 5 path/to/corpus.description.csv``. The ``--proto-out`` option
gives a path to write prototypes/exemplars from each cluster, usable to align
the clusters after-the-fact, while the ``--model-out`` option shows where to
dump the model, which provides an alternative way of achieving the same thing,
as explained in the next section.

## Labelling clusters

There are a three approaches to labelling clusters:

 1. Manual labelling based on saved prototypes
 2. Labelling by comparing a reference to saved prototypes
 3. Labelling by comparing a reference to a saved kNN+clustering model (only
    supported for pynndescent+RNN-DBSCAN)

In the first case of manual labelling, we usually want to dump the images of
the prototypes for each cluster like so: [``skelshop iden
writeprotos``](cli.md#writeprotos)`` protos path/to/corpus.description.csv
path/to/proto/images``. You can either install
[sxiv](https://github.com/muennich/sxiv) and use [``skelshop iden
whoisthis``](cli.md#whoisthis)`` path/to/proto/images
cluster.identifications.csv`` to interactively labels the clusters, or manually
then use an image viewer to view the cluster prototypes and then create a CSV
file ``cluster.identifications.csv`` in the following format:

```
label,clus
Q42,c0
```

In the second case of prototype-based automatic labelling, we can use:
[``skelshop iden idclus``](cli.md#idclus)`` myref.h5 protos
path/to/corpus.description.csv cluster.identifications.csv``. Finally, to use
a saved RNN-DBSCAN model [``skelshop iden idrnnclus``](cli.md#idrnnclus)``
myref.h5 model.pkl cluster.identifications.csv``.

After completing any of these three approaches, the labelled clusters can then
be applied to the clustering outputs to produce a CSV with a mix of identities
and clusters like so: [``skelshop iden applymap``](cli.md#applymap)``
clustering.output.csv cluster.identifications.csv outputids.csv``.

[^sklearn-dbscan-memory]: This is discussed further in [this StackOverflow
discussion](https://stackoverflow.com/questions/16381577/scikit-learn-dbscan-memory-usage).

[^sklearn-ann]: The wrapper code enabling this, as well as the implementation of
RNN-DBSCAN is found in [sklearn-ann](https://github.com/frankier/sklearn-ann).

[^rnn-dbscan]: [A. Bryant and K. Cios, "RNN-DBSCAN: A Density-Based Clustering
Algorithm Using Reverse Nearest Neighbor Density Estimates," in IEEE
Transactions on Knowledge and Data Engineering, vol. 30, no. 6, pp. 1109-1121,
1 June 2018, doi: 10.1109/TKDE.2017.2787640.](https://ieeexplore.ieee.org/document/8240674)
