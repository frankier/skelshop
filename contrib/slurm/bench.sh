#!/bin/bash
#
#SBATCH --job-name=skelshop_dump_bench
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gpu2080

cd /mnt/rds/redhen/gallina
module load singularity

wget https://youtube-dl.org/downloads/latest/youtube-dl
chmod +x ./youtube-dl
./youtube-dl -o breakingnews.mp4 -f 'bestvideo' 'https://www.youtube.com/watch?v=9U4Ha9HQvMo'

echo "pyscenedetect\n" > results.txt

{ time \
singularity exec ~/sifs/skelshop.sif snakemake \
    -CVIDEO_BASE=`pwd` -CDUMP_BASE=`pwd` breakingnews-Scenes.csv \
    breakingnews.mp4 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "ffprobe\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif snakemake \
    -CVIDEO_BASE=`pwd` -CDUMP_BASE=`pwd` breakingnews.ffprobe.scene.csv \
    breakingnews.mp4 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "BODY_25 dump\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop dump \
    --mode BODY_25 breakingnews.mp4 body25.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "BODY_25_ALL dump\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop dump \
    --mode BODY_25_ALL breakingnews.mp4 body25all.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "lighttrackish track\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf lighttrackish \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.lighttrackish.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "opt_lighttrack track\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf opt_lighttrack \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.opt_lighttrack.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "deepsortlike track\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf deepsortlike \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.deepsortlike.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "opt_lighttrack track body25_all (not strictly part of the benchmark)\n" >> results.txt

{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf opt_lighttrack \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25all.dump.h5 body25all.tracked.opt_lighttrack.dump.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "dlib-hog-face5 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    dlib-hog-face5 breakingnews.mp4 breakingnews.dlibhogface5.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face5 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    dlib-cnn-face5 breakingnews.mp4 breakingnews.dlibcnnface5.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-hog-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    dlib-hog-face68 breakingnews.mp4 breakingnews.dlibhogface68.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    dlib-cnn-face68 breakingnews.mp4 breakingnews.dlibcnnface5.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "openpose-face3 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --from-skels body25.tracked.opt_lighttrack.dump.h5 \
    openpose-face3 breakingnews.mp4 breakingnews.openposeface3.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "openpose-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --from-skels body25all.tracked.opt_lighttrack.dump.h5 \
    openpose-face68 breakingnews.mp4 breakingnews.openposeface68.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "(again but with bboxes/chips [slower])" >> results.txt
echo "dlib-hog-face5 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-hog-face5 breakingnews.mp4 breakingnews.dlibhogface5.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face5 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-cnn-face5 breakingnews.mp4 breakingnews.dlibcnnface5.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-hog-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-hog-face68 breakingnews.mp4 breakingnews.dlibhogface68.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-cnn-face68 breakingnews.mp4 breakingnews.dlibcnnface5.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "openpose-face3 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    --from-skels body25.tracked.opt_lighttrack.dump.h5 \
    openpose-face3 breakingnews.mp4 breakingnews.openposeface3.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo "openpose-face68 face\n" >> results.txt
{ time \
singularity exec ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    --from-skels body25all.tracked.opt_lighttrack.dump.h5 \
    openpose-face68 breakingnews.mp4 breakingnews.openposeface68.chips.h5 \
> /dev/null 2>&1 ; } 2>> results.txt

echo >> results.txt
