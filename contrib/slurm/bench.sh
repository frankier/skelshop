#!/bin/bash
#
#SBATCH --job-name=skelshop_dump_bench
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gpu2080

module load singularity
module load cuda/10.1

if [ ! -f "breakingnews.mp4" ]; then
    wget https://youtube-dl.org/downloads/latest/youtube-dl
    chmod +x ./youtube-dl
    ./youtube-dl -o breakingnews.mp4 -f 'bestvideo' 'https://www.youtube.com/watch?v=9U4Ha9HQvMo'
fi

echo "pyscenedetect" > results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif snakemake \
    "$(pwd)/breakingnews-Scenes.csv" -j1 \
    --snakefile /opt/skelshop/workflow/Snakefile \
    -CVIDEO_BASE="$(pwd)" -CDUMP_BASE="$(pwd)" \
> pyscenedetect.log 2>&1 ; } 2>> results.txt

echo "ffprobe" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif snakemake \
    "$(pwd)/breakingnews.ffprobe.scene.csv" -j1 \
    --snakefile /opt/skelshop/workflow/Snakefile \
    -CVIDEO_BASE="$(pwd)" -CDUMP_BASE="$(pwd)" \
> ffprobe.log 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "BODY_25 dump" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop dump \
    --mode BODY_25 breakingnews.mp4 body25.dump.h5 \
> body25.dump.log 2>&1 ; } 2>> results.txt

echo "BODY_25_ALL dump" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop dump \
    --mode BODY_25_ALL breakingnews.mp4 body25all.dump.h5 \
> body25all.dump.log 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "lighttrackish track" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf lighttrackish \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.lighttrackish.dump.h5 \
> body25.tracked.lighttrackish.dump.log 2>&1 ; } 2>> results.txt

echo "opt_lighttrack track" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf opt_lighttrack \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.opt_lighttrack.dump.h5 \
> body25.tracked.opt_lighttrack.dump.log 2>&1 ; } 2>> results.txt

echo "deepsortlike track" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf deepsortlike \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25.dump.h5 body25.tracked.deepsortlike.dump.h5 \
> body25.tracked.deepsortlike.dump.log 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "opt_lighttrack track body25_all (not strictly part of the benchmark)" >> results.txt

{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop filter \
    --track \
    --track-conf opt_lighttrack \
    --pose-matcher-config /opt/skelshop/work/gcn_config.yaml \
    --shot-seg=psd \
    --segs-file breakingnews-Scenes.csv \
    body25all.dump.h5 body25all.tracked.opt_lighttrack.dump.h5 \
> body25all.tracked.opt_lighttrack.dump.log 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "dlib-hog-face5 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    dlib-hog-face5 breakingnews.mp4 breakingnews.dlibhogface5.h5 \
> breakingnews.dlibhogface5.log 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face5 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    dlib-cnn-face5 breakingnews.mp4 breakingnews.dlibcnnface5.h5 \
> breakingnews.dlibcnnface5.log 2>&1 ; } 2>> results.txt

echo "dlib-hog-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    dlib-hog-face68 breakingnews.mp4 breakingnews.dlibhogface68.h5 \
> breakingnews.dlibhogface68.log 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    dlib-cnn-face68 breakingnews.mp4 breakingnews.dlibcnnface68.h5 \
> breakingnews.dlibcnnface68.log 2>&1 ; } 2>> results.txt

echo "openpose-face3 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --from-skels body25.tracked.opt_lighttrack.dump.h5 \
    openpose-face3 breakingnews.mp4 breakingnews.openposeface3.h5 \
> breakingnews.openposeface3.log 2>&1 ; } 2>> results.txt

echo "openpose-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --from-skels body25all.tracked.opt_lighttrack.dump.h5 \
    openpose-face68 breakingnews.mp4 breakingnews.openposeface68.h5 \
> breakingnews.openposeface68.log 2>&1 ; } 2>> results.txt

echo >> results.txt
echo "(again but with bboxes/chips [slower])" >> results.txt
echo "dlib-hog-face5 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-hog-face5 breakingnews.mp4 breakingnews.dlibhogface5.chips.h5 \
> breakingnews.dlibhogface5.chips.log 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face5 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-cnn-face5 breakingnews.mp4 breakingnews.dlibcnnface5.chips.h5 \
> breakingnews.dlibcnnface5.chips.log 2>&1 ; } 2>> results.txt

echo "dlib-hog-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-hog-face68 breakingnews.mp4 breakingnews.dlibhogface68.chips.h5 \
> breakingnews.dlibhogface68.chips.log 2>&1 ; } 2>> results.txt

echo "dlib-cnn-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    dlib-cnn-face68 breakingnews.mp4 breakingnews.dlibcnnface68.chips.h5 \
> breakingnews.dlibcnnface68.chips.log  2>&1 ; } 2>> results.txt

echo "openpose-face3 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    --from-skels body25.tracked.opt_lighttrack.dump.h5 \
    openpose-face3 breakingnews.mp4 breakingnews.openposeface3.chips.h5 \
> breakingnews.openposeface3.chips.log 2>&1 ; } 2>> results.txt

echo "openpose-face68 face" >> results.txt
{ time \
singularity exec --nv ~/sifs/skelshop.sif python -m skelshop face \
    --write-bboxes --write-chip \
    --from-skels body25all.tracked.opt_lighttrack.dump.h5 \
    openpose-face68 breakingnews.mp4 breakingnews.openposeface68.chips.h5 \
> breakingnews.openposeface68.chips.log 2>&1 ; } 2>> results.txt

echo >> results.txt
