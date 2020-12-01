-die kommandos -mp, --list, show-graph, --delete-all-output!


rm *.out && rm -r work small_dataset/snek && rm hpc_run

./run_coord.sh poetry run snakemake --list

./run_coord.sh tracked_all --config VIDEO_BASE=/home/cxs800/small_dataset/ DUMP_BASE=/home/cxs800/small_dataset/snek -p &> hpc_run & 

./run_coord.sh tracked_all --delete-all-output --config VIDEO_BASE=/home/cxs800/small_dataset/ DUMP_BASE=/home/cxs800/small_dataset/snek
./run_coord.sh tracked_all --unlock --config VIDEO_BASE=/home/cxs800/small_dataset/ DUMP_BASE=/home/cxs800/small_dataset/snek

[bash] scp cxs800@hpclogin.case.edu:~/*.out /home/chris/Documents/JOBS/Uhrig-Gesture-Recog/data/tiny_hpc
scp -r cxs800@hpclogin.case.edu:~/{tiny_dataset,work} /home/chris/Documents/JOBS/Uhrig-Gesture-Recog/data/tiny_hpc
