for i in {0..2}; do
    job_script="job_script_${i}.sh"
    
    cat > "$job_script" <<EOF
#!/bin/bash
#PBS -N EmuBC
#PBS -lselect=1:ncpus=62:mem=256gb

cd /home/ljy/BettiCurveCosmo/Emulator
python3 auto-sklearn.py --trial_index 7 --dim ${i} --data_path "../Data/EmulatorData/nwLH_fof_emulator_dimensionless_rsdz_[(1,6),(2,15),(9,19)].bc"

EOF
    qsub "$job_script"
done

wait

rm -f job_script*.sh