import os
import subprocess


modelname1 = "MLP"
modelname2 = "SHNN"
envs = ['2ants']  

sds = [0,6,8,24,77,81,520,999,1024,2048,32767]


for sd in sds:
  for env in envs:
            command = f"CUDA_VISIBLE_DEVICES=0 python train_mappo_mix.py --logdir ../../results/mode_test/{env}/{modelname1}_vs_{modelname2} --seed {sd} --env {env} --total_env_steps 20000000 --eval_frequency 50 --setname ao_mappo_mix_{modelname1}_vs_{modelname2}_seed{sd} --modelname1 {modelname1} --modelname2 {modelname2}"

            os.system(command)