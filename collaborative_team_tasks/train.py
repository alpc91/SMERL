import os
import subprocess


modelnames = ["SHNN"]#, "MLP"]
envs = ['1ant']#'1centipede','2claws','2unimals','2antclaw','2ants','3acc'
sds = [0,6,8,24,77,81,520,999,1024,2048,32767]

for sd in sds:
  for env in envs:
    for modelname in modelnames:
            command = f"CUDA_VISIBLE_DEVICES=0 python train_mappo_colla.py --logdir ../../results/test/{env}/{modelname} --seed {sd} --env {env} --total_env_steps 50000000 --eval_frequency 50 --setname ao_mappo_col_{modelname}_seed{sd} --modelname {modelname}"

            os.system(command)
