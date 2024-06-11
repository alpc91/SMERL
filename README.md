## Subequivariant Reinforcement Learning in 3D Multi-Entity Physical Environments ##
### ICML 2024
#### [[Project Page]](https://alpc91.github.io/SMERL/) [[Paper]]()

### Citation
If you use this codebase for your research, please cite the paper:

```
@inproceedings{chen2024smerl,
    title = {Subequivariant Reinforcement Learning in 3D Multi-Entity Physical Environments},
    author = {Chen, Runfa and Wang, ling and Du, Yu and Xue, Tianrui and Sun, Fuchun and Zhang, Jianwei and Huang, Wenbing},
    booktitle={International Conference on Machine Learning},
    year={2024},
    organization={PMLR}
}
```


### Installation
```bash
conda create -n jax python=3.10
conda activate jax
pip install --upgrade pip
pip install jax[cuda11_pip]==0.4.14 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  
pip install -r requirements.txt
```


### Team Reach
```bash
cd collaborative_team_tasks
python train.py
```

### Team Sumo
```bash
cd mix_colla_compe_team_tasks
python train.py
```


## Acknowledgement
The MARL code is based on [Brax](https://github.com/google/brax) and the morphology-based implementation is built on top of [MxT Bench (Furuta et al., ICLR 2023)](https://github.com/frt03/mxt_bench), [SGRL (Chen et al., ICML 2023 Oral)](https://github.com/alpc91/SGRL)repository.