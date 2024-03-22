## Codes for the TMLR paper [_Effective Latent Differential Equation Models via Attention and Multiple Shooting_](https://openreview.net/pdf?id=uxNfN2PU1W)

### Video Presentation
[![Video Presentation](https://img.youtube.com/vi/XYv10fuumCQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=XYv10fuumCQ)

### Instructions for running the experiments

1. Download and install Julia 1.8.5:  
https://julialang.org/downloads/oldreleases/

3. Install all dependencies:  
`julia Experiments/scripts/install_dependencies.jl`

4. Generate train and test datasets:  
`julia Experiments/scripts/datasets_generation.jl`

5. Train models  
`julia Experiments/scripts/Training/train_goku.jl`  
`julia Experiments/scripts/Training/Baselines/LSTMs/train_lstm.jl`  
`julia Experiments/scripts/Training/Baselines/LatentODE/train_latent_ode.jl`

Notice that when ran on a personal computer, these trainings could take weeks. Alternatively, if you have access to a SLURM cluster, you can adapt the corresponing batch scripts to your setup and run the trainings there:

`sbatch Experiments/scripts/Training/batch_job_goku.sh`  
`sbatch Experiments/scripts/Training/Baselines/LSTMs/batch_job_lstm.sh`  
`sbatch Experiments/scripts/Training/Baselines/LatentODE/batch_job_latent_ode.sh`  

5. Evaluate trained models on test data  
`julia Experiments/scripts/Evaluation/evaluate.jl`

6. Plot results and perform statistical tests  
`julia Experiments/scripts/Evaluation/statistics_and_plots.jl`

### Data availability

Due to privacy limitations to share the fMRI dataset used in the study, this repository includes only the experiments conducted on the simulated Stuart-Landau oscillators.

### Citation
If you find this codebase useful, please consider citing:

```bibtex
@article{abrevaya2024effective,
  title = {Effective Latent Differential Equation Models via Attention and Multiple Shooting},
  author = {Germ{\'a}n Abrevaya and Mahta Ramezanian-Panahi and Jean-Christophe Gagnon-Audet and Pablo Polosecki and Irina Rish and Silvina Ponce Dawson and Guillermo Cecchi and Guillaume Dumas},
  journal = {Transactions on Machine Learning Research},
  issn = {2835-8856},
  year = {2024},
  url = {https://openreview.net/forum?id=uxNfN2PU1W}
}
```