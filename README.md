## Transfer Learning on Heterogeneous Feature Spaces for Treatment Effects Estimation

### Ioana Bica, Mihaela van der Schaar

#### Neural Information Processing Systems (NeurIPS) 2022


## HTCE-learners

PyTorch implementation for the HTCE-learners.  

Note that the code in contrib/ for CATENets [1, 2] is from:  https://github.com/AliciaCurth/CATENets
and the code for RadialGAN [3] is adapted from: https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/RadialGAN 


To reproduce the experiments in the paper (for the Twins dataset), use the following commands:
- Benchmark comparison (Table 1)
```
python run_experiments.py --experiment_name="baseline_experiment"
```
- Varying the information sharing between domains (Figure 6 - top)
```
python run_experiments.py --experiment_name="po_sharing_across_domains"
```
- Varying the target dataset size (Figure 6 - bottom)
```
python run_experiments.py --experiment_name="target_size"
```
- Effect of selection bias (Figure 7)
```
python run_experiments.py --experiment_name="selection_bias"
```

The results are saved in results/. To plot the results from the paper, use the Jupyter notebook in results/results_figs/analyze_results.ipynb.


#### References
[1] Curth, Alicia, and Mihaela van der Schaar. "Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms". International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

[2] Curth, Alicia, and Mihaela van der Schaar. "On inductive biases for heterogeneous treatment effect estimation". Advances in Neural Information Processing Systems, 2021.

[3] Yoon, Jinsung, James Jordon, and Mihaela Schaar. "RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks". International Conference on Machine Learning. PMLR, 2018.

