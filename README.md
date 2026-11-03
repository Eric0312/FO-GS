# Learning Fractional-Order Dynamics from a Single Trajectory

This repository contains the synthetic experiments and EEG experiments for Figures 2–4 and Tables 1–2. Use Python 3.10 or later and install the required packages:

```bash
python -m pip install numpy scipy pandas matplotlib jupyterlab
```

Run each notebook from its own experiment directory, with that directory as the working directory.

## Synthetic Experiment

### Figure 2

Open the notebooks in `Synthetic_Experiment/` and run their cells in order:

- `Vary_t.ipynb`: estimation errors versus trajectory length.
- `Vary_sigma.ipynb`: estimation errors versus noise scale.
- `Vary_M.ipynb`: estimation errors versus the number of grid points.

These notebooks compare *FO-GS*, *FO-BS*, and *FO-WT* for the fractional orders and the system matrix. They generate the corresponding Figure 2 panels and save plots and numerical summaries in experiment output folders. The trajectory length and grid size notebooks run multiple system instance seeds and save a separate set of results for each seed.

### Tables 1 and 2

The scripts are in `Synthetic_Experiment/Tables_1_2/`. From that directory, run:

```bash
python experiments.py
python reproduce_tables.py
```

The first command generates the fixed seed *FO-GS* experiments for both tables and saves simulation results in `regenerated/`. The second command fits the convergence rates and saves CSV and LaTeX tables in `results/`. Table 1 uses short horizons for small and large fractional orders. Table 2 uses extended horizons up to 25,600 for small fractional orders, with both full-range and long-horizon fits.

The simulations use five systems and 20 rollouts per system. Long-horizon runs may take substantial time. Completed rollouts are checkpointed, so rerunning the same command resumes the experiment. 

## EEG Experiment

### Figures 3 and 4

The `EEG_Experiment/` directory includes the subject data files `s00.csv` through `s35.csv` and two notebooks:

1. Run `EEG_case_final.ipynb` from `EEG_Experiment/`. It reads the subject CSV files, evaluates *FO-GS*, *FO-BS*, and *FO-WT* using non-overlapping windows of 150 samples and a 70%/30% train/test split, and saves the evaluation summaries in `eeg_case_study_outputs_no_normalization/`.
2. Run `EEG_plot_final.ipynb` from the same directory. It reads the generated subject-level summary and creates `eeg_test_nmse_comparison.pdf` for Figure 3 and `eeg_train_nmse_comparison.pdf` for Figure 4.

The plots report subject-level average one-step normalized mean squared error (NMSE).

# Citation

```bash
@misc{zhang2026learningfractionalorderdynamicssingle,
      title={Learning Fractional-Order Dynamics from a Single Trajectory}, 
      author={Xiaole Zhang and Ziyi Zhang and Zehao Zhao and Stephen Tu and Guannan Qu and Yorie Nakahira and Paul Bogdan},
      year={2026},
      eprint={2609.18127},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.18127}, 
}
```
