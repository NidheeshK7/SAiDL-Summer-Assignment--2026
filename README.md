# SAiDL Summer Assignment 2026

This repository contains my attempt at the SAiDL Summer CoreML and Mechanistic Interpretability assignments.

## Architecture

```text
SAiDL-Summer-Assignment--2026/
├── coreml/
│   ├── AFTvariants.py #contains the 4 Attention Free Transformer Implementations
│   ├── AttentionVariants.py #contains MHA, SWA, GQA, Linear Attention and both the convolution variants implementation
│   ├── HybridModel.py #contains the model for the convolution+attention hybrid
│   ├── PositionalVariants.py #contains the implementations of all three positional variants
│   ├── config.py
│   ├── data_setup.py
│   ├── eval_extrapolation.py #code for length extrapolation test
│   ├── model.py
│   ├── run_experiments.py #main file used to run experiments for all tasks
│   ├── train.py 
│   ├── requirements.txt
│   ├── README.md
│   └── Report/
└── mechinterp/
    ├── mech_exp.py #code for Mechanistic Explanation part of task 4, saves json file task4_part1_metrics which can be found at results/json
    ├── pipeline_setup.py #code to train the Top-K SAE
    ├── quantization_analysis.py #runs code for task 2 and saves metrics to metrics_m512 and metrics_m1024 which can be found at results/json
    ├── representation_damage.py #code for representation damage part of task 3, saves json file task3_metrics_method2 which can be found at results/json
    ├── robust_quantization.py #Implements subspace preserving quantization and saves json files metrics_m512_task4 and metrics_m1024_task4 which can be found at results/json
    ├── spectral_analysis.py #code for spectral analysis part of task 3 saves json file spectral_analysis_metrics which can be found at results/json
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── Report/
    └── results/
```


