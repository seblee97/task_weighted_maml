<!-- git clone git@github.com:seblee97/maml.git

cd maml -->

Once cloned, run the following command:

pip install -r requirements.txt

To run experiments, run the following command:

source experiment.sh

# Repo

**Summary:**

# Table of Contents
1. [Installation](#installation)
2. [Structure of repository](#repository-structure)

## Installation


## Repository Structure

```
│
├── setup.py
│
├── requirements.txt
│
├── README.md
│     
├── docs
│     
├── experiments
│    │
│    │
│    ├── configs
│    │   │
│    │   ├── base_config.yaml
│    │   ├── test_base_config.yaml
│    │   │
│    │   ├── maml_config.yaml
│    │   ├── pq_maml_config.yaml
│    │   ├── pq_sample_maml_config.yaml
│    │   ├── pq_importance_maml_config.yaml
│    │   ├── pq_sample_delta.yaml
│    │   └── pq_importance_sample_delta.yaml
│    │
│    ├── results
│    │   │
│    │   └── **result files (not tracked/commited)**
│    │
│    ├── __init__.py 
│    ├── context.py
│    ├── experiment.sh
│    ├── kill_experiments.sh
│    └── main.py
│     
├── jax_maml
│    │
│    │
│    ├── __init__.py 
│    ├── jax_model.py 
│    └── jax_sinusoid.py
│     
├── maml
│    │
│    │
│    ├── __init__.py 
│    ├── model.py 
│    └── sinusoid.py
│     
├── tests
│    │
│    │
│    ├── test_configs
│    │   │
│    │   ├── test_base_config.yaml
│    │   └── test_maml_config.yaml
│    │
│    ├── __init__.py 
│    ├── context.py
│    ├── test_base_priority_queue.py
│    └── test_sin_priority_queue.py
│     
└── utils
     │
     │
     ├── __init__.py 
     ├── custom_functions.py
     ├── parameters.py 
     └── priority.py             
```