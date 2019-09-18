<!-- git clone git@github.com:seblee97/maml.git

cd maml -->

Once cloned, run the following command:

pip install -r requirements.txt

To run experiments, run the following command:

source experiment.sh

# Task-Weighted MAML Repository

## Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Structure of repository](#repository-structure)

### Summary

This repository contains code for the [model-agnostic meta-learning algorithm](https://arxiv.org/pdf/1703.03400.pdf) (Finn et al.) including investigation
into modifying the task sampling regime. Rather than sampling uniformly, we can sample according to a probability distribution that we
construct using a priority queue containing losses previously seen during training. So far the following sampling regimes have been implemented:

* Uniform (vanilla)
* Epsilon Greedy (max with certain probability, random otherwise)
* Weighted Sample (sample according to probability distribution given by priority queue)
* Importance Weighted Sample (same as above but meta update is now weighted by importance weights)
* Delta Sample (sample according to probability distribution given by change in priority queue - biases sample to parameter space in which progress has been made)
* Importance Delta Sample (same as above but meta update is now weighted by importance weights)

The priority queue is essentially a mapping between parameter tuples and losses where the parameters are those governing the task distribution (e.g. in sinusoidal regression the parameters are the phase shift and amplitude scaling). Each time a task is sampled in the inner loop, the parameter associated with this task in the priority queue will be updated with the loss incurred.

So far the following tasks have been implemented:

* 2D sinusoidal regression (parameters: phase shift, amplitude)
* 3D sinusoidal regression (parameters: phase shift, amplitude & frequency scaling)

I hope to add implementations for image classification and control tasks soon. 

### Installation

### Experiments

### Repository Structure

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