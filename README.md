<!-- git clone git@github.com:seblee97/maml.git

cd maml -->

# Task-Weighted MAML Repository

## Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)

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

Implementations for image classification and control tasks will hopefully be added soon. 

This repository uses [Jax](https://github.com/google/jax) for the MAML implementation.  

### Installation

Clone this repository. Then run 

```pip install -r requirements.txt```

The primary requirements are:

* jax
* numpy
* matplotlib (for visualisation)
* tensorboardX (for visualisation)

### Experiments

Individual experiments can be run with the following command:

```python main.py -config *relative path to config choice*```

Config files are in the config folder. Variables for the experiment generally can be changed in the base_config.py file. Variables specific to the sampling regime used can be changed in the specific config files.

To run the full suite of experiments, use the following command:

```source experiment.sh```

Note, currently jax does not support multiple GPU support and by default GPU memory is pre-allocated so running multiple experiments simulataneously will likely not be possible when running in GPU mode depending on the size of your GPU.

To monitor experiments, you can use tensorboard. By default log files are in the results folder under experiments.

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