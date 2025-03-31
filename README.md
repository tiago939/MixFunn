# MixFunn: a Neural Network Architecture for Differential Equations v0.1.0

Welcome to the MixFunn repository! This repository contains the code for the MixFunn model, a neural network architecture designed to solve differential equations with improved generalization and interpretability. The model leverages mixed-function and second-order neurons to provide a compact, robust, and interpretable solution to complex differential equations.

## Overview

MixFunn introduces two key concepts:

- Mixed-Function Neurons: These neurons combine multiple parameterized nonlinear functions to improve representational flexibility.

- Second-Order Neurons: These incorporate both a linear transformation and a quadratic term to capture interactions between input variables.

Together, these features allow MixFunn to solve differential equations with fewer parameters, better generalization to unseen domains, and the ability to extract interpretable analytical expressions.

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/MixFunn.git
cd MixFunn
```

## Usage

The script mixfunn.py contains the second-order and mixed-function neuron modules. For the mixed-function neuron, this script includes a built-in list of predefined functions. To add, remove, or edit the functions that this neuron uses, simply modify the list inside mixfunn.py.

To run an experiment, navigate to the examples/ directory and execute one of the train scripts. Before running an experiment, make sure to copy the mixfunn.py script into the examples/system folder. For instance, to run the damped harmonic oscillator experiment:

```
python train.py
```

Each experiment script is designed to be self-contained, allowing you to modify hyperparameters, select different differential equations, or incorporate your own datasets. The code is modular to facilitate experimentation and extension.

For your own experiments, you can build entire neural networks with second-order neurons and mixed-function neurons. For example:

```
import torch
import torch.nn as nn
import mixfunn as mf

class MixFunn(nn.Module):
    
    def __init__(self):
        super(MixFunn, self).__init__()

        self.layers = nn.Sequential(
            mf.Quad(2, 4),
            mf.MixFun(4, 5, second_order_function=True),
            mf.Quad(5, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x
```

This code builds a neural network with three layers:

- The first layer consists of four second-order neurons.

- The second layer outputs five mixed-function neurons.

- The last layer outputs a single second-order neuron.

Feel free to modify and extend the architecture to suit your experimental needs.

## Citation

If you use MixFunn in your research, please cite our work:

```
@misc{2025mixfunn,
      title={MixFunn: A Neural Network for Differential Equations with Improved Generalization and Interpretability}, 
      author={Tiago de Souza Farias and Gubio Gomes de Lima and Jonas Maziero and Celso Jorge Villas-Boas},
      year={2025},
      eprint={2503.22528},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.22528}, 
}
```

## Contact

For questions or suggestions, please open an issue on GitHub or contact us at tiago.farias@ufscar.br
