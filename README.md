# Distributed-ML

Implementing an optimizer (_Nesterov SGD_) for training a CNN model on the [CIFAR-10](https://www.kaggle.com/c/cifar-10) dataset in the following settings:
* Shared memory [Hogwild!](https://arxiv.org/abs/1106.5730) | Directory: `hogwild`
* Distributed [Local-SGD](https://openreview.net/pdf?id=S1g2JnRcFX) | Directory: `Local-SGD`

Directory `optimizer-benchmarks` contains benchmarks for various first-order and second-order based GD methods:
* SGD
* Momentum SGD
* Nesterov SGD
* Adagrad
* RMSProp
* ADAM



