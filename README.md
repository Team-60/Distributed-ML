# Distributed-ML

Implementing an optimizer (_Nesterov SGD_) for training a CNN model on CIFAR-10 in the following settings:
* Shared memory [Hogwild!](https://arxiv.org/abs/1106.5730) (Directory: `hogwild`)
* Distributed [Local-SGD](https://openreview.net/pdf?id=S1g2JnRcFX) (Directory: `Local-SGD`)

`optimizer-benchmarks` contains benchmarks for various first-order and second-order based GD methods:
* SGD
* Momentum SGD
* Nesterov SGD
* Adagrad
* RMSProp
* ADAM



