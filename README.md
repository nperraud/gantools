# GANtools

Gantools is a set of classes written in Tensorflow to work with Generative Adversarial Netork (GAN). Their exist many other framework to do this. The reason I coded this one is to have a full control over every parts of the network.

Warning: the code is insuficiently documented, not entirely tested and may contains some bugs.

In the repository, ... , you will find a demonstration notebook.


## Installation

Since this is intented as a package for developpement, we highly suggest to install it with a simlink.

```
git clone git@github.com:nperraud/gantools.git
cd gantools
pip install -r requirements.txt
pip install -e .
```

## Oganisation of the code

The code is composed of a package named *gantools*. It is composed of the following submodules:
* gansystem: implement the basic training and generating system for a gan
* modely: contains the different network architecture
* data: data module
* blocks: basic tensorflow units
* utils: useful functions
* metrics: computation of the different error functions
* plot: helper for the different plots

## Some code using gantools

* TiFGAN https://github.com/tifgan/stftGAN
*
*

