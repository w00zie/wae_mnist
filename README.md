# Wasserstein Autoencoders for MNIST

This is a re-implementation in Tensorflow 2 (2.2.0) of the WAE-GAN model proposed in [1] using the MNIST dataset.

## Random draws

These are a few samples decoded from the latent space. The quality of the generation is inferior to the original paper's [implementation](https://github.com/tolstikhin/wae).

<p align="center">
  <img src="/wae_mnist/saved_models/23_06_2020-09:11:19/img/random.png" />
</p>

## Reconstructions

Even rows contain original samples while odd rows their relative reconstruction.

<p align="center">
  <img src="/wae_mnist/saved_models/23_06_2020-09:11:19/img/recons.png" />
</p>

## Downloading & running

```
git clone https://github.com/w00zie/wae_mnist.git
cd wae_mnist
```
I provide a few (at the moment only one) trained models. You can explore the `saved_model/` directory, where, for each experiment, I provide the weights (`models/`), the hyperparameters (`config.json`) and the Tensorboard logs (`events.out.tfevents....`) that you can load by running `tensorboard --logdir saved_model/{experiment}/`.

Take a look at `test.py` for inference and if you want to train a new model just open `main.py`, set your hparams and run
```
python main.py
```


### Requirements
```
tensorflow >= 2.2.0
numpy
matplotlib

```

[1]: https://arxiv.org/pdf/1711.01558.pdf
