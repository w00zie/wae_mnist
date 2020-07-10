# Wasserstein Autoencoders for MNIST

This is a re-implementation in Tensorflow 2 (2.2.0) of both the WAE-GAN and WAE-MMD models proposed in [1] on the MNIST dataset.

## Random draws

These are a few samples decoded from the latent space. The quality of the generation is inferior to the original paper's [implementation](https://github.com/tolstikhin/wae).

- WAE-GAN

<p align="center">
  <img src="https://github.com/w00zie/wae_mnist/blob/master/saved_models/wae_gan/img/random.png" />
</p>

- WAE-MMD

<p align="center">
  <img src="https://github.com/w00zie/wae_mnist/blob/master/saved_models/wae_mmd/img/random.png" />
</p>

## Reconstructions

Even rows contain original samples while odd rows their relative reconstruction.

- WAE-GAN

<p align="center">
  <img src="https://github.com/w00zie/wae_mnist/blob/master/saved_models/wae_gan/img/recons.png" />
</p>

- WAE-MMD
  
<p align="center">
  <img src="https://github.com/w00zie/wae_mnist/blob/master/saved_models/wae_mmd/img/recons.png" />
</p>

## Downloading & running

```
git clone https://github.com/w00zie/wae_mnist.git
cd wae_mnist
```

Both architectures (WAE-GAN and WAE-MMD) are self contained in their respective directories (some code is duplicated, I'll refactor everything in the near future).

I provide a few (at the moment only one) trained models per architecture. You can explore the `saved_models/` directory, where, for each experiment, I provide the weights (`models/`), the hyperparameters (`config.json`) and the Tensorboard logs (`events.out.tfevents....`) that you can load and inspect by running `tensorboard --logdir saved_model/{experiment}/`.

Take a look at `test.py` for inference and if you want to train a new model just open `main.py`, set your hparams and run
```
python main.py
```

### Requirements
```
tensorflow >= 2.2.0
matplotlib
```

[1]: https://arxiv.org/pdf/1711.01558.pdf
