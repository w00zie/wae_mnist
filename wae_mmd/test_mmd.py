import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import os
import json
sys.path.insert(0, os.getcwd())

from data import get_dataset
from utils import get_ae


def load_models(config: dict,
                enc_chkpt: str,
                dec_chkpt: str):

    enc, dec = get_ae(config)
    
    enc.load_weights(enc_chkpt)
    dec.load_weights(dec_chkpt)

    return (enc, dec)

if __name__ == "__main__":
    import os
    import argparse
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='./saved_models/wae_mmd/')
    args = parser.parse_args()

    logdir = args.exp_dir
    with open(os.path(logdir,"config.json")) as json_file: 
        config = json.load(json_file) 

    enc_chkpt = os.path.join(logdir,"models","encoder","encoder")
    dec_chkpt = os.path.join(logdir,"models","decoder","decoder")

    enc, dec = load_models(config,
                           enc_chkpt,
                           dec_chkpt)

    n_random_batches = 10

    _, test_dataset = get_dataset(batch_size=config["batch_size"])

    # These two following blocks of code are responsible for:

    ## Sampling the latent space and decoding a few images, then stacking these
    ## decoded samples into a png and saving it.
    img = []
    for n, _ in enumerate(range(n_random_batches)):
        z = tf.random.normal(shape=(16, config["z_dim"]), stddev=config["sigma_z"])
        decoded_img = dec(z, training=False)
        img.append(tf.squeeze(tf.concat([decoded_img[i,:,:,:] for i in range(16)], axis=1), axis=-1))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave(os.path.join(logdir,"img","random.png"), img)

    ## Sampling the real (test) data distribution and reconstructing these samples,
    ## stacking them into a png and saving it.
    img = []
    for n, batch in test_dataset.take(n_random_batches).enumerate():
        x_hat = dec(enc(batch, training=False), training=False)
        reals = tf.squeeze(tf.concat([batch[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        fakes = tf.squeeze(tf.concat([x_hat[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        img.append(tf.concat([reals, fakes], axis=0))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave(os.path.join(logdir,"img","recons.png"), img)