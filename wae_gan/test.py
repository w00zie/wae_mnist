import tensorflow as tf
from model import Encoder, Decoder, Discriminator
from data import get_dataset
import matplotlib.pyplot as plt

def load_models(h_dim: int,
                z_dim: int,
                disc_units: int,
                kernel_size: tuple,
                kernel_init: str,
                enc_chkpt: str,
                dec_chkpt: str,
                dis_chkpt: str):

    enc = Encoder(h_dim=h_dim, z_dim=z_dim, 
                kernel_size=conv_kernel_size, 
                kernel_init=kernel_init)

    dec = Decoder(h_dim=h_dim, z_dim=z_dim, 
                kernel_size=conv_kernel_size,
                kernel_init=kernel_init)

    dis = Discriminator(units=disc_units, z_dim=z_dim, 
                        kernel_init=kernel_init)
    
    enc.load_weights(enc_chkpt)
    dec.load_weights(dec_chkpt)
    dis.load_weights(dis_chkpt)

    return (enc, dec, dis)

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    logdir = "../saved_models/wae_gan/"
    with open(logdir+'config.json') as json_file: 
        config = json.load(json_file) 

    h_dim, z_dim = config['h_dim'], config['z_dim']
    conv_kernel_size = config['conv_kernel_size']
    kernel_init = "TruncatedNormal"
    sigma_z = config['sigma_z']

    enc_chkpt = logdir+"models/encoder/encoder"
    dec_chkpt = logdir+"models/decoder/decoder"

    enc, dec, dis = load_models(h_dim, z_dim,
                           conv_kernel_size,
                           kernel_init,
                           enc_chkpt,
                           dec_chkpt)

    n_random_batches = 10

    img = []
    for n, _ in enumerate(range(n_random_batches)):
        z = tf.random.normal(shape=(16, 10), mean=0., stddev=1.)
        decoded_img = dec(z, training=False)
        img.append(tf.squeeze(tf.concat([decoded_img[i,:,:,:] for i in range(16)], axis=1), axis=-1))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave(logdir+"img/random.png", img)

    _, test_dataset = get_dataset(batch_size=100)

    img = []
    for n, batch in test_dataset.take(10).enumerate():
        x_hat = dec(enc(batch, training=False), training=False)
        reals = tf.squeeze(tf.concat([batch[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        fakes = tf.squeeze(tf.concat([x_hat[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        img.append(tf.concat([reals, fakes], axis=0))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave(logdir+"img/recons.png", img)
        