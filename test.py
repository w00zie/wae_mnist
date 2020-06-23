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

    h_dim, z_dim = 64, 10
    disc_units = 512
    conv_kernel_size = (5,5)
    kernel_init = "TruncatedNormal"

    enc_chkpt = "saved_models/23_06_2020-09:11:19/models/encoder/encoder"
    dec_chkpt = "saved_models/23_06_2020-09:11:19/models/decoder/decoder"
    dis_chkpt = "saved_models/23_06_2020-09:11:19/models/discriminator/discriminator"

    enc, dec, dis = load_models(h_dim, z_dim,
                                disc_units,
                                conv_kernel_size,
                                kernel_init,
                                enc_chkpt,
                                dec_chkpt,
                                dis_chkpt)

    n_random_batches = 10

    img = []
    for n, _ in enumerate(range(n_random_batches)):
        z = tf.random.normal(shape=(16, 10), mean=0., stddev=1.)
        decoded_img = dec(z, training=False)
        img.append(tf.squeeze(tf.concat([decoded_img[i,:,:,:] for i in range(16)], axis=1), axis=-1))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave("saved_models/23_06_2020-09:11:19/img/random.png", img)

    _, test_dataset = get_dataset(batch_size=100)

    img = []
    for n, batch in test_dataset.take(10).enumerate():
        x_hat = dec(enc(batch, training=False), training=False)
        reals = tf.squeeze(tf.concat([batch[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        fakes = tf.squeeze(tf.concat([x_hat[i,:,:,:] for i in range(16)], axis=1), axis=-1)
        img.append(tf.concat([reals, fakes], axis=0))
    img = tf.concat([i for i in img], axis=0)
    plt.imsave("saved_models/23_06_2020-09:11:19/img/recons.png", img)
        