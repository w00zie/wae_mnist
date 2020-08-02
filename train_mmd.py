import tensorflow as tf
import matplotlib.pyplot as plt

import os
import json
from datetime import datetime, timedelta
from time import time

from data import get_dataset
from utils import get_ae

class Train:

    def __init__(self, config):
        
        self.h_dim = config["h_dim"]
        self.z_dim = config["z_dim"]
        self.epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.sigma_z = config["sigma_z"]
        self.lmbda = config["lambda"]

        # Experiment directory
        self.logdir = os.path.join("runs", datetime.now().strftime("wae_mmd_%d_%m_%Y-%H:%M:%S"))
        self.writer = tf.summary.create_file_writer(self.logdir)
        with self.writer.as_default():
            tf.summary.text("Hyperparams", json.dumps(config), step=0)
        self.writer.flush()
        
        os.mkdir(os.path.join(self.logdir,"img"))
        os.mkdir(os.path.join(self.logdir,"img", "random"))
        os.mkdir(os.path.join(self.logdir,"img","recons"))
        os.mkdir(os.path.join(self.logdir,"models"))
        os.mkdir(os.path.join(self.logdir,"models","encoder"))
        os.mkdir(os.path.join(self.logdir,"models","decoder"))
        with open(os.path.join(self.logdir,"config.json"), "w") as f:
            json.dump(config, f)

        # Models ================================================================
        self.encoder, self.decoder = get_ae(config)

        # Optimizer =============================================================
        ae_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["ae_lr"],
            decay_steps=config["ae_dec_steps"],
            decay_rate=config["ae_dec_rate"])

        self.enc_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.dec_optim = tf.keras.optimizers.Adam(ae_scheduler)

        # Data ==================================================================
        tf.print("Loading data...")
        self.train_dataset, self.test_dataset = \
            get_dataset(batch_size=self.batch_size)
        tf.print("Done.")

        # Metric trackers =======================================================
        self.avg_mse_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mmd_test = tf.keras.metrics.Mean(dtype=tf.float32)

    # Prior =====================================================================
    def sample_pz(self, batch_size: tf.Tensor) -> tf.Tensor:
        return tf.random.normal(shape=(batch_size, self.z_dim),
                                stddev=tf.sqrt(self.sigma_z))

    # Losses ====================================================================
    def mmd_penalty(self, 
                    pz: tf.Tensor, 
                    qz: tf.Tensor, 
                    batch_size: tf.Tensor) -> tf.Tensor:
        """This method calculates the unbiased U-statistic estimator of
        the MMD with the IMQ kernel. It's taken from
        https://github.com/tolstikhin/wae/blob/master/wae.py#L233

        Here the property that the sum of positive definite kernels is 
        still a p.d. kernel is used. Various kernels calculated at different
        scales are summed together in order to "simultaneously look at various
        scales" [https://github.com/tolstikhin/wae/issues/2].
        """
        norms_pz = tf.reduce_sum(tf.square(pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(pz, pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(qz, qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(qz, pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        cbase = tf.constant(2. * self.z_dim * self.sigma_z)
        stat = tf.constant(0.)
        nf = tf.cast(batch_size, dtype=tf.float32)

        for scale in tf.constant([.1, .2, .5, 1., 2., 5., 10.]):
            C = cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(batch_size))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    def total_loss(self, 
                   batch: tf.Tensor, 
                   x_hat: tf.Tensor, 
                   pz: tf.Tensor, 
                   qz: tf.Tensor, 
                   batch_size: tf.Tensor) -> tf.Tensor:
        c = tf.reduce_mean(tf.reduce_sum(tf.square(batch - x_hat), axis=[1,2,3]))
        penalty = self.mmd_penalty(pz, qz, batch_size)
        return c + self.lmbda * penalty

    # Optimization step =========================================================
    @tf.function
    def train_enc_dec(self, batch: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(batch)[0]
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            qz = self.encoder(batch, training=True)
            x_hat = self.decoder(qz, training=True)
            pz = self.sample_pz(batch_size)
            enc_dec_loss = self.total_loss(batch, x_hat, pz, qz, batch_size)
        enc_grads = enc_tape.gradient(enc_dec_loss, self.encoder.trainable_variables)
        dec_grads = dec_tape.gradient(enc_dec_loss, self.decoder.trainable_variables)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        return enc_dec_loss

    # One-epoch train steps =====================================================
    def validation_step(self):
        # Reset metrics
        self.avg_enc_dec_test_loss.reset_states()
        self.avg_mse_test_loss.reset_states()
        self.avg_mmd_test.reset_states()
        # Run through an epoch of validation
        for batch in self.test_dataset:
            batch_size = tf.shape(batch)[0]
            qz = self.encoder(batch, training=False)
            x_hat = self.decoder(qz, training=False)
            c = tf.reduce_sum(tf.square(batch - x_hat), axis=[1,2,3])
            pz = self.sample_pz(batch_size)
            mmd_penalty = self.mmd_penalty(pz, qz, batch_size)
            # Log metrics for every batch
            self.avg_mse_test_loss(c)
            self.avg_enc_dec_test_loss(c + self.lmbda*mmd_penalty)
            self.avg_mmd_test(mmd_penalty)
        return (
            self.avg_enc_dec_test_loss.result(), 
            self.avg_mse_test_loss.result(),
            self.avg_mmd_test.result()
        )

    def train_step(self):
        # Reset metrics
        self.avg_enc_dec_train_loss.reset_states()
        # Run through an epoch of training
        for batch in self.train_dataset:
            enc_dec_loss = self.train_enc_dec(batch)
            # Log metrics for every batch
            self.avg_enc_dec_train_loss(enc_dec_loss)
        return self.avg_enc_dec_train_loss.result()

    # Plot utils ================================================================
    def log_hist(self, step):
        for w in self.encoder.trainable_weights:
            tf.summary.histogram("Encoder/{}".format(w.name), w, step=step)
        for w in self.decoder.trainable_weights:
            tf.summary.histogram("Decoder/{}".format(w.name), w, step=step)

    def display_random_bar(self, epoch):
        p_z = self.sample_pz(tf.constant(8))
        x_hat = self.decoder(p_z, training=False)
        img = tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=1)
        tf.summary.image("Random", tf.expand_dims(img, axis=0), step=epoch)
      
    def display_reconstruction(self, epoch):
        for batch in self.test_dataset.take(1):
            x_hat = self.decoder(self.encoder(batch, training=False), training=False)
            reals = tf.concat([batch[i,:,:,:] for i in range(8)], axis=1)
            fakes = tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=1)
            img = tf.concat([reals, fakes], axis=0)
            tf.summary.image("Reconstruction", tf.expand_dims(img, axis=0), step=epoch)

    # General utils =============================================================
    def save_weights(self):
        self.encoder.save_weights(self.logdir+"/models/encoder/encoder")
        self.decoder.save_weights(self.logdir+"/models/decoder/decoder")

    # Training procedure ========================================================
    def train(self):
        with self.writer.as_default():
            start = time()
            for epoch in range(self.epochs):
                # Train for one epoch
                e_d_loss = self.train_step()
                # Display a few bars
                if epoch % 2 == 0:
                    self.display_random_bar(epoch)
                    self.display_reconstruction(epoch)
                if epoch % 10 == 0:
                    self.save_weights()
                # Validate on the test set
                test_loss, mse, mmd = self.validation_step()
                # Logging
                tf.summary.scalar("Train/Encoder-Decoder Loss", 
                                  e_d_loss, step=epoch)
                tf.summary.scalar("Learning Rates/Encoder", 
                                  self.enc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Decoder", 
                                  self.dec_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Test/Encoder-Decoder Loss", 
                                  test_loss, step=epoch)
                tf.summary.scalar("Test/MSE", mse, step=epoch)
                tf.summary.scalar("Test/MMD", mmd, step=epoch)
                self.log_hist(epoch)

                tf.print("Epoch {}/{}  -  [train_loss = {}]  -  [test_loss = {}]  -  {}".\
                    format(epoch, self.epochs-1, 
                           e_d_loss.numpy(), test_loss.numpy(),
                           datetime.now().strftime('%H:%M:%S')))

        print(f"Training took {timedelta(seconds=time()-start)}")
        self.save_weights()
        self.writer.flush()

if __name__ == "__main__":
    
    from wae_mmd.config_mmd import config_mmd
    # Logging only (W)arnings and (E)rrors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    train = Train(config_mmd)
    train.train()