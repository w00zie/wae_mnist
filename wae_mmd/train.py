import tensorflow as tf
from model import Encoder, Decoder
#from func_model import encoder, decoder
from data import get_dataset
import os
import json
from datetime import datetime, timedelta
from time import time
import matplotlib.pyplot as plt

class Train:

    def __init__(self, 
                 batch_size=100, 
                 epochs=10,
                 h_dim=32,
                 z_dim=10,
                 conv_kernel_size=(5,5),
                 kernel_init="glorot_normal",
                 sigma_z=1.,
                 enc_dec_lr=1e-3,
                 lmbda=1):
        
        # Experiment directory
        self.logdir = "runs/" + datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
        config = {"h_dim": h_dim,
                  "z_dim": z_dim,
                  "epochs": epochs,
                  "conv_kernel_size": conv_kernel_size,
                  "sigma_z": sigma_z,
                  "encoder/decoder learning rate": enc_dec_lr,
                  "exp_dir": self.logdir,
                  "lambda": lmbda}

        self.writer = tf.summary.create_file_writer(self.logdir)
        with self.writer.as_default():
            text = """# Hyperparameters used
                      1. h_dim = {}
                      2. z_dim = {}
                      3. kernel_size = {}
                      5. sigma_z = {}
                      7. ae_lr = {}
                      8. lambda = {}""".format(h_dim, z_dim, 
                                                conv_kernel_size,
                                                sigma_z,
                                                enc_dec_lr, lmbda)
            tf.summary.text("Hyperparams", text, step=0)
        self.writer.flush()
        os.mkdir(self.logdir+"/img")
        os.mkdir(self.logdir+"/img/random")
        os.mkdir(self.logdir+"/img/recons")
        os.mkdir(self.logdir+"/models")
        os.mkdir(self.logdir+"/models/encoder")
        os.mkdir(self.logdir+"/models/decoder")
        with open(self.logdir+"/config.json", "w") as f:
            json.dump(config, f)

        self.h_dim = h_dim #tf.constant(h_dim)
        self.z_dim = z_dim #tf.constant(z_dim)
        self.epochs = epochs
        self.batch_size = batch_size #tf.constant(batch_size)
        self.sigma_z = sigma_z #tf.constant(sigma_z, dtype=tf.float32)
        self.lmbda = lmbda #tf.constant(lmbda, dtype=tf.float32)

        # Models --------------------------------------------------------------
        self.encoder = Encoder(h_dim=h_dim, 
                               z_dim=z_dim, 
                               kernel_size=conv_kernel_size)
        self.decoder = Decoder(h_dim=h_dim, 
                               z_dim=z_dim, 
                               kernel_size=conv_kernel_size)

        # Optimizer -----------------------------------------------------------
        ae_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=enc_dec_lr,
            decay_steps=10000,
            decay_rate=0.95)

        self.enc_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.dec_optim = tf.keras.optimizers.Adam(ae_scheduler)

        # Data -----------------------------------------------------------------
        tf.print("Loading data...")
        self.train_dataset, self.test_dataset = get_dataset(batch_size=batch_size)
        tf.print("Done.")

        # Metric trackers ------------------------------------------------------
        self.avg_mse_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mmd_test = tf.keras.metrics.Mean(dtype=tf.float32)

    def sample_pz(self, batch_size: tf.Tensor) -> tf.Tensor:
        return tf.random.normal(shape=(batch_size, self.z_dim),
                                stddev=tf.sqrt(self.sigma_z))

    def mmd_penalty(self, pz: tf.Tensor, qz: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
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
        c = tf.math.reduce_mean(tf.reduce_sum(tf.square(batch - x_hat), axis=[1,2,3]))
        penalty = self.mmd_penalty(pz, qz, batch_size)
        return c + self.lmbda * penalty

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

    def log_hist(self, step):
        for w in self.encoder.trainable_weights:
            tf.summary.histogram("Encoder/{}".format(w.name), w, step=step)
        for w in self.decoder.trainable_weights:
            tf.summary.histogram("Decoder/{}".format(w.name), w, step=step)

    def display_random_bar(self, epoch):
        """Logs a decoded random sample into Tensorboard.
        """
        p_z = self.sample_pz(tf.constant(8))
        x_hat = self.decoder(p_z, training=False)
        img = tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=1)
        plt.clf()
        plt.figure(figsize=(16,9))
        plt.title("Epoch {}".format(epoch))
        plt.imsave(self.logdir+"/img/random/{:03d}.png".format(epoch), tf.squeeze(img, axis=-1))
        plt.close()
        tf.summary.image("Random", tf.expand_dims(img, axis=0), step=epoch)
      
    def display_reconstruction(self, epoch):
        """Logs a few reconstructions into Tensorboard.
        """
        for batch in self.test_dataset.take(1):
            x_hat = self.decoder(self.encoder(batch, training=False), training=False)
            reals = tf.concat([batch[i,:,:,:] for i in range(8)], axis=1)
            fakes = tf.concat([x_hat[i,:,:,:] for i in range(8)], axis=1)
            img = tf.concat([reals, fakes], axis=0)
            plt.clf()
            plt.figure(figsize=(16,9))
            plt.title("Epoch {}".format(epoch))
            plt.imsave(self.logdir+"/img/recons/{:03d}.png".format(epoch), tf.squeeze(img, axis=-1))
            plt.close()
            tf.summary.image("Reconstruction", tf.expand_dims(img, axis=0), step=epoch)

    def save_weights(self):
        self.encoder.save_weights(self.logdir+"/models/encoder/encoder")
        self.decoder.save_weights(self.logdir+"/models/decoder/decoder")

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
            #c = tf.reduce_mean(tf.keras.losses.MSE(y_true=batch, y_pred=x_hat))
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
        return (
            self.avg_enc_dec_train_loss.result()
        )

    def train(self):
        with self.writer.as_default():
            #with tf.summary.record_if(self.log_train):
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
