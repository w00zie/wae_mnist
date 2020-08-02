import tensorflow as tf

import os
import json
from datetime import datetime, timedelta
from time import time

from data import get_dataset
from utils import get_ae_disc


class Train:

    def __init__(self, config):

        self.h_dim = config["h_dim"]
        self.z_dim = config["z_dim"]
        self.epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.sigma_z = config["sigma_z"]
        self.lmbda = config["lambda"]
        
        # Experiment directory
        self.logdir = os.path.join("runs", \
            datetime.now().strftime("wae_gan_%d_%m_%Y-%H:%M:%S"))
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
        os.mkdir(os.path.join(self.logdir,"models","discriminator"))
        with open(os.path.join(self.logdir,"config.json"), "w") as f:
            json.dump(config, f)

        # Models ================================================================
        self.encoder, self.decoder, self.discriminator = get_ae_disc(config)

        # Optimizers ============================================================
        ae_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["ae_lr"],
            decay_steps=config["ae_dec_steps"],
            decay_rate=config["ae_dec_rate"])
        disc_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["d_lr"],
            decay_steps=config["d_dec_steps"],
            decay_rate=config["d_dec_rate"])

        self.enc_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.dec_optim = tf.keras.optimizers.Adam(ae_scheduler)
        self.disc_optim = tf.keras.optimizers.Adam(disc_scheduler)

        # Data ==================================================================
        tf.print("Loading data...")
        self.train_dataset, self.test_dataset = \
            get_dataset(batch_size=self.batch_size)
        tf.print("Done.")

        # Metric trackers =======================================================
        self.avg_d_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_d_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_d_z_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_mse_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_enc_dec_test_loss = tf.keras.metrics.Mean(dtype=tf.float32)

    # Prior =====================================================================
    def sample_pz(self, batch_size: tf.Tensor) -> tf.Tensor:
        return tf.random.normal(shape=(batch_size, self.z_dim),
                                stddev=tf.sqrt(self.sigma_z))

    # Losses ====================================================================
    #@tf.function
    def disc_loss(self, d_pz: tf.Tensor, d_qz: tf.Tensor) -> tf.Tensor:
        real_loss = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_pz), 
                                                        y_pred=d_pz)
        fake_loss = tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(d_qz), 
                                                        y_pred=d_qz)
        return self.lmbda*(tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss))

    #@tf.function
    def ae_loss(self, 
                batch: tf.Tensor,
                x_hat: tf.Tensor,
                d_qz: tf.Tensor) -> tf.Tensor:
        c = tf.reduce_sum(tf.square(batch - x_hat), axis=[1,2,3])
        penalty = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_qz), 
                                                      y_pred=d_qz)
        return tf.reduce_mean(c) + tf.reduce_mean(self.lmbda*penalty)

    # Optimization steps ========================================================
    @tf.function
    def train_discriminator(self, batch: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(batch)[0]
        with tf.GradientTape() as tape:
            pz = self.sample_pz(batch_size)
            d_pz = self.discriminator(pz, training=True)
            d_qz = self.discriminator(self.encoder(batch, training=True), 
                                     training=True)
            disc_loss = self.disc_loss(d_pz, d_qz)
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optim.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def train_enc_dec(self, batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            q_z = self.encoder(batch, training=True)
            x_hat = self.decoder(q_z, training=True)
            d_qz = self.discriminator(q_z, training=True)
            enc_dec_loss = self.ae_loss(batch, x_hat, d_qz)
        enc_grads = enc_tape.gradient(enc_dec_loss, self.encoder.trainable_variables)
        dec_grads = dec_tape.gradient(enc_dec_loss, self.decoder.trainable_variables)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        return enc_dec_loss

    # One-epoch train steps =====================================================
    #@tf.function
    def validation_step(self):
        # Reset metrics
        self.avg_enc_dec_test_loss.reset_states()
        self.avg_mse_test_loss.reset_states()
        self.avg_d_test_loss.reset_states()
        # Run through an epoch of validation
        for batch in self.test_dataset:
            z_hat = self.encoder(batch, training=False)
            x_hat = self.decoder(z_hat, training=False)
            d_z_hat = self.discriminator(z_hat, training=False)
            c = tf.reduce_mean(tf.keras.losses.MSE(y_true=batch, y_pred=x_hat), axis=[1,2])
            penalty = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(d_z_hat), 
                                                          y_pred=d_z_hat)
            # Log metrics for every batch
            self.avg_d_test_loss(tf.reduce_mean(d_z_hat))
            self.avg_mse_test_loss(tf.reduce_mean(c))
            self.avg_enc_dec_test_loss(tf.reduce_mean(c + self.lmbda*penalty))
        return (
            self.avg_d_test_loss.result(), 
            self.avg_enc_dec_test_loss.result(), 
            self.avg_mse_test_loss.result()
        )

    #@tf.function
    def train_step(self):
        # Reset metrics
        self.avg_d_train_loss.reset_states()
        self.avg_enc_dec_train_loss.reset_states()
        # Run through an epoch of training
        for batch in self.train_dataset:
            enc_dec_loss = self.train_enc_dec(batch)
            disc_loss = self.train_discriminator(batch)
            # Log metrics for every batch
            self.avg_d_train_loss(disc_loss)
            self.avg_enc_dec_train_loss(enc_dec_loss)
        return (
            self.avg_d_train_loss.result(), 
            self.avg_enc_dec_train_loss.result()
        )

    # Plot utils ================================================================
    def log_hist(self, step):
        for w in self.encoder.trainable_weights:
            tf.summary.histogram("Encoder/{}".format(w.name), w, step=step)
        for w in self.decoder.trainable_weights:
            tf.summary.histogram("Decoder/{}".format(w.name), w, step=step)
        for w in self.discriminator.trainable_weights:
            tf.summary.histogram("Discriminator/{}".format(w.name), w, step=step)

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
    def calc_discriminator_latent(self):
        self.avg_d_z_loss.reset_states()
        for _ in tf.range(100):
            z = self.sample_pz(self.batch_size)
            d_z = self.discriminator(z, training=False)
            self.avg_d_z_loss(d_z)
        return self.avg_d_z_loss.result()

    def save_weights(self):
        self.encoder.save_weights(os.path.join(self.logdir,"models","encoder","encoder"))
        self.decoder.save_weights(os.path.join(self.logdir,"models","decoder","decoder"))
        self.discriminator.save_weights(os.path.join(self.logdir, "models","discriminator","discriminator"))

    # Training procedure ========================================================
    def train(self):
        with self.writer.as_default():
            start = time()
            for epoch in range(self.epochs):
                # Train for one epoch
                d_loss, e_d_loss = self.train_step()
                # Display a few bars
                if epoch % 2 == 0:
                    self.display_random_bar(epoch)
                    self.display_reconstruction(epoch)
                if epoch % 10 == 0:
                    self.save_weights()
                # Validate on the test set
                d_g_z, test_loss, mse = self.validation_step()
                # Calculate D(z)
                d_z = self.calc_discriminator_latent()
                # Logging
                tf.summary.scalar("Train/Discriminator Loss", 
                                  d_loss, step=epoch)
                tf.summary.scalar("Train/Encoder-Decoder Loss", 
                                  e_d_loss, step=epoch)
                tf.summary.scalar("Learning Rates/Encoder", 
                                  self.enc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Decoder", 
                                  self.dec_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Learning Rates/Discriminator", 
                                  self.disc_optim._decayed_lr('float32').numpy(), 
                                  step=epoch)
                tf.summary.scalar("Test/D(G(x))", d_g_z, step=epoch)
                tf.summary.scalar("Test/D(z)", d_z, step=epoch)
                tf.summary.scalar("Test/Encoder-Decoder Loss", 
                                  test_loss, step=epoch)
                tf.summary.scalar("Test/MSE", mse, step=epoch)

                self.log_hist(epoch)
                
                tf.print("Epoch {}/{}  -  [d_loss = {}]  -  [ae_loss = {}]  -  {}".\
                    format(epoch, self.epochs-1, d_loss.numpy(), 
                           e_d_loss.numpy(), datetime.now().strftime('%H:%M:%S')))
       
        print(f"Training took {timedelta(seconds=time()-start)}")
        self.writer.flush()

if __name__ == "__main__":
    
    from wae_gan.config_gan import config_gan
    # Logging only (W)arnings and (E)rrors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    train = Train(config_gan)
    train.train()