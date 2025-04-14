import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras.optimizers import Adam


class VAEModel(Model):
    def __init__(self, encoder, decoder, reconstruction_loss_weight=1000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, inputs):
        mu, log_variance, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mu, log_variance, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3])
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_variance - tf.square(mu) - tf.exp(log_variance), axis=1)
            )
            total_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



class VAE:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        self.model = VAEModel(self.encoder, self.decoder, self.reconstruction_loss_weight)

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding="same",
                name=f"encoder_conv_layer_{i+1}"
            )(x)
            x = ReLU()(x)
            x = BatchNormalization()(x)

        self._shape_before_bottleneck = tf.keras.backend.int_shape(x)[1:]
        x = Flatten()(x)
        mu = Dense(self.latent_space_dim, name="mu")(x)
        log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sampling(args):
            mu, log_variance = args
            epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu))
            return mu + tf.exp(log_variance / 2) * epsilon

        z = Lambda(sampling, name="encoder_output")([mu, log_variance])
        self.encoder = Model(encoder_input, [mu, log_variance, z], name="encoder")
        self._model_input = encoder_input

    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_space_dim,), name="decoder_input")
        dense_neurons = np.prod(self._shape_before_bottleneck)
        x = Dense(dense_neurons)(decoder_input)
        x = Reshape(self._shape_before_bottleneck)(x)

        for i in reversed(range(1, self._num_conv_layers)):
            x = Conv2DTranspose(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding="same",
                name=f"decoder_conv_transpose_layer_{i}"
            )(x)
            x = ReLU()(x)
            x = BatchNormalization()(x)

        x = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name="decoder_output"
        )(x)
        decoder_output = Activation("sigmoid")(x)

        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)

    def train(self, x_train, batch_size, epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def reconstruct(self, images):
        _, _, z = self.encoder.predict(images)
        return self.decoder.predict(z)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        params = [self.input_shape, self.conv_filters, self.conv_kernels, self.conv_strides, self.latent_space_dim]
        with open(os.path.join(save_folder, "parameters.pkl"), "wb") as f:
            pickle.dump(params, f)

    def _save_weights(self, save_folder):
    # Make sure the model is built by running one forward pass with dummy data
        if not self.model.built:
            dummy_input = tf.zeros((1,) + self.input_shape)
            self.model(dummy_input)

        self.model.save_weights(os.path.join(save_folder, "weights.weights.h5"))


    @classmethod
    def load(cls, save_folder="."):
        with open(os.path.join(save_folder, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)
        autoencoder = cls(*params)

        # 🔧 Add a dummy forward pass to build the model
        dummy_input = tf.zeros((1,) + autoencoder.input_shape)
        autoencoder.model(dummy_input)  # This builds the model

        # ✅ Now load weights safely
        autoencoder.load_weights(os.path.join(save_folder, "weights.weights.h5"))
        return autoencoder


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.compile()
    autoencoder.summary()
