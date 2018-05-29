"""
Attempted implementation of Semi-Supervised GAN
https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py

Semi-Supervised Learning with Generative Adversarial Networks
Augustus Odena
https://arxiv.org/pdf/1606.01583.pdf
"""

import numpy as np
import lib
import keras
from keras import Model
from keras.layers import Input, Dense, Flatten, Reshape, GRU, Conv2D, Dropout, BatchNormalization
import keras.backend as K
from keras.utils import to_categorical

# TODO Callbacks
# Generate 10 characters, pass to discriminator?


class SGAN():
    def __init__(self, rnn=True):
        self.latent_dim = 100
        self.output_len = lib.char_size  # However many characters
        self.gen_length = 10

        self.discriminator = self.build_discriminator()

        if rnn:
            self.generator = self.build_generator_rnn()
        else:
            self.generator = self.build_generator_dense()
        optimizer = keras.optimizers.Adam()

        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[0.5, 0.5],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        noise = Input(shape=(self.latent_dim, ))
        generated = self.generator(noise)
        self.discriminator.trainable = False

        # Determining validity
        valid, _ = self.discriminator(generated)

        # Combined model
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator_rnn(self):
        # Untested
        seq_input = Input(shape=(self.latent_dim, ), name='rgen_input')
        x = Dense(units=256, activation='elu', name='rgen_predense')(seq_input)
        x = GRU(units=256, dropout=0.2, recurrent_dropout=0.2, name='rgen_gru')(x)
        x = Dense(units=256, activation='elu', name='rgen_postdense')(x)
        out = Dense(units=self.output_len, activation='softmax', name='rgen_out')(x)
        return Model(seq_input, out)

    def build_generator_dense(self):
        seq_input = Input(shape=(self.latent_dim, ), name='dgen_input')
        x = Dense(units=256, activation='elu', name='dgen_d1')(seq_input)
        x = BatchNormalization(momentum=0.8, name='dgen_norm1')(x)
        x = Dense(units=512, activation='elu', name='dgen_d2')(x)
        x = BatchNormalization(momentum=0.8, name='dgen_norm2')(x)
        x = Dense(units=1024, activation='elu', name='dgen_d3')(x)
        x = BatchNormalization(momentum=0.8, name='dgen_norm3')(x)
        out = Dense(units=self.output_len, activation='tanh', name='dgen_out')(x)

        return Model(seq_input, out)

    def build_discriminator(self):
        seq_input = Input(shape=(self.gen_length, ))
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='elu', name='conv1')(seq_input)
        x = BatchNormalization(momentum=0.8, name='norm1')(Dropout(rate=0.2, name='drop1')(x))
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='elu', name='conv2')(x)
        x = BatchNormalization(momentum=0.8, name='norm2')(Dropout(rate=0.2, name='drop2')(x))
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='elu', name='conv3')(x)
        x = Dropout(rate=0.2, name='drop3')(x)
        x = Flatten(name='flat')(x)

        valid = Dense(units=1, activation='sigmoid', name='valid_out')(x)
        label = Dense(self.output_len + 1, activation='softmax', name='label_out')(x)

        return Model(seq_input, [valid, label])

    def train(self, epochs, batch_size, sample_interval, corpus):
        # Ground truth
        valid = np.ones((batch_size, 1))
        invalid = np.zeros((batch_size, 1))

        for epoch in epochs:
            # region Discriminator Training
            chunk = self.random_chunk(text=corpus, chunksize=50)
            real = chunk[-self.gen_length:]
            # TODO need to convert chunks into numbers via get input target probably
            # Generate
            # noise = np.random.normal(0, 1, (batch_size, self.output_len))
            gen_input = chunk[:self.gen_length]
            fake = self.generator.predict(gen_input)

            d_loss_real = self.discriminator.train_on_batch(real, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake, invalid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # endregion

            # region Generator Training
            # Train to get discriminator to label as valid
            # noise = np.random.normal(0, 1, (batch_size, self.output_len))
            g_loss = self.combined.train_on_batch(gen_input, valid)
            # endregion

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # Generate samples
            if epoch % sample_interval == 0:
                self.sample_text(epoch)

    def sample_text(self, epoch):
        pass

    def random_chunk(self, text, chunksize=50):
        start = np.random.randint(0, len(text) - chunksize)
        return text[start:start + chunksize + 1]


if __name__ == '__main__':
    model = SGAN()
    model.train()
