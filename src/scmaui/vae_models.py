import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from scmaui.layers import CUSTOM_OBJECTS
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.model = encoder

    def call(self, inputs):
        return self.model(inputs)

    def predict_step(self, ds):
        (inputs), (conditional, adversarial) = ds
        idata, imask = inputs
        idata = tuple([tf.where(tf.math.is_nan(I), tf.zeros_like(I), I) for I in idata])

        z = self.model([idata, imask, conditional], training=True)
        return z


class VAE(tf.keras.Model):
    """
    z, batch_pred = encoder([data, batch])
    recon = decoder([z, data, batch])
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_mean = Encoder(
            tf.keras.Model(
                [inp for inp in self.encoder.inputs if "advinput_" not in inp.name],
                self.encoder.get_layer("z_mean").output,
            )
        )

        self.decoder = decoder

        self.encoder_params = tf.keras.Model(
            [inp for inp in self.encoder.inputs if "advinput_" not in inp.name],
            self.encoder.get_layer("random_latent").output,
        ).trainable_weights
        self.batch_params = [
            w for w in encoder.trainable_weights if "advnet_" in w.name
        ]
        Ntotal = sum([np.prod(w.shape) for w in encoder.trainable_weights])
        Nencoder = sum(
            [np.prod(w.shape) for w in self.encoder_params.trainable_weights]
        )
        Nadv = sum([np.prod(w.shape) for w in self.batch_params])
        Ndecoder = sum([np.prod(w.shape) for w in self.decoder.trainable_weights])

        # debugging prints
        # print(f'Num total weights: {Ntotal}')
        # print(f'Num encoder weights: {Nencoder}')
        # print(f'Num encoder weights: {Nadv}')
        # print(f'Num decoder weights: {Ndecoder}')

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split(".h5")[0]
        s = ".h5"
        self.encoder.save(f + "_encoder_" + s)
        self.decoder.save(f + "_decoder_" + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder):
        encoder = _create_encoder(params)
        decoder = _create_decoder(params)

        return cls(encoder, decoder)

    @classmethod
    def load(cls, filename):
        f = filename.split(".h5")[0]
        s = ".h5"

        custom_objects = CUSTOM_OBJECTS
        encoder = load_model(f + "_encoder_" + s, custom_objects=custom_objects)
        decoder = load_model(f + "_decoder_" + s, custom_objects=custom_objects)
        return cls(encoder, decoder)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split(".h5")[0]
        s = ".h5"

        self.encoder.save_weights(f + "_encoder_" + s)
        self.decoder.save_weights(f + "_decoder_" + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split(".h5")[0]
        s = ".h5"

        self.encoder.load_weights(f + "_encoder_" + s)
        self.decoder.load_weights(f + "_decoder_" + s)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def predict_step(self, data):
        (inputs, outputs), (conditional, adversarial) = data
        idata, imask = inputs
        odata, omask = outputs
        idata = tuple([tf.where(tf.math.is_nan(I), tf.zeros_like(I), I) for I in idata])

        intercept = tf.ones_like(imask[0])

        ret = self.encoder([idata, imask, conditional, adversarial], training=False)
        if isinstance(ret, (list, tuple)):
            z, advloss = ret
        else:
            z = ret

        pred = self.decoder(
            [z, odata, omask, intercept, conditional, adversarial], training=False
        )
        return pred

    def train_step(self, data):
        losses = dict()
        (inputs, outputs), (conditional, adversarial) = data
        idata, imask = inputs
        odata, omask = outputs
        idata = tuple([tf.where(tf.math.is_nan(I), tf.zeros_like(I), I) for I in idata])

        intercept = tf.ones_like(imask[0])

        with tf.GradientTape(persistent=True) as tape:
            ret = self.encoder([idata, imask, conditional, adversarial], training=True)
            if isinstance(ret, (list, tuple)):
                z, advloss = ret
                losses["adv"] = advloss
            else:
                z = ret

            losses["kl"] = sum(self.encoder.losses)

            pred = self.decoder(
                [z, odata, omask, intercept, conditional, adversarial], training=True
            )

            # Decoder returns a loss value for each modality
            losses["recon"] = sum(self.decoder.losses)

            losses["loss"] = losses["kl"] + losses["recon"]

            if "adv" in losses:
                losses["loss"] -= losses["adv"]

        tf.debugging.check_numerics(losses["loss"], "check total loss NaN")
        grads = tape.gradient(
            losses["loss"], self.encoder_params + self.decoder.trainable_weights
        )
        [tf.debugging.check_numerics(g, "grad recon is nan") for g in grads]
        self.optimizer.apply_gradients(
            zip(grads, self.encoder_params + self.decoder.trainable_weights)
        )

        if "adv" in losses:
            grads = tape.gradient(losses["adv"], self.batch_params)
            self.optimizer.apply_gradients(zip(grads, self.batch_params))

        del tape
        return losses

    def test_step(self, data):
        losses = dict()
        (inputs, outputs), (conditional, adversarial) = data
        idata, imask = inputs
        odata, omask = outputs
        idata = tuple([tf.where(tf.math.is_nan(I), tf.zeros_like(I), I) for I in idata])

        intercept = tf.ones_like(imask[0])

        ret = self.encoder([idata, imask, conditional, adversarial], training=False)
        if isinstance(ret, (list, tuple)):
            z, advloss = ret
            losses["adv"] = advloss
        else:
            z = ret

        losses["kl"] = sum(self.encoder.losses)

        pred = self.decoder(
            [z, odata, omask, intercept, conditional, adversarial], training=False
        )

        losses["recon"] = sum(self.decoder.losses)

        losses["loss"] = losses["kl"] + losses["recon"]

        if "adv" in losses:
            losses["loss"] -= losses["adv"]
        return losses
