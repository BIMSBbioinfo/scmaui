import os
import tensorflow as tf
from scmaui.layers import Sampling
from scmaui.layers import KLlossLayer
from scmaui.layers import ClipLayer
from scmaui.layers import MSEEndpoint
from scmaui.layers import MultinomialEndpoint
from scmaui.layers import BinomialEndpoint
from scmaui.layers import NegativeBinomialEndpoint
from scmaui.layers import ZeroInflatedNegativeBinomialEndpoint
from scmaui.layers import NegativeMultinomialEndpoint
from scmaui.layers import NegativeMultinomialEndpointV2
from scmaui.layers import BatchLoss
from scmaui.layers import ExpandDims
from scmaui.layers import MutInfoLayer
from keras.models import load_model

CUSTOM_OBJECTS = {'Sampling': Sampling,
                  'KLlossLayer': KLlossLayer,
                  'ClipLayer': ClipLayer,
                  'MSEEndpoint': MSEEndpoint,
                  'MultinomialEndpoint': MultinomialEndpoint,
                  'BinomialEndpoint': BinomialEndpoint,
                  'NegativeBinomialEndpoint': NegativeBinomialEndpoint,
                  'ZeroInflatedNegativeBinomialEndpoint': ZeroInflatedNegativeBinomialEndpoint,
                  'NegativeMultinomialEndpoint': NegativeMultinomialEndpoint,
                  'NegativeMultinomialEndpointV2': NegativeMultinomialEndpointV2,
                  'MutInfoLayer': MutInfoLayer,
                 }

class Encoder(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.model = encoder

    def predict_step(self, ds):
        data, batch = ds

        idata, imask = data
        z = self.model([idata, imask, batch], training=False)
        return z


class VAE(tf.keras.Model):
    """
    z = encoder(data)
    recon = decoder([z, data])
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = Encoder(tf.keras.Model(self.encoder.inputs,
                                              self.encoder.get_layer('z_mean').output))
        self.decoder = decoder

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'
        self.encoder.save(f + '_encoder_' + s)
        self.decoder.save(f + '_decoder_' + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder):
         encoder = _create_encoder(params)
         decoder = _create_decoder(params)

         return cls(encoder, decoder)

    @classmethod
    def load(cls, filename):
        f = filename.split('.h5')[0]
        s='.h5'

        custom_objects = CUSTOM_OBJECTS
        encoder = load_model(f + '_encoder_' + s, custom_objects=custom_objects)
        decoder = load_model(f + '_decoder_' + s, custom_objects=custom_objects)
        return cls(encoder, decoder)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.save_weights(f + '_encoder_' + s)
        self.decoder.save_weights(f + '_decoder_' + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.load_weights(f + '_encoder_' + s)
        self.decoder.load_weights(f + '_decoder_' + s)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, input):
        losses = dict()
        data, batch = input
        inputs, outputs = data
        idata, imask = inputs
        odata, omask = outputs
        
        with tf.GradientTape() as tape:
            z = self.encoder([idata, imask, batch], training=True)
            #z = self.encoder(inputs, training=True)
            for i, loss in enumerate(self.encoder.losses):
                losses[f'kl_loss_{i}'] = loss
            pred = self.decoder([z, odata, omask, batch], training=True)
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        losses['loss'] = total_loss
        return losses

    def test_step(self, input):
        losses = dict()
        data, batch = input
        inputs, outputs = data
        idata, imask = inputs
        odata, omask = outputs
        z = self.encoder([idata, imask, batch], training=False)
        pred = self.decoder([z, odata, omask, batch], training=False)

        total_loss = sum(self.encoder.losses) + sum(self.decoder.losses)

        losses['loss'] = total_loss
        return losses

    def predict_step(self, input):
        data, batch = input
        inputs, outputs = data
        idata, imask = inputs
        odata, omask = outputs
        z = self.encoder([idata, imask, batch], training=False)
        pred = self.decoder([z, odata, omask, batch], training=False)
        return pred

    def impute(self, ds):
        return self.predict(ds)

    def encode(self, ds):
        return self.encoder_predict.predict(ds)

class BAVARIA(tf.keras.Model):
    """
    z, batch_pred = encoder([data, batch])
    recon = decoder([z, data, batch])
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(BAVARIA, self).__init__(**kwargs)
        self.encoder = encoder
        self.encoder_predict = Encoder(tf.keras.Model([inp for inp  in self.encoder.inputs if 'batch_input' not in inp.name],
                                           self.encoder.get_layer('z_mean').output))

        self.decoder = decoder

        self.encoder_params = tf.keras.Model([inp for inp  in self.encoder.inputs if 'batch_input' not in inp.name],
                                          self.encoder.get_layer('random_latent').output).trainable_weights
        self.batch_params = [w for w in encoder.trainable_weights if 'batchcorrect' in w.name]

        #ba = [l.output for l in self.encoder.layers if 'combine_batches' in l.name]
        #test_encoder = tf.keras.Model(self.encoder.inputs,
        #                           [self.encoder.get_layer('random_latent').output, ba], name="encoder")
        #self.test_encoder = test_encoder

    def save(self, filename):
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'
        self.encoder.save(f + '_encoder_' + s)
        self.decoder.save(f + '_decoder_' + s)

    @classmethod
    def create(cls, params, _create_encoder, _create_decoder):
         encoder = _create_encoder(params)
         decoder = _create_decoder(params)

         return cls(encoder, decoder)

    @classmethod
    def load(cls, filename):
        f = filename.split('.h5')[0]
        s='.h5'

        custom_objects = CUSTOM_OBJECTS
        encoder = load_model(f + '_encoder_' + s, custom_objects=custom_objects)
        decoder = load_model(f + '_decoder_' + s, custom_objects=custom_objects)
        return cls(encoder, decoder)

    def save_weights(self, filename, overwrite=True, save_format=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.save_weights(f + '_encoder_' + s)
        self.decoder.save_weights(f + '_decoder_' + s)

    def load_weights(self, filename, by_name=False, skip_mismatch=False):
        f = filename.split('.h5')[0]
        s='.h5'

        self.encoder.load_weights(f + '_encoder_' + s)
        self.decoder.load_weights(f + '_decoder_' + s)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def train_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, batch = data

        with tf.GradientTape(persistent=True) as tape:
            z, b = self.encoder([profile, batch])
            kl_loss, batch_loss = self.encoder.losses
            losses['kl_loss'] = kl_loss
            losses['bloss'] = batch_loss

            pred = self.decoder([z, profile, batch])
            for i, loss in enumerate(self.decoder.losses):
                losses[f'recon_loss_{i}'] = loss

            recon_loss = sum(self.decoder.losses)
            total_loss = kl_loss + recon_loss - batch_loss

        grads = tape.gradient(total_loss, self.encoder_params + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder_params + self.decoder.trainable_weights))

        grads = tape.gradient(batch_loss, self.batch_params)
        self.optimizer.apply_gradients(zip(grads, self.batch_params))

        del tape
        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


    def test_step(self, data):
        losses = dict()
        if isinstance(data, tuple):
            profile, batch = data
        z, b = self.encoder([profile, batch])

        kl_loss, batch_loss = self.encoder.losses
        pred = self.decoder([z, profile, batch])
        recon_loss = sum(self.decoder.losses)

        total_loss = kl_loss + recon_loss - batch_loss

        losses['loss'] = total_loss
        for mn, s in zip(self.encoder.metrics_names, self.encoder.metrics):
            losses[mn] = s.result()
        return losses


