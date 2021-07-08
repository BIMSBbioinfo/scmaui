import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

@tf.function
def gaussian_likelihood(targets, mask, mu, sigma):
    #return - mask * (.5*tf.math.square((targets - mu)/sigma) + tf.math.log(sigma))
    return - mask * tf.math.square(targets - mu)

@tf.function
def gaussian_mean(mu, sigma):
    return mu

@tf.function
def gamma_likelihood(targets, mask, alpha, beta):
    likeli = alpha * tf.math.log(beta)
    likeli -= tf.math.lgamma(alpha)
    likeli += (alpha - 1) * targets
    likeli -= beta * targets
    return mask * likeli 

@tf.function
def gamma_mean(alpha, beta):
    return alpha/beta

@tf.function
def mixture_predict(pi, mu):
    return tf.reduce_sum(_softmax(pi) * mu, axis=-1)


@tf.function
def log_softmax(x):
    #x = x - xmax
    sp = x - tf.reduce_logsumexp(x, axis=-1, keepdims=True)
    return sp

@tf.function
def _softmax(x):
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - xmax
    sp = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
    tf.debugging.check_numerics(sp, "_softmax is NaN")
    return sp

@tf.function
def softmax1p_mask(x, mask):
    x = x + tf.math.log(mask)
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - xmax
    sp = tf.exp(x) / (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p_mask is NaN")
    return sp

@tf.function
def softmax1p0_mask(x, mask):
    x = x + tf.math.log(mask)
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    sp = tf.exp(-xmax)/ (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p0_mask is NaN")
    return sp


@tf.function
def softmax1p(x):
    xmax = tf.reduce_max(x, axis=-1, keepdims=True)
    x = x - xmax
    sp = tf.exp(x) / (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p is NaN")
    return sp


@tf.function
def softmax1p0(x):
    xmax = tf.reduce_max(x, axis=-1)
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    sp = tf.exp(-xmax)/ (tf.exp(-xmax)+ tf.reduce_sum(tf.exp(x), axis=-1))
    sp = tf.expand_dims(sp, axis=-1)
    tf.debugging.check_numerics(sp, "softmax1p0 is NaN")
    return sp


@tf.function
def multinomial_likelihood(targets, mask, logits):
    logits += tf.math.log(mask)
    log = logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
    return tf.where(tf.math.is_inf(log), tf.zeros_like(log), targets * log)

@tf.function
def binomial_likelihood(targets, mask, logits, N):
    log = targets * tf.math.log_sigmoid(logits) + (N-targets) * tf.math.log_sigmoid(-logits)
    return mask * log

@tf.function
def negative_binomial_likelihood(targets, mask, logits, r):
    likeli = tf.math.lgamma(targets + r) - tf.math.lgamma(r) - tf.math.lgamma(targets + 1.)
    likeli += targets * tf.math.log_sigmoid(logits)
    likeli += r * tf.math.log_sigmoid(-logits)
    return mask * likeli

@tf.function
def zero_inflated_negative_binomial_likelihood(targets, mask, logits, r, pi):
    # negative binom
    likeli = tf.math.lgamma(targets + r) - tf.math.lgamma(r) - tf.math.lgamma(targets + 1.)
    likeli += targets * tf.math.log_sigmoid(logits)
    likeli += r * tf.math.log_sigmoid(-logits)
    #zi
    likeli0 = tf.where(targets>0, tf.math.log(tf.zeros_like(targets)), targets)
    likeli = tf.experimental.numpy.logaddexp(tf.math.log_sigmoid(pi) + likeli0, 
                                             tf.math.log_sigmoid(-pi) + likeli)
    return mask * likeli

@tf.function
def mixture_likelihood(pi, likelihood_, *args):
    return tf.math.logsumexp(log_softmax(pi) + likehood(*args), axis=-1)

@tf.function
def negative_multinomial_likelihood(targets, mask, logits, r):
    likeli = tf.reduce_sum(tf.math.xlogy(targets, softmax1p_mask(logits, mask)+1e-10), axis=-1)
    tf.debugging.check_numerics(likeli, "targets * log(p)")
    likeli += tf.math.xlogy(r, softmax1p0_mask(logits, mask) + 1e-10)
    tf.debugging.check_numerics(likeli, "r * log(1-p)")
    likeli += tf.math.lgamma(r + tf.reduce_sum(targets*mask, axis=-1))
    tf.debugging.check_numerics(likeli, "lgamma(r + x)")
    likeli -= tf.math.lgamma(r)
    tf.debugging.check_numerics(likeli, "lgamma(r)")
    return likeli


@tf.function
def negative_multinomial_likelihood_v2(targets, mask, mul_logits, p0_logits, r):
    X = tf.reduce_sum(targets*mask, axis=-1)
    #nb likelihood
    likeli = tf.math.lgamma(r + X)
    likeli -= tf.math.lgamma(r)
    likeli -= tf.math.lgamma(X + 1.)

    likeli += r * tf.math.log_sigmoid(p0_logits)
    likeli += X * tf.math.log_sigmoid(-p0_logits)

    #likeli += tf.reduce_sum(tf.math.xlogy(r, tf.math.sigmoid(p0_logits)+1e-10), axis=-1)

    # mul likelihood
    logp = log_softmax(mul_logits + tf.math.log(mask))
    logp = tf.where(tf.math.is_inf(logp), tf.zeros_like(logp), logp)
    likeli += tf.reduce_sum(targets * logp, axis=-1)
    tf.debugging.check_numerics(likeli, "negative_multinomial_likelihood_v2")
    return likeli


class ExpandDims(layers.Layer):
    def __init__(self, axis=1, *args, **kwargs):
        super(ExpandDims, self).__init__(*args, **kwargs)
        self.axis = axis
    def call(self, inputs):
        o = tf.expand_dims(inputs, axis=self.axis)
        return tf.expand_dims(inputs, axis=self.axis)
    def get_config(self):
        config = {'axis':self.axis}
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ClipLayer(layers.Layer):
    def __init__(self, min_value, max_value, *args, **kwargs):
        super(ClipLayer, self).__init__(*args, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    def call(self, inputs):
        return tf.clip_by_value(inputs, clip_value_min=self.min_value,
                                clip_value_max=self.max_value)
    def get_config(self):
        config = {'min_value':self.min_value,
                  'max_value': self.max_value}
        base_config = super(ClipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        #z_mean = tf.expand_dims(z_mean, axis=1)
        #z_log_var = tf.expand_dims(z_log_var, axis=1)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[-1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLlossLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return z_mean, z_log_var

class BatchLoss(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BatchLoss, self).__init__(*args, **kwargs)
        self.catmet = tf.keras.metrics.CategoricalAccuracy(name='bacc')
        self.binmet = tf.keras.metrics.AUC(name='bauc')

    def call(self, inputs):
        if len(inputs) < 2:
            return inputs
        pred_batch, true_batch = inputs
        tf.debugging.assert_non_negative(pred_batch)
        tf.debugging.assert_non_negative(true_batch)
        loss = 0.0
        for tb, pb in zip(true_batch, pred_batch):
            loss += tf.reduce_sum(-tf.math.xlogy(tb, pb+1e-9))
            self.add_metric(self.catmet(tb,pb))
            self.add_metric(self.binmet(tb[:,0,0],pb[:,-1,0]))
        tf.debugging.check_numerics(loss, "targets * log(p)")

        self.add_loss(loss)
        return pred_batch

    def compute_output_shape(self, input_shape):
        return input_shape[0]

#class AverageChannel(layers.Layer):
#    def call(self, inputs):
#        return tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
#    def compute_output_shape(self, input_shape):
#        return input_shape[:-1] + (1,)


#class ScalarBiasLayer(layers.Layer):
#    def build(self, input_shape):
#        self.bias = self.add_weight('bias',
#                                    shape=(1,),
#                                    initializer='ones',
#                                    trainable=True)
#    def call(self, x):
#        return tf.ones((tf.shape(x)[0],) + (1,)*(len(x.shape.as_list())-1))*self.bias


class MutInfoLayer(layers.Layer):
    def __init__(self, start_delay=100, *args, **kwargs):
        self.start_delay = start_delay
        super(MutInfoLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = {'start_delay':self.start_delay}
        base_config = super(MutInfoLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        #input_shape = tensor_shape.TensorShape(input_shape)

        self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=[1 for _ in input_shape[:-1]] + [input_shape[-1]],
          dtype='float32',
          trainable=False,
          initializer=initializers.get('zeros'),)

        self.delay_count = self.add_weight(
          name='delay',
          shape=(),
          dtype='float32',
          trainable=False,
          initializer=initializers.get('zeros'),)

        self.delay_count.assign_add(-self.start_delay)

        self.built=True

    def call(self, x, training=None):
        input_shape = tf.shape(x)

        def _operation():
            # compute covariance
            #x_zero = x - self.moving_mean
            x_zero = x - self.moving_mean
            x_zero_0 = tf.expand_dims(x_zero, -1)
            x_zero_1 = tf.expand_dims(x_zero, -2)
            cov = tf.reduce_mean(x_zero_0*x_zero_1, axis=[i for i in range(len(input_shape)-1)])
            cov = cov + tf.eye(cov.shape[0])

            tf.debugging.assert_positive(tf.linalg.det(cov))

            #ml_loss = -0.5 * (tf.linalg.logdet(cov) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cov))))
            ml_loss = -0.5 * (tf.linalg.logdet(cov) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cov))))
            return ml_loss

        # update feature means
        if training is None or training:
            self.delay_count.assign_add(1)
            x_mean = tf.math.reduce_mean(x, axis=[i for i in range(len(input_shape)-1)], keepdims=True)
            self.moving_mean.assign(.3*self.moving_mean + .7 *x_mean)

        ml_loss = tf.case([(tf.less(self.delay_count, 1), lambda: tf.constant(0.0))], default=_operation)

        self.add_loss(ml_loss)

        return x


class MSEEndpoint(layers.Layer):
    def call(self, inputs):
        targets = None
        targets, masks, mu, sigma = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(gaussian_likelihood(targets, masks, mu, sigma), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "MSEEndpoint NaN")

        return mu

class GammaEndpoint(layers.Layer):
    def call(self, inputs):
        targets = None
        targets, masks, alpha, gamma = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(gamma_likelihood(targets, alpha, gamma), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "MSEEndpoint NaN")

        return alpha/gamma

class MultinomialEndpoint(layers.Layer):
    def call(self, inputs):
        targets, masks, logits = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(multinomial_likelihood(targets, masks, logits), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "MultinomialEndpoint NaN")

        return _softmax(logits)

class BinomialEndpoint(layers.Layer):
    def __init__(self, N, *args, **kwargs):
        self.N = N
        super(BinomialEndpoint, self).__init__(*args, **kwargs)
 
    def get_config(self):
        config = {'N':self.N,}
        base_config = super(BinomialEndpoint, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        targets = None
        targets, masks, logits = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(binomial_likelihood(targets, masks, logits, self.N), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "BinomialEndpoint NaN")

        return tf.math.sigmoid(logits)

class ZeroInflatedNegativeBinomialEndpoint(layers.Layer):
 
    def call(self, inputs):
        targets = None
        targets, masks, logits, r, pi = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(zero_inflated_negative_binomial_likelihood(targets, masks, logits, r, pi), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeBinomialEndpoint NaN")

        return r * tf.math.sigmoid(logits) / tf.math.sigmoid(-logits)

class NegativeBinomialEndpoint(layers.Layer):
 
    def call(self, inputs):
        targets = None
        targets, masks, logits, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           tf.reduce_sum(negative_binomial_likelihood(targets, masks, logits, r), axis=-1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeBinomialEndpoint NaN")

        return r * tf.math.sigmoid(logits) / tf.math.sigmoid(-logits)

class NegativeMultinomialEndpoint(layers.Layer):
    def call(self, inputs):
        targets = None
        targets, mask, logits, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           negative_multinomial_likelihood(targets, mask, logits, r)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeMultinomialEndpoint NaN")

        p = softmax1p_mask(logits, mask)
        p0 = softmax1p0_mask(logits, mask)
        return p * r / (p0 + 1e-10)


class NegativeMultinomialEndpointV2(layers.Layer):
    def call(self, inputs):
        targets = None
        targets, mask, mul_logits, p0_logit, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                           negative_multinomial_likelihood_v2(targets, mask, mul_logits, p0_logit, r)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "NegativeMultinomialEndpoint NaN")

        p0 = tf.math.sigmoid(p0_logit)
        p1 = tf.math.sigmoid(-p0_logit)

        f = p1*r/p0
        pm = _softmax(mul_logits)
        return pm *f

        #return p * r / (p0 + 1e-10)

class MixtureModelEndpoint(layers.Layer):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        if model == 'gaussian':
             self.likelihood = gaussian_likelihood
             self.mean = gaussian_mean
        elif model == 'gamma':
             self.likelihood = gamma_likelihood
             self.mean = gamma_mean
        else:
             raise ValueError(f'unknown model: {model}')
        super(MixtureModelEndpoint, self).__init__(*args, **kwargs)
 
    def get_config(self):
        config = {'model':self.model,}
        base_config = super(MixtureModelEndpoint, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        args = inputs

        targets, mask = inputs[:2]
        params = inputs[2:-1]
        pi = inputs[-1]
        targets = tf.expand_dims(targets, -1)
        mask = tf.expand_dims(mask, -1)
        pi = tf.expand_dims(pi, -2)
         
        if targets is not None:
            inp = (targets, mask,) + tuple(params)
            reconstruction_loss = -tf.reduce_mean(
                           tf.math.reduce_sum(mask * tf.math.reduce_logsumexp(self.likelihood(*inp) + tf.math.log_softmax(pi) - tf.math.log(1e5), axis=-1), -1)
                         )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss,
                                        "MixtureModel NaN")
        mu = self.mean(*params)
        pred = mixture_predict(pi, mu)
        return pred
