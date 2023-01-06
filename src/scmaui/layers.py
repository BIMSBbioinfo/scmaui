import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

# helper functions
@tf.function
def gaussian_mean(mu, sigma):
    """return gaussian mean"""
    return mu


@tf.function
def gamma_mean(alpha, beta):
    """return mean of gamma distribution"""
    return alpha / beta


@tf.function
def mixture_mean(pi, mu):
    """compute mixture model mean"""
    return tf.reduce_sum(_softmax(pi) * mu, axis=-1)


@tf.function
def log_softmax(x):
    """compute (nan-aware) log-softmax"""
    x = tf.where(tf.math.is_nan(x), tf.math.log(tf.zeros_like(x)), x)
    sp = x - tf.reduce_logsumexp(x, axis=-1, keepdims=True)
    return sp


@tf.function
def _softmax(x):
    """compute softmax"""
    xmax = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    x = x - xmax
    sp = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
    tf.debugging.check_numerics(sp, "_softmax is NaN")
    return sp


@tf.function
def softmax1p_mask(x, mask):
    """compute exp(xi) / [1 + sum_j exp(xj)] missing-value aware"""
    # set activation of missing components to -inf (through log(0.0))
    # so the softmax effectively rangers over a reduced dimenionality
    x = x + tf.math.log(mask)
    xmax = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    x = x - xmax
    sp = tf.exp(x) / (tf.exp(-xmax) + tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p_mask is NaN")
    return sp


@tf.function
def softmax1p0_mask(x, mask):
    """compute 1 / [1 + sum_j exp(xj)] missing-value aware"""
    # set activation of missing components to -inf (through log(0.0))
    # so the softmax effectively rangers over a reduced dimenionality
    x = x + tf.math.log(mask)
    xmax = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    x = x - xmax
    sp = tf.exp(-xmax) / (
        tf.exp(-xmax) + tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
    )
    tf.debugging.check_numerics(sp, "softmax1p0_mask is NaN")
    return sp


@tf.function
def softmax1p(x):
    """compute exp(xi) / [1 + sum_j exp(xj)]"""
    xmax = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    x = x - xmax
    sp = tf.exp(x) / (tf.exp(-xmax) + tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True))
    tf.debugging.check_numerics(sp, "softmax1p is NaN")
    return sp


@tf.function
def softmax1p0(x):
    """compute 1 / [1 + sum_j exp(xj)]"""
    xmax = tf.reduce_max(x, axis=-1)
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    sp = tf.exp(-xmax) / (tf.exp(-xmax) + tf.reduce_sum(tf.exp(x), axis=-1))
    sp = tf.expand_dims(sp, axis=-1)
    tf.debugging.check_numerics(sp, "softmax1p0 is NaN")
    return sp


# log-likelihood functions


@tf.function
def gaussian_likelihood(targets, mu):
    """gaussian-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    loglikeli = -mask * tf.math.square(targets - mu)
    return loglikeli


@tf.function
def gamma_likelihood(targets, alpha, beta):
    """gamma-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    loglikeli = alpha * tf.math.log(beta)
    loglikeli -= tf.math.lgamma(alpha)
    loglikeli += (alpha - 1) * tf.math.log(targets + 1e-7)
    loglikeli -= beta * targets

    loglikeli *= mask
    return loglikeli


@tf.function
def multinomial_likelihood(targets, logits):
    """multinomial-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    logits += tf.math.log(mask)
    loglikeli = targets * (
        logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True)
    )
    return loglikeli


@tf.function
def poisson_likelihood(targets, logits):
    """poisson-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    loglikeli = mask * (targets * tf.math.log(logits) - logits)
    return loglikeli


@tf.function
def binomial_likelihood(targets, logits, N):
    """binomial-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    loglikeli = mask * (
        targets * tf.math.log_sigmoid(logits)
        + (N - targets) * tf.math.log_sigmoid(-logits)
    )
    return loglikeli


@tf.function
def negative_binomial_likelihood(targets, logits, r):
    """negative binomial-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    loglikeli = (
        tf.math.lgamma(targets + r) - tf.math.lgamma(r) - tf.math.lgamma(targets + 1.0)
    )
    loglikeli += targets * tf.math.log_sigmoid(logits)
    loglikeli += r * tf.math.log_sigmoid(-logits)

    loglikeli *= mask
    return loglikeli


@tf.function
def zero_inflated_negative_binomial_likelihood(targets, logits, r, pi):
    """zero-inflated negative binomial-likelhood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    # negative binom
    loglikeli = (
        tf.math.lgamma(targets + r) - tf.math.lgamma(r) - tf.math.lgamma(targets + 1.0)
    )
    loglikeli += targets * tf.math.log_sigmoid(logits)
    loglikeli += r * tf.math.log_sigmoid(-logits)
    # zi
    loglikeli0 = tf.where(targets > 0, tf.math.log(tf.zeros_like(targets)), targets)
    # loglikeli = tf.experimental.numpy.logaddexp(tf.math.log_sigmoid(pi) + loglikeli0,
    #                                         tf.math.log_sigmoid(-pi) + loglikeli)

    loglikeli = tf.math.log(
        tf.math.exp(tf.math.log_sigmoid(pi) + loglikeli0)
        + tf.math.exp(tf.math.log_sigmoid(-pi) + loglikeli)
    )

    loglikeli *= mask
    return loglikeli


@tf.function
def mixture_likelihood(pi, likelihood_, *args):
    """general likelihood for mixture models"""
    return tf.math.logsumexp(log_softmax(pi) + likehood(*args), axis=-1)


@tf.function
def negative_multinomial_likelihood(targets, logits, r):
    """negative multinomial-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    targets = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)

    # mask = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets))
    likeli = tf.reduce_sum(
        tf.math.xlogy(targets, softmax1p_mask(logits, mask) + 1e-10), axis=-1
    )
    tf.debugging.check_numerics(likeli, "targets * log(p)")
    likeli += tf.math.xlogy(r, softmax1p0_mask(logits, mask) + 1e-10)
    tf.debugging.check_numerics(likeli, "r * log(1-p)")
    likeli += tf.math.lgamma(r + tf.reduce_sum(targets, axis=-1))
    tf.debugging.check_numerics(likeli, "lgamma(r + x)")
    likeli -= tf.math.lgamma(r)
    tf.debugging.check_numerics(likeli, "lgamma(r)")

    return likeli


@tf.function
def negative_multinomial_likelihood_v2(targets, mul_logits, p0_logits, r):
    """negative multinomial-likelihood (using independent p0 parameters)"""
    T = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)
    X = tf.reduce_sum(T, axis=-1)
    # nb likelihood
    likeli = tf.math.lgamma(r + X)
    likeli -= tf.math.lgamma(r)
    likeli -= tf.math.lgamma(X + 1.0)

    likeli += r * tf.math.log_sigmoid(p0_logits)
    likeli += X * tf.math.log_sigmoid(-p0_logits)

    # mul likelihood
    logp = log_softmax(
        mul_logits
        + tf.where(tf.math.is_nan(targets), tf.math.log(tf.zeros_like(targets))),
        tf.zeros_like(targets),
    )

    likeli += tf.reduce_sum(
        tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets * logp),
        axis=-1,
    )
    tf.debugging.check_numerics(likeli, "negative_multinomial_likelihood_v2")
    return likeli


@tf.function
def dirichlet_multinomial_likelihood(targets, mul_logits):
    """dirichlet multinomial-likelihood"""
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )

    T = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)
    X = tf.reduce_sum(T, axis=-1)

    # correct in case there are missing values
    alpha = mul_logits
    alphasum = tf.reduce_sum(alpha * mask, axis=-1)

    # nb likelihood
    likeli0 = tf.math.lgamma(alphasum)
    likeli0 += tf.math.lgamma(X + 1)
    likeli0 -= tf.math.lgamma(X + alphasum)
    likeli1 = mask * tf.math.lgamma(T + alpha)
    likeli1 -= mask * tf.math.lgamma(alpha)
    likeli1 -= mask * tf.math.lgamma(T + 1.0)
    likeli1 = tf.reduce_sum(likeli1, axis=-1)

    likeli = likeli0 + likeli1
    tf.debugging.check_numerics(likeli, "dirichlet_multinomial_likelihood")
    return likeli


@tf.function
def negative_dirichlet_multinomial_likelihood(
    targets, mul_logits, alphasum, r0, alpha0
):
    mask = tf.where(
        tf.math.is_nan(targets), tf.zeros_like(targets), tf.ones_like(targets)
    )
    T = tf.where(tf.math.is_nan(targets), tf.zeros_like(targets), targets)
    X = tf.reduce_sum(T, axis=-1)

    # correct in case there are missing values
    alpha = _softmax(mul_logits) * alphasum
    alphasum_new = alpha * mask

    # nb likelihood
    likeli0 = _lbeta(X + r0, alphasum + alpha0)
    likeli0 -= _lbeta(r0, alpha0)

    likeli1 += mask * tf.math.lgamma(T + alpha)
    likeli1 -= mask * tf.math.lgamma(T + 1)
    likeli1 -= mask * tf.math.lgamma(alpha)

    likeli = likeli0 + tf.reduce_sum(likeli1, axis=-1)
    tf.debugging.check_numerics(likeli, "negative_dirichlet_multinomial_likelihood")
    return likeli


class ExpandDims(layers.Layer):
    """helper layer to expand the dimensions of a tensor"""

    def __init__(self, axis=1, *args, **kwargs):
        super(ExpandDims, self).__init__(*args, **kwargs)
        self.axis = axis

    def call(self, inputs):
        o = tf.expand_dims(inputs, axis=self.axis)
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JointSigma(layers.Layer):
    """Compute combined latent sigma across modalities"""

    def __init__(self, *args, **kwargs):
        super(JointSigma, self).__init__(*args, **kwargs)

    def get_config(self):
        return super(JointSigma, self).get_config()

    def compute_output_shape(self, input_sigma, input_masks):
        return input_sigma[0]

    def call(self, sigmas, masks):
        if not isinstance(sigmas, (list, tuple)):
            sigmas = [sigmas]
        if not isinstance(masks, (list, tuple)):
            masks = [masks]

        [tf.debugging.check_numerics(o, "jointsigma_inputs nan") for o in sigmas]
        mtotal = tf.math.add_n(masks)
        jointvar = tf.math.add_n(
            [m * tf.math.exp(-sigma) for m, sigma in zip(masks, sigmas)]
        )

        # add correction if all modalities are zero
        # to avoid division by zero
        c = tf.where(mtotal < 1, tf.ones_like(mtotal), tf.zeros_like(mtotal))
        jointvar += c

        jointvar = tf.math.log(tf.math.reciprocal(jointvar))
        tf.debugging.check_numerics(jointvar, "jointvar_output nan")
        return jointvar


class JointMean(layers.Layer):
    """Compute combined latent means across modelities"""

    def __init__(self, *args, **kwargs):
        super(JointMean, self).__init__(*args, **kwargs)

    def get_config(self):
        return super(JointMean, self).get_config()

    def compute_output_shape(self, input_mu, input_sigma, input_masks):
        return input_mu[0]

    def call(self, mus, sigmas, masks):
        if not isinstance(mus, (list, tuple)):
            # some issue with keras?
            # after loading a saved model this sees
            mus = [mus]
        if not isinstance(sigmas, (list, tuple)):
            sigmas = [sigmas]
        if not isinstance(masks, (list, tuple)):
            masks = [masks]

        jointmean = tf.math.add_n(
            [m * mu * tf.math.exp(-sigma) for m, mu, sigma in zip(masks, mus, sigmas)]
        )
        jointvar = tf.math.add_n(
            [m * tf.math.exp(-sigma) for m, sigma in zip(masks, sigmas)]
        )

        # add correction if all modalities are zero
        # to avoid division by zero
        mtotal = tf.math.add_n(masks)
        c = tf.where(mtotal < 1, tf.ones_like(mtotal), tf.zeros_like(mtotal))
        jointvar += c

        jointmean *= tf.math.reciprocal(jointvar)
        tf.debugging.check_numerics(jointmean, "jointmean_output nan")
        return jointmean


class ClipLayer(layers.Layer):
    """Clip values between a min and max value

    This is used to avoid too large or too small variance terms.
    """

    def __init__(self, min_value, max_value, *args, **kwargs):
        super(ClipLayer, self).__init__(*args, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(
            inputs, clip_value_min=self.min_value, clip_value_max=self.max_value
        )

    def get_config(self):
        config = {"min_value": self.min_value, "max_value": self.max_value}
        base_config = super(ClipLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[-1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLlossLayer(layers.Layer):
    def __init__(self, kl_weight, *args, **kwargs):
        self.kl_weight = kl_weight
        super(KLlossLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = {"kl_weight": self.kl_weight}
        base_config = super(KLlossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    """ Compute KL-divergence """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
        kl_loss *= -0.5
        kl_loss *= tf.constant(self.kl_weight)
        tf.debugging.check_numerics(kl_loss, "kl_loss layer nan")
        self.add_loss(kl_loss)
        return z_mean, z_log_var


class MutInfoLayer(layers.Layer):
    """Compute mutual information loss across feature activities

    This can be used to enforce latent feature activities to
    be uncorrelated.
    """

    def __init__(self, start_delay=100, *args, **kwargs):
        self.start_delay = start_delay
        super(MutInfoLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = {"start_delay": self.start_delay}
        base_config = super(MutInfoLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape)

        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=[1 for _ in input_shape[:-1]] + [input_shape[-1]],
            dtype="float32",
            trainable=False,
            initializer=initializers.get("zeros"),
        )

        self.delay_count = self.add_weight(
            name="delay",
            shape=(),
            dtype="float32",
            trainable=False,
            initializer=initializers.get("zeros"),
        )

        self.delay_count.assign_add(-self.start_delay)

        self.built = True

    def call(self, x, training=None):
        input_shape = tf.shape(x)

        def _operation():
            # compute covariance
            x_zero = x - self.moving_mean
            x_zero_0 = tf.expand_dims(x_zero, -1)
            x_zero_1 = tf.expand_dims(x_zero, -2)
            cov = tf.reduce_mean(
                x_zero_0 * x_zero_1, axis=[i for i in range(len(input_shape) - 1)]
            )
            cov = cov + tf.eye(cov.shape[0])

            # estimated covariance matrix needs to be positive-definite
            tf.debugging.assert_positive(tf.linalg.det(cov))

            ml_loss = -0.5 * (
                tf.linalg.logdet(cov)
                - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cov)))
            )
            return ml_loss

        # update feature means
        if training is None or training:
            self.delay_count.assign_add(1)
            x_mean = tf.math.reduce_mean(
                x, axis=[i for i in range(len(input_shape) - 1)], keepdims=True
            )
            self.moving_mean.assign(0.3 * self.moving_mean + 0.7 * x_mean)

        ml_loss = tf.case(
            [(tf.less(self.delay_count, 1), lambda: tf.constant(0.0))],
            default=_operation,
        )

        self.add_loss(ml_loss)

        return x


class MAEEndpoint(layers.Layer):
    """MAE-endpoint"""

    def call(self, inputs):
        targets = None
        targets, masks, mu = inputs

        if targets is not None:

            reconstruction_loss = tf.reduce_mean(
                masks * tf.keras.losses.mean_absolute_error(targets, mu)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "MAEEndpoint NaN")

        return mu


class MSEEndpoint(layers.Layer):
    """MSE-endpoint"""

    def call(self, inputs):
        targets = None
        targets, masks, mu = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                masks * tf.reduce_sum(gaussian_likelihood(targets, mu), axis=-1)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "MSEEndpoint NaN")

        return mu


class GammaEndpoint(layers.Layer):
    """Gamma-endpoint"""

    def call(self, inputs):
        targets = None
        targets, masks, alpha, gamma = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                masks * tf.reduce_sum(gamma_likelihood(targets, alpha, gamma), axis=-1)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "GammaEndpoint NaN")

        return alpha / gamma


class MultinomialEndpoint(layers.Layer):
    """Multinomial-endpoint"""

    def call(self, inputs):
        targets, masks, logits = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                masks * tf.reduce_sum(multinomial_likelihood(targets, logits), axis=-1)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "MultinomialEndpoint NaN")

        return _softmax(logits)


class PoissonEndpoint(layers.Layer):
    """Poisson-endpoint"""

    def call(self, inputs):
        targets, masks, logits = inputs

        if targets is not None:
            reconstruction_loss = -tf.reduce_mean(
                masks * tf.reduce_sum(poisson_likelihood(targets, logits), axis=-1)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "PoissonEndpoint NaN")
        return logits


class BinomialEndpoint(layers.Layer):
    """Binomial model"""

    def __init__(self, N, *args, **kwargs):
        self.N = N
        super(BinomialEndpoint, self).__init__(*args, **kwargs)

    def get_config(self):
        config = {
            "N": self.N,
        }
        base_config = super(BinomialEndpoint, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        targets = None
        targets, masks, logits = inputs

        if targets is not None:
            x = tf.reduce_sum(binomial_likelihood(targets, logits, self.N), axis=-1)
            reconstruction_loss = -tf.reduce_mean(
                masks
                * tf.reduce_sum(binomial_likelihood(targets, logits, self.N), axis=-1)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "BinomialEndpoint NaN")

        return tf.math.sigmoid(logits)


class ZeroInflatedNegativeBinomialEndpoint(layers.Layer):
    """Zero-inflated negative binomial-endpoint"""

    def call(self, inputs):
        targets = None
        targets, masks, logits, r, pi = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                masks
                * tf.reduce_sum(
                    zero_inflated_negative_binomial_likelihood(targets, logits, r, pi),
                    axis=-1,
                )
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(
                reconstruction_loss, "NegativeBinomialEndpoint NaN"
            )

        return r * tf.math.sigmoid(logits) / tf.math.sigmoid(-logits)


class NegativeBinomialEndpoint(layers.Layer):
    """Negative binomial-endpoint"""

    def call(self, inputs):
        targets = None
        targets, masks, logits, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                masks
                * tf.reduce_sum(
                    negative_binomial_likelihood(targets, logits, r), axis=-1
                )
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(
                reconstruction_loss, "NegativeBinomialEndpoint NaN"
            )

        return r * tf.math.sigmoid(logits) / tf.math.sigmoid(-logits)


class NegativeMultinomialEndpoint(layers.Layer):
    """Negative multinomial-endpoint"""

    def call(self, inputs):
        targets = None
        targets, mask, logits, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                mask
                * tf.reduce_sum(
                    negative_multinomial_likelihood(targets, logits, r), axis=-1
                )
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(
                reconstruction_loss, "NegativeMultinomialEndpoint NaN"
            )

        p = softmax1p(logits)
        p0 = softmax1p0(logits)
        return p * r / (p0 + 1e-10)


class DirichletMultinomialEndpoint(layers.Layer):
    """Dirichlet-multinomial-endpoint"""

    def call(self, inputs):
        targets = None
        targets, mask, mul_logits = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                mask * dirichlet_multinomial_likelihood(targets, mul_logits)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(
                reconstruction_loss, "DirichletMultinomialEndpoint NaN"
            )

        p = mul_logits
        return p


class NegativeMultinomialEndpointV2(layers.Layer):
    """Negative multinomial-endpoint"""

    def call(self, inputs):
        targets = None
        targets, mask, mul_logits, p0_logit, r = inputs

        if targets is not None:

            reconstruction_loss = -tf.reduce_mean(
                mask
                * negative_multinomial_likelihood_v2(targets, mul_logits, p0_logit, r)
            )
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(
                reconstruction_loss, "NegativeMultinomialEndpoint NaN"
            )

        p0 = tf.math.sigmoid(p0_logit)
        p1 = tf.math.sigmoid(-p0_logit)

        f = p1 * r / p0
        pm = _softmax(mul_logits)
        return pm * f


class MixtureModelEndpoint(layers.Layer):
    """General mixture model-endpoint"""

    def __init__(self, model, *args, **kwargs):
        self.model = model
        if model == "gaussian":
            self.likelihood = gaussian_likelihood
            self.mean = gaussian_mean
        elif model == "gamma":
            self.likelihood = gamma_likelihood
            self.mean = gamma_mean
        else:
            raise ValueError(f"unknown model: {model}")
        super(MixtureModelEndpoint, self).__init__(*args, **kwargs)

    def get_config(self):
        config = {
            "model": self.model,
        }
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
            inp = (
                targets,
                mask,
            ) + tuple(params)
            reconstruction_loss = -tf.reduce_mean(
                tf.math.reduce_sum(
                    mask
                    * tf.math.reduce_logsumexp(
                        self.likelihood(*inp)
                        + tf.math.log_softmax(pi)
                        - tf.math.log(1e5),
                        axis=-1,
                    ),
                    -1,
                )
            )
            # gaussian_likelihood(targets, mu):
            self.add_loss(reconstruction_loss)

            tf.debugging.check_numerics(reconstruction_loss, "MixtureModel NaN")
        mu = self.mean(*params)
        pred = mixture_mean(pi, mu)
        return pred


# defining all custom classes for reloading
CUSTOM_OBJECTS = {
    "Sampling": Sampling,
    "KLlossLayer": KLlossLayer,
    "ClipLayer": ClipLayer,
    "MAEEndpoint": MAEEndpoint,
    "MSEEndpoint": MSEEndpoint,
    "MultinomialEndpoint": MultinomialEndpoint,
    "BinomialEndpoint": BinomialEndpoint,
    "PoissonEndpoint": PoissonEndpoint,
    "NegativeBinomialEndpoint": NegativeBinomialEndpoint,
    "ZeroInflatedNegativeBinomialEndpoint": ZeroInflatedNegativeBinomialEndpoint,
    "NegativeMultinomialEndpoint": NegativeMultinomialEndpoint,
    "NegativeMultinomialEndpointV2": NegativeMultinomialEndpointV2,
    "MutInfoLayer": MutInfoLayer,
    "MixtureModelEndpoint": MixtureModelEndpoint,
    "JointMean": JointMean,
    "JointSigma": JointSigma,
}
