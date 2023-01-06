import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scmaui.layers import *


def _concatenate(x):
    if len(x) > 1:
        x = layers.Concatenate()(x)
    elif len(x) == 1:
        x = x[0]
    else:
        x = None
    return x


def resnet_block(x0, x):
    x = layers.LayerNormalization()(x)
    x = layers.Dense(x.shape[-1], activation="elu")(x)
    x = layers.Add()([x0, x])
    return x


def create_modality_encoder(inputs, cond, params):
    """build a modality-specific encoder"""
    latent_dim = params["nlatent"]
    nhidden_e = params["nunits_encoder"]
    hidden = []
    x = inputs

    x = layers.Dropout(params["dropout_input"])(x)

    x = layers.Dense(nhidden_e, activation="elu")(x)
    if cond is not None:
        x = layers.Concatenate()([x, cond])
    hidden.append(x)
    x0 = x
    for i in range(params["nlayers_encoder"]):
        x = resnet_block(x0, x)

        hidden.append(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z_log_var = ClipLayer(-10.0, 10.0)(z_log_var)

    return hidden, z_mean, z_log_var


def create_encoder_base(params):
    """Build encoders for all modalities and combine them."""
    # configs
    input_shape = params["inputdims"]
    mode_names = params["input_modality"]

    # inputs and modality masks
    inputs = tuple(
        [
            keras.Input(shape=(inp,), name="modality_" + name)
            for inp, name in zip(input_shape, mode_names)
        ]
    )
    masks = tuple([keras.Input(shape=(1,), name="mask_" + name) for name in mode_names])

    # conditional inputs
    conditional = tuple(
        [
            keras.Input(shape=(ncat,), name="condinput_" + name)
            for name, ncat in zip(params["conditional_name"], params["conditional_dim"])
        ]
    )
    covariates = _concatenate(conditional)

    # adversarial inputs
    adversarial = tuple(
        [
            keras.Input(shape=(ncat,), name="advinput_" + name)
            for name, ncat in zip(params["adversarial_name"], params["adversarial_dim"])
        ]
    )

    # resnet per modality
    hiddens = []
    z_means = []
    z_log_vars = []
    for inp in inputs:
        hidden, z_mean, z_log_var = create_modality_encoder(inp, covariates, params)
        hiddens += hidden
        z_means.append(z_mean)
        z_log_vars.append(z_log_var)

    z_joint_log_var = JointSigma(name="z_sigma")(z_log_vars, masks)
    z_joint_mean = JointMean(name="z_mean")(z_means, z_log_vars, masks)
    z_joint_mean, z_joint_log_var = KLlossLayer(kl_weight=params["kl_weight"])(
        [z_joint_mean, z_joint_log_var]
    )

    z = Sampling(name="random_latent")([z_joint_mean, z_joint_log_var])

    outputs = [z]
    if len(adversarial) > 0:
        # if adversarial labels are available, add
        # a discriminator on top of the latent features.
        loss = get_adversarial_net(z_means, masks, adversarial, params)
        outputs.append(loss)

    encoder = keras.Model(
        ((inputs, masks), (conditional, adversarial)), outputs, name="encoder"
    )

    return encoder


def create_common_encoder(params):
    """Build encoders for all modalities and combine them."""

    # configs
    input_shape = params["inputdims"]
    mode_names = params["input_modality"]

    # inputs and modality masks
    inputs = tuple(
        [
            keras.Input(shape=(inp,), name="modality_" + name)
            for inp, name in zip(input_shape, mode_names)
        ]
    )
    masks = tuple([keras.Input(shape=(1,), name="mask_" + name) for name in mode_names])

    inputs_ = _concatenate(inputs)

    # conditional inputs
    conditional = tuple(
        [
            keras.Input(shape=(ncat,), name="condinput_" + name)
            for name, ncat in zip(params["conditional_name"], params["conditional_dim"])
        ]
    )
    covariates = _concatenate(conditional)

    # adversarial inputs
    adversarial = tuple(
        [
            keras.Input(shape=(ncat,), name="advinput_" + name)
            for name, ncat in zip(params["adversarial_name"], params["adversarial_dim"])
        ]
    )

    # resnet per modality
    hiddens = []
    hidden, z_mean, z_log_var = create_modality_encoder(inputs_, covariates, params)
    hiddens += hidden

    z_joint_mean = JointMean(name="z_mean")([z_mean], [z_log_var], masks[:1])

    z_mean, z_log_var = KLlossLayer(kl_weight=params["kl_weight"])(
        [z_joint_mean, z_log_var]
    )

    z = Sampling(name="random_latent")([z_mean, z_log_var])

    outputs = [z]
    if len(adversarial) > 0:
        # if adversarial labels are available, add
        # a discriminator on top of the latent features.
        loss = get_adversarial_net([z_mean], masks, adversarial, params)
        outputs.append(loss)

    encoder = keras.Model(
        ((inputs, masks), (conditional, adversarial)), outputs, name="encoder"
    )

    return encoder


def regressout(x, b, nhidden):
    breg = layers.Dense(nhidden)
    breg0 = layers.Dense(nhidden)
    b0 = tf.zeros_like(b)
    x_hat = breg(b)
    x_hat_0 = breg0(b0)

    loss = tf.math.reduce_mean(tf.math.square(x - x_hat))
    loss += tf.math.reduce_mean(tf.math.square(x - x_hat_0))

    output = x - (breg(b) - breg0(b0))
    return output, loss


def create_decoder_base(params):
    """build decoders for all modalities"""
    input_shape = params["outputdims"]
    latent_dim = params["nlatent"]
    losses = params["losses"]
    mode_names = params["output_modality"]

    latent_inputs = keras.Input(shape=(latent_dim,), name="latent_input")

    targets = tuple(
        [
            keras.Input(shape=(inp,), name="target_" + name)
            for inp, name in zip(input_shape, mode_names)
        ]
    )
    masks = tuple(
        [keras.Input(shape=(1,), name="target_mask_" + name) for name in mode_names]
    )

    # conditional inputs
    intercept = (keras.Input(shape=(1,), name="intercept_"),)

    # conditional inputs
    conditional = tuple(
        [
            keras.Input(shape=(ncat,), name="cond_" + name)
            for name, ncat in zip(params["conditional_name"], params["conditional_dim"])
        ]
    )

    # adversarial inputs
    adversarial = tuple(
        [
            keras.Input(shape=(ncat,), name="adv_" + name)
            for name, ncat in zip(params["adversarial_name"], params["adversarial_dim"])
        ]
    )
    covariates = intercept + conditional + adversarial
    covariates = _concatenate(covariates)

    outputs = [
        create_modality_decoder(params, latent_inputs, target, mask, covariates, loss)
        for target, mask, loss in zip(targets, masks, losses)
    ]

    decoder = keras.Model(
        (
            latent_inputs,
            targets,
            masks,
            intercept,
            conditional,
            adversarial,
        ),
        outputs,
        name="decoder",
    )

    return decoder


def create_modality_decoder(params, latent, target, mask, covariates, loss):
    """build a modality specific decoder"""
    x = latent
    x = layers.Concatenate()([latent, covariates])

    for nhidden in range(params["nlayers_decoder"]):
        x = layers.Dense(params["nunits_decoder"], activation="elu")(x)
        x = layers.Dropout(params["dropout_decoder"])(x)

    target_ = target
    mask_ = mask

    logits = layers.Dense(target.shape[1], activation="linear")(x)

    covariate_logits = layers.Dense(target.shape[1], activation="linear")(covariates)
    logits = layers.Add()([logits, covariate_logits])

    if loss == "negmul":

        r = layers.Dense(1, activation="linear")(covariates)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeMultinomialEndpoint()([target_, mask_, logits, r])

    elif loss == "negmul2":

        logits_mul = logits
        logits_nb = layers.Dense(1, activation="linear")(covariates)
        r = layers.Dense(1, activation="linear")(covariates)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeMultinomialEndpointV2()(
            [target_, mask_, logits_mul, logits_nb, r]
        )

    elif loss == "dirmul":

        logits_mul = logits
        logits_mul = layers.Activation(activation=tf.math.softplus)(logits)

        prob_loss = DirichletMultinomialEndpoint()([target_, mask_, logits_mul])

    elif loss == "mul":

        prob_loss = MultinomialEndpoint()([target_, mask_, logits])

    elif loss == "binary":

        prob_loss = BinomialEndpoint(
            1,
        )([target_, mask_, logits])

    elif loss == "poisson":

        logits = layers.Activation(activation=tf.math.softplus)(logits)
        prob_loss = PoissonEndpoint()([target_, mask_, logits])

    elif loss == "mae":

        mu = logits
        prob_loss = MAEEndpoint()([target_, mask_, mu])

    elif loss == "mse":

        mu = logits
        prob_loss = MSEEndpoint()([target_, mask_, mu])

    elif loss == "negbinom":

        r = layers.Dense(1, activation="linear")(covariates)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeBinomialEndpoint()([target_, mask_, logits, r])

    elif loss == "zinb":

        r = layers.Dense(1, activation="linear")(covariates)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        pi = layers.Dense(1, activation="linear")(x)

        prob_loss = ZeroInflatedNegativeBinomialEndpoint()(
            [target_, mask_, logits, r, pi]
        )

    elif loss == "mixgaussian":

        pi = layers.Dense(params["nmixcomp"], activation="linear")(x)

        mu = layers.Dense(params["nmixcomp"] * target.shape[1], activation="linear")(x)
        sigma = layers.Dense(params["nmixcomp"] * target.shape[1], activation="linear")(
            x
        )
        sigma = layers.Activation(activation=tf.math.softplus)(sigma)

        mu = layers.Reshape((target.shape[1], params["nmixcomp"]))(mu)
        sigma = layers.Reshape((target.shape[1], params["nmixcomp"]))(sigma)

        prob_loss = MixtureModelEndpoint(model="gaussian")(
            [target_, mask_, mu, sigma, pi]
        )
    # elif loss == 'mixgamma':
    #    pass
    else:
        raise ValueError(f"Unknown reconstruction loss: {loss}")

    return prob_loss
    # return prob_loss, logits


# Network parts for the adversarial network modules


def _getlabel(name):
    if name == "category":
        return "softmax"
    else:
        return "linear"


def get_loss(t, mask, p, ptype):
    """use a categorical or numerical adversarial loss depending on  ptype"""
    if ptype == "category":
        loss = mask * tf.reduce_sum(-tf.math.xlogy(t, p + 1e-9), axis=-1, keepdims=True)
    else:
        # numeric
        m_ = tf.where(tf.math.is_nan(t), tf.zeros_like(t), tf.ones_like(t))
        t_ = tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
        losses = m_ * tf.math.square(t_ - p)
        loss = mask * tf.reduce_sum(losses, axis=-1, keepdims=True)
    return loss


def adversarial_predictor(layer, mask, targets, params, idx):
    """build an adversarial network on top of layer"""
    x = layer

    for i in range(params["nlayers_adversary"]):
        x = layers.Dense(
            params["nunits_adversary"], activation="elu", name=f"advnet_{idx}1{i}"
        )(x)
        # x = layers.BatchNormalization(name=f'advnet_{idx}2{i}')(x)
    pred_targets = [
        layers.Dense(
            dim, activation=_getlabel(ptype), name=f"advnet_out_{idx}" + bname
        )(x)
        for dim, ptype, bname in zip(
            params["adversarial_dim"],
            params["adversarial_type"],
            params["adversarial_name"],
        )
    ]

    losses = [
        get_loss(t, mask, p, ptype)
        for t, p, ptype in zip(targets, pred_targets, params["adversarial_type"])
    ]
    losses = tf.math.add_n(losses)

    return losses


def get_adversarial_net(layers, masks, targets, params):
    """aggregate loss from all adversarial network modules"""
    ret = [
        adversarial_predictor(layer, mask, targets, params, idx)
        for idx, (layer, mask) in enumerate(zip(layers, masks))
    ]
    losses = tf.math.reduce_sum(tf.math.add_n([r for r in ret]))

    return losses
