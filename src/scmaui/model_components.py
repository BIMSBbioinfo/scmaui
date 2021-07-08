import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scmaui.layers import ClipLayer
from scmaui.layers import KLlossLayer
from scmaui.layers import Sampling
from scmaui.layers import ExpandDims
from scmaui.layers import BatchLoss
from scmaui.layers import MutInfoLayer
from scmaui.layers import MSEEndpoint
from scmaui.layers import GammaEndpoint
from scmaui.layers import MixtureModelEndpoint
from scmaui.layers import MultinomialEndpoint
from scmaui.layers import BinomialEndpoint
from scmaui.layers import NegativeBinomialEndpoint
from scmaui.layers import ZeroInflatedNegativeBinomialEndpoint
from scmaui.layers import NegativeMultinomialEndpoint
from scmaui.layers import NegativeMultinomialEndpointV2

def _concatenate(x):
    if len(x)>1:
        x = layers.Concatenate()(x)
    else:
        x = x[0]
    return x

def create_encoder_base(params):
    """ Encoder without batch correction."""
    input_shape = params['inputdatadims']
    mask_shape = params['inputmaskdims']
    mode_names = params['inputmodality']
    latent_dim = params['latentdims']

    inputs = tuple([keras.Input(shape=(inp,), name=name) for inp, name in zip(input_shape, mode_names)])
    masks = tuple([keras.Input(shape=(inp,), name='mask_'+name) for inp, name in zip(mask_shape, mode_names)])
    x = _concatenate([layers.Multiply()(l) for l in zip(inputs, masks)])
    
    batches = tuple([keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])])
    batches_ = _concatenate(batches)

    x = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']
    x = layers.Dense(nhidden_e, activation="relu")(x)
    xin = x
    for _ in range(params['nlayers_e']):
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation='relu')(x)
        x = layers.Add()([x, xin])

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(((inputs, masks), batches), z, name="encoder")
    encoder.summary()
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

def create_regout_encoder_base(params):
    """ Condition on batches in all hidden layers."""

    input_shape = params['inputdatadims']
    mask_shape = params['inputmaskdims']
    mode_names = params['inputmodality']
    latent_dim = params['latentdims']

    inputs = [keras.Input(shape=(inp,), name=name) for inp, name in zip(input_shape, mode_names)]
    masks = [keras.Input(shape=(inp,), name='mask_'+name) for inp, name in zip(mask_shape, mode_names)]

    batches = [keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batches_ = _concatenate(batches)

    encoder_inputs = inputs
    encoder_masks = masks
    
    encoder_inputs = _concatenate(encoder_inputs)
    encoder_masks = _concatenate(encoder_masks)
    x = layers.Multiply()([encoder_inputs, encoder_masks])

    x = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']

    xin = x

    loss = 0.0
    #xin = layers.Concatenate()([xin, batches_])
    xin = layers.Dense(nhidden_e, activation="relu")(xin)
    xin, loss_ = regressout(xin, batches_, nhidden_e)
    loss += loss_

    for _ in range(params['nlayers_e']):
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation='linear')(x)
        #xb = layers.Dense(nhidden_e)(batches_)
        x = layers.Add()([x, xin])
        x = layers.Activation(activation='relu')(x)
        x, loss_ = regressout(x, batches_, nhidden_e)
        loss += loss_

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_mean, loss_ = regressout(z_mean, batches_, latent_dim)
    loss += loss_

    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(((inputs, masks), batches), z, name="encoder")
    encoder.add_loss(loss)

    return encoder


def create_cond_encoder_base(params):
    """ Condition on batches in all hidden layers."""

    input_shape = params['inputdatadims']
    mask_shape = params['inputmaskdims']
    mode_names = params['inputmodality']
    latent_dim = params['latentdims']

    inputs = [keras.Input(shape=(inp,), name=name) for inp, name in zip(input_shape, mode_names)]
    masks = [keras.Input(shape=(inp,), name='mask_'+name) for inp, name in zip(mask_shape, mode_names)]

    batches = [keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batches_ = _concatenate(batches)

    encoder_inputs = inputs
    encoder_masks = masks
    
    encoder_inputs = _concatenate(encoder_inputs)
    encoder_masks = _concatenate(encoder_masks)
    x = layers.Multiply()([encoder_inputs, encoder_masks])

    x = layers.Dropout(params['inputdropout'])(x)

    nhidden_e = params['nhidden_e']

    xin = x

    xin = layers.Concatenate()([xin, batches_])
    xin = layers.Dense(nhidden_e, activation="relu")(xin)


    for _ in range(params['nlayers_e']):
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(nhidden_e, activation='linear')(x)
        xb = layers.Dense(nhidden_e)(batches_)
        x = layers.Add()([x, xin, xb])
        x = layers.Activation(activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(((inputs, masks), batches), z, name="encoder")

    return encoder


def create_decoder_base(params):

    input_shape = params['outputdatadims']
    mask_shape = params['outputmaskdims']
    latent_dim = params['latentdims']
    losses = params['losses']
    mode_names = params['outputmodality']

    latent_inputs = keras.Input(shape=(latent_dim,), name='latent_input')

    targets = tuple([keras.Input(shape=(inp,), name='target_'+name) for inp, name in zip(input_shape, mode_names)])
    masks = tuple([keras.Input(shape=(inp,), name='target_mask_'+name) for inp, name in zip(mask_shape, mode_names)])

    batches = tuple([keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])])
 
    #inputs = [latent_inputs,  [targets, masks, batches]]

    outputs = [create_modality_decoder(params,
                                       latent_inputs,
                                       target,
                                       mask,
                                       batches,
                                       loss) for target, mask, loss in zip(targets, masks, losses)]

    decoder = keras.Model((latent_inputs, (targets, masks, batches)), outputs, name="decoder")

    return decoder

def create_modality_decoder(params, latent, target, mask, batch, loss):
    x = latent

    batch_ = _concatenate(batch)

    x = layers.Concatenate()([latent, batch_])

    for nhidden in range(params['nlayers_d']):
        x = layers.Dense(params['nhiddendecoder'], activation="relu")(x)
        x = layers.Dropout(params['hidden_d_dropout'])(x)

    target_ = target
    mask_ = mask

    logits = layers.Dense(target.shape[1], activation='linear')(x)
    
    # add a batch-specific bias
    batch_logits = layers.Dense(target.shape[1], activation='linear')(batch_)
    logits = layers.Add()([logits, batch_logits])

    if loss == 'negmul':
        r = layers.Dense(1, activation='linear')(batch_)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeMultinomialEndpoint()([target_, mask_, logits, r])
    if loss == 'negmul2':
        logits_mul = logits
        logits_nb = layers.Dense(1, activation='linear')(batch_)
        r = layers.Dense(1, activation='linear')(batch_)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)

        prob_loss = NegativeMultinomialEndpointV2()([target_, mask_, logits_mul, logits_nb, r])
    elif loss == 'mul':
        prob_loss = MultinomialEndpoint()([target_, mask_, logits])
    elif loss == 'binary':
        prob_loss = BinomialEndpoint(1, )([target_, mask_, logits])
    elif loss == 'binom':
        prob_loss = BinomialEndpoint(2, )([target_, mask_, logits])
    elif loss == 'mse':
        mu = logits
        #sigma = layers.Dense(target.shape[1], tf.math.softplus)(x)
        #sigma = ClipLayer(1e-10, 1e5)(sigma)
        #prob_loss = MSEEndpoint()([target_, mask_, mu, sigma])
        prob_loss = MSEEndpoint()([target_, mask_, mu, None])
    elif loss == 'negbinom':
        r = layers.Dense(1, activation='linear')(batch_)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)
        prob_loss = NegativeBinomialEndpoint()([target_, mask_, logits, r])
    elif loss == 'zinb':
        r = layers.Dense(1, activation='linear')(batch_)
        r = layers.Activation(activation=tf.math.softplus)(r)
        r = ClipLayer(1e-10, 1e5)(r)
        pi = layers.Dense(1, activation='linear')(x)
        prob_loss = ZeroInflatedNegativeBinomialEndpoint()([ target_, mask_, logits, r, pi])
    elif loss == 'mixgaussian':
        
        pi = layers.Dense(params['nmixcomp'], activation='linear')(x)

        mu = layers.Dense(params['nmixcomp']*target.shape[1], activation='linear')(x)
        sigma = layers.Dense(params['nmixcomp']*target.shape[1], activation='linear')(x)
        sigma = layers.Activation(activation=tf.math.softplus)(sigma)

        mu = layers.Reshape((target.shape[1], params['nmixcomp']))(mu)
        sigma = layers.Reshape((target.shape[1], params['nmixcomp']))(sigma)

        prob_loss = MixtureModelEndpoint(model='gaussian')([target_, mask_, mu, sigma, pi])
    elif loss == 'mixgamma': 
        pass
    # possible future models
    # negative binomial
    
    return prob_loss, logits


def create_encoder_mutinfo(params):
    """ Encoder without batch correction."""
    raise NotImplemented()
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs

    #xinit = layers.Dropout(params['inputdropout'])(x)
    xinit = x

    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)
    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        #x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean = MutInfoLayer()(z_mean)
    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, z, name="encoder")

    return encoder


def create_cond_encoder_base0(params):
    """ Condition on batches at first layer."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    batch_layer = batch_inputs

    batch_layer = _concatenate(batch_layer)

    xinit = layers.Concatenate()([xinit, batch_layer])
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e, activation="relu")(xinit)

    for _ in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    encoder = keras.Model([encoder_inputs, batch_inputs], z, name="encoder")

    return encoder


def create_batch_encoder_gan(params):
    """ With batch-adversarial learning on all hidden layers."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batches = []
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e)(xinit)
    batches.append(create_batch_net(xinit, params, '00'))
    xinit = layers.Activation(activation='relu')(xinit)


    for i in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        batches.append(create_batch_net(x, params, f'1{i}'))
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    batches.append(create_batch_net(z, params, f'20'))

    pred_batches = combine_batch_net(batches)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    true_batch_layer = [ExpandDims()(l) for l in batch_inputs]

    batch_loss = BatchLoss(name='batch_loss')([pred_batches, true_batch_layer])

    encoder = keras.Model([encoder_inputs, batch_inputs], [z, batch_loss], name="encoder")

    return encoder


def create_batch_encoder_gan_lastlayer(params):
    """ With batch-adversarial learning on last latent dims."""
    input_shape = (params['datadims'],)
    latent_dim = params['latentdims']

    encoder_inputs = keras.Input(shape=input_shape,
                                 name='input_data')

    x = encoder_inputs
    xinit = layers.Dropout(params['inputdropout'])(x)

    batches = []
    nhidden_e = params['nhidden_e']
    xinit = layers.Dense(nhidden_e)(xinit)
    #batches.append(create_batch_net(xinit, params, '00'))
    xinit = layers.Activation(activation='relu')(xinit)


    for i in range(params['nlayers_e']):
        x = layers.Dense(nhidden_e, activation="relu")(xinit)
        x = layers.Dropout(params['hidden_e_dropout'])(x)
        x = layers.Dense(nhidden_e)(x)
        x = layers.Add()([x, xinit])
        #batches.append(create_batch_net(x, params, f'1{i}'))
        xinit = layers.Activation(activation='relu')(x)

    x = xinit

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = ClipLayer(-10., 10.)(z_log_var)

    z_mean, z_log_var = KLlossLayer()([z_mean, z_log_var])

    z = Sampling(name='random_latent')([z_mean, z_log_var])

    batches.append(create_batch_net(z, params, f'20'))

    pred_batches = combine_batch_net(batches)

    batch_inputs = [keras.Input(shape=(ncat,), name='batch_input_'+bname) for bname, ncat in zip(params['batchnames'], params['nbatchcats'])]
    true_batch_layer = [ExpandDims()(l) for l in batch_inputs]

    batch_loss = BatchLoss(name='batch_loss')([pred_batches, true_batch_layer])

    encoder = keras.Model([encoder_inputs, batch_inputs], [z, batch_loss], name="encoder")

    return encoder


def create_batch_net(inlayer, params, name):
    x = layers.BatchNormalization(name='batchcorrect_batch_norm_1_'+name)(inlayer)
    for i in range(params['nlayersbatcher']):
       x = layers.Dense(params['nhiddenbatcher'], activation='relu', name=f'batchcorrect_{name}_hidden_{i}')(x)
       x = layers.BatchNormalization(name=f'batchcorrect_batch_norm_2_{name}_{i}')(x)
    if len(x.shape.as_list()) <= 2:
        x = ExpandDims()(x)
    targets = [layers.Dense(nl, activation='softmax', name='batchcorrect_'+name + '_out_' + bname)(x) \
               for nl,bname in zip(params['nbatchcats'], params['batchnames'])]
    return targets

def combine_batch_net(batches):
    if len(batches)<=1:
        return batches[0]
    new_output = []
    for bo,_ in enumerate(batches[0]):
        new_output.append(layers.Concatenate(axis=1, name=f'combine_batches_{bo}')([batch[bo] for batch in batches]))
    return new_output

