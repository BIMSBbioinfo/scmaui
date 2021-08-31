import os
from itertools import product
from collections import OrderedDict
import sys
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def init_model_params():
    params = OrderedDict(
      [
        ('nhidden_e', 32),
        ('nlayers_e', 5),
        ('nhiddendecoder', 20),
        ('nlayers_d', 1),
        ('inputdropout', 0.1),
        ('hidden_e_dropout', 0.0),
        ('hidden_d_dropout', 0.0),
        ('nhiddenbatcher', 128),
        ('nlayersbatcher', 2),
        ('nlasthiddenbatcher', 5),
        ('latentdims', 10),
        ('nmixcomp', 1),
      ]
    )

    return params
   
def get_model_params(dataset, args=None):
    params = init_model_params()

    modalities = dataset.modalities()
    params['inputmodality'] = modalities[0]
    params['outputmodality'] = modalities[1]

    params.update(dataset.adversarial_config())
    params.update(dataset.conditional_config())
    params.update({'losses': dataset.losses})

    if args is not None:

        nparams = OrderedDict(
          [
            ('nlayers_d', args.nlayers_d),
            ('nhidden_e', args.nhidden_e),
            ('nlayers_e', args.nlayers_e),
            ('nhiddendecoder', args.nhidden_d),
            ('inputdropout', args.inputdropout),
            ('hidden_e_dropout', args.hidden_e_dropout),
            ('hidden_d_dropout', args.hidden_d_dropout),
            ('nhiddenbatcher', args.nhidden_b),
            ('nlayersbatcher', 2),
            ('nlasthiddenbatcher', 5),
            ('latentdims', args.nlatent),
            ('losses', args.loss),
            ('nmixcomp', args.nmixcomp),
          ]
        )
        params.update(nparams)

    return params



