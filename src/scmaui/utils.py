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
            ("nunits_encoder", 32),
            ("nlayers_encoder", 5),
            ("nunits_decoder", 20),
            ("nlayers_decoder", 1),
            ("dropout_input", 0.1),
            ("dropout_encoder", 0.0),
            ("dropout_decoder", 0.0),
            ("nunits_adversary", 128),
            ("nlayers_adversary", 2),
            ("kl_weight", 0.0),
            ("nlatent", 10),
            ("nmixcomp", 1),
        ]
    )

    return params


def get_model_params(dataset, args=None):
    params = init_model_params()

    modalities = dataset.modalities()
    params["input_modality"] = modalities[0]
    params["output_modality"] = modalities[1]

    params.update(dataset.adversarial_config())
    params.update(dataset.conditional_config())
    params.update({"losses": dataset.losses})

    if args is not None:

        nparams = OrderedDict(
            [
                ("nlayers_encoder", args.nlayers_encoder),
                ("nunits_encoder", args.nunits_encoder),
                ("nlayers_decoder", args.nlayers_decoder),
                ("nunits_decoder", args.nunits_decoder),
                ("dropout_input", args.dropout_input),
                ("dropout_encoder", args.dropout_encoder),
                ("dropout_decoder", args.dropout_decoder),
                ("nunits_adversary", args.nunits_adversary),
                ("nlayers_adversary", 2),
                ("nlatent", args.nlatent),
                ("losses", args.loss),
                ("nmixcomp", args.nmixcomp),
            ]
        )
        params.update(nparams)

    return params
