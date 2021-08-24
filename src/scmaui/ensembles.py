import os
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
import anndata as ad
from scmaui.vae_models import *
from scmaui.model_components import *
from scmaui.data import to_dataset
from scmaui.data import to_sparse
from scipy.stats import iqr

class EnsembleVAE:
    def __init__(self, params, repeats, model=None, feature_fraction=1.):
        self.repeats = repeats
        self.models = []

        self.space = params
        self.feature_fraction = max(min(feature_fraction, 1.), 0.)
        if model is None:
            model = 'vae'
        self.model = model
        print(f'using {self.model}')

    def _create(self, model, space):
        """ Create VAE model"""
        if model == 'vae':
            model = VAE.create(space, create_encoder_base, create_decoder_base)
        else:
            raise ValueError(f"Unknown model: {model}")
        return model

    def _load(self, path):
        """ Reload VAE model"""
        if self.model == 'vae':
            model = VAE.load(path)
        else:
            raise ValueError(f"Unknown model: {model}")
        return model

    def save(self, path, overwrite=False):
        """ save the ensemble """
        for r, model in enumerate(self.models):
            subpath = os.path.join(path, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            model.save(os.path.join(subpath, 'model', 'vae.h5'))

    def load(self, path):
        """ load a previously trained ensemble """
        for r in range(self.repeats):
            subpath = os.path.join(path, f'repeat_{r+1}')
            if not os.path.exists(os.path.join(subpath, 'model')):
                print(f'no model in {subpath}')
                print('re-load model')
                model = self._load(os.path.join(subpath, 'model', 'vae.h5'))
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                self.models.append(model)

    def fit(self, dataset,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15,):
        """ fit an ensemble of VAE models."""
        space = self.space
        histories = []

        for r in range(self.repeats):
            tf_X, tf_X_test = dataset.training_data(batch_size=batch_size, validation_split=0.15)

            print(f'Run repetition {r+1}')
            space.update(dataset.shapes())
            model = self._create(self.model, space)

            model.compile(optimizer=
                          keras.optimizers.Adam(
                              learning_rate=0.001,
                              amsgrad=True),
                         )
            history = model.fit(tf_X, epochs = epochs, validation_data=(tf_X_test,),
                                callbacks=CSVLogger('train_summary.csv'))
            histories.append(history)
            self.models.append(model)
        return histories

    def encode_full(self, dataset, batch_size=64, skip_outliers=True):
        tf_X = dataset.evaluation_data(batch_size=batch_size)
        dfs = []
        for i, model in enumerate(self.models):
           
            out = model.encoder_mean.predict(tf_X)
            df = pd.DataFrame(out, index=dataset.adata['input'][0].obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df, dfs

    def encode(self, dataset, batch_size=64, skip_outliers=True):
        """ Inference on the latent features """
        return self.encode_full(dataset, batch_size, skip_outliers=skip_outliers)

    def impute(self, dataset, batch_size=64):
        """ Inference on the output features """
        tf_X = dataset.imputation_data(batch_size=batch_size)

        output = [np.zeros((dataset.size(), m)) for m in dataset.shapes()['outputdims']]
        
        for i, model in enumerate(self.models):
            out = model.predict(tf_X)
            for i, o in enumerate(out):
                output[i] += o / len(self.models)

        return output
