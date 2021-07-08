import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
import anndata as ad
from scmaui.vae_models import *
from scmaui.model_components import *
from scmaui.data import to_dataset
from scmaui.data import to_encoder_dataset
from scmaui.data import to_sparse
from scipy.stats import iqr

class BatchEnsembleVAE:
    #def __init__(self, params, repeats, output, overwrite, name='vae', feature_fraction=1.):
    def __init__(self, name, params, repeats, output, overwrite, feature_fraction=1., batchnames=[],
                 adversarial=True, conditional=False, ):
        self.repeats = repeats
        self.output = output
        self.models = []
        self.joined_model = None
        self.overwrite = overwrite
        if os.path.exists(output) and overwrite:
            shutil.rmtree(output)

        os.makedirs(output, exist_ok=True)
        self.space = params
        self.feature_fraction = max(min(feature_fraction, 1.), 0.)
        self.name = name
        self.batchnames = ['dummy']
        self.batchnames = batchnames
        #self.name = name
        self.adversarial = True if 'scmaui' in name else False
        self.conditional = True if name.startswith('cond') else False
        print(f'using {self.name}')

    def _get_label(self, adata, dummy_labels=True):
        labels=[]
        firstkey = list(adata.keys())[0]
        for label  in self.batchnames:
            if label in adata[firstkey].obsm:
                labels.append(adata[firstkey].obsm[label])
            else:
                labels.append(np.ones((adata.shape[0], 1)))
            if dummy_labels:
                labels[-1] = np.zeros_like(labels[-1])
                labels[-1][:,0] = 1
        return labels

    def _split_train_test_data(self, x_data, validation_split):
        """ Split training - validation input data """
        x_split = [train_test_split(x,
                                    test_size=validation_split,
                                    random_state=42) for x in x_data]
        x_train = [x[0] for x in x_split]
        x_test = [x[1] for x in x_split]
        return x_train, x_test

    def _split_train_test(self, x_data, mask, label, validation_split):
        """ Split training - validation (data + labels)"""
        x_train, x_test = self._split_train_test_data(x_data, validation_split)
        mask_train, mask_test = self._split_train_test_data(mask, validation_split)
        label_train, label_test = self._split_train_test_data(label, validation_split)
        return x_train, x_test, mask_train, mask_test, label_train, label_test

    def _get_predict_data(self, adata, dummy_labels=True):
        """ get predict data + labels.
        Labels are only relevant for batch annotation.
        """
        x_data = [to_sparse(adata[k].X) for k in adata]
        mask = [to_sparse(adata[k].obsm['mask']) for k in adata]
        labels = self._get_label(adata, dummy_labels=dummy_labels)
        return x_data, mask, labels

    def _get_data(self, adata):
        x_data = [to_sparse(adata[k].X) for k in adata]
        mask = []
        for k in adata:
             if 'mask' not in adata[k].obsm:
                 adata[k].obsm['mask'] = np.ones((adata[k].shape[0], 1))
             m = to_sparse(adata[k].obsm['mask'])
             mask.append(m)
        return x_data, mask

    def _get_input_data(self, adata, dummy_labels=False):
        x_data, mask_x = self._get_data(adata['input'])
        labels = self._get_label(adata['input'], dummy_labels=dummy_labels)
        return x_data, mask_x, labels

 
    def _get_input_output_data(self, adata, dummy_labels=False):
        x_data, mask_x = self._get_data(adata['input'])
        y_data, mask_y = self._get_data(adata['output'])
        labels = self._get_label(adata['input'], dummy_labels=dummy_labels)
        return x_data, mask_x, y_data, mask_y, labels

    def _create(self, name, space):
        """ Create VAE model"""
        if name == 'vae':
            model = VAE.create(space, create_encoder_base, create_decoder_base)
        elif name == 'cond-vae':
            model = VAE.create(space, create_cond_encoder_base, create_decoder_base)
        elif name == 'regout-vae':
            model = VAE.create(space, create_regout_encoder_base, create_decoder_base)
        elif name == 'scmaui':
            model = BAVARIA.create(space, create_adv_encoder_base, create_batch_decoder)
        #elif name == 'scmaui':
        #    model = BAVARIA.create(space, create_batch_encoder_gan, create_batch_decoder)
        elif name == 'scmaui-0':
            model = BAVARIA.create(space, create_batch_encoder_gan_lastlayer, create_batch_decoder)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def _load(self, path):
        """ Reload VAE model"""
        if self.name == 'vae':
            model = VAE.load(path)
        elif self.name == 'cond-vae':
            model = VAE.load(path)
        elif self.name == 'regout-vae':
            model = VAE.load(path)
        elif self.name == 'vae-ml':
            model = VAE.load(path)
        elif self.name == 'scmaui':
            model = BAVARIA.load(path)
        elif self.name == 'scmaui-0':
            model = BAVARIA.load(path)
        else:
            raise ValueError(f"Unknown model: {name}")
        return model

    def fit(self, adata,
            shuffle=True, batch_size=64,
            epochs=1, validation_split=.15
           ):
        """ fit ensemble of VAE models (or reload pre-existing model)."""
        space = self.space
        X, mask_x, Y, mask_y, label = self._get_input_output_data(adata)

        for r in range(self.repeats):
            # random feature subset
            #x_subdata = X

            x_train, x_test = self._split_train_test_data(X, validation_split)
            y_train, y_test = self._split_train_test_data(Y, validation_split)
            mask_x_train, mask_x_test = self._split_train_test_data(mask_x, validation_split)
            mask_y_train, mask_y_test = self._split_train_test_data(mask_y, validation_split)
            label_train, label_test = self._split_train_test_data(label, validation_split)

            x_train = [to_sparse(x) for x in x_train]
            x_test = [to_sparse(x) for x in x_test]
            mask_x_train = [to_sparse(x) for x in mask_x_train]
            mask_x_test = [to_sparse(x) for x in mask_x_test]
            y_train = [to_sparse(x) for x in y_train]
            y_test = [to_sparse(x) for x in y_test]
            mask_y_train = [to_sparse(x) for x in mask_y_train]
            mask_y_test = [to_sparse(x) for x in mask_y_test]

            tf_X = to_dataset(x_train, mask_x_train, y_train, mask_y_train, label_train, shuffle=shuffle, batch_size=batch_size)
            tf_X_test = to_dataset(x_test, mask_x_test, y_test, mask_y_test, label_test, shuffle=False)

            subpath = os.path.join(self.output, f'repeat_{r+1}')
            os.makedirs(subpath, exist_ok=True)
            if not os.path.exists(os.path.join(subpath, 'model')):

                print(f'Run repetition {r+1}')
                space['inputdatadims'] = [x.shape[1] for x in x_train]
                space['inputmaskdims'] = [x.shape[1] for x in mask_x_train]
                space['outputdatadims'] = [x.shape[1] for x in y_train]
                space['outputmaskdims'] = [x.shape[1] for x in mask_y_train]
                model = self._create(self.name, space)

                # initialize the output bias based on the overall read coverage
                # this slightly improves results
                #model.decoder.get_layer('extra_bias').set_weights([output_bias])

                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True),
                             )
                csvcb = CSVLogger(os.path.join(subpath, 'train_summary.csv'))
                model.fit(tf_X, epochs = epochs,
                          validation_data=(tf_X_test,),
                          callbacks=[csvcb])
                model.save(os.path.join(subpath, 'model', 'vae.h5'))
                self.models.append(model)
            else:
                model = self._load(os.path.join(subpath, 'model', 'vae.h5'))
                model.compile(optimizer=
                              keras.optimizers.Adam(
                                  learning_rate=0.001,
                                  amsgrad=True)
                             )
                self.models.append(model)
        model.summary()

    def _get_dataset_truebatchlabels(self, adata, batch_size=64):
        """ used without dummy labels"""
        x_data, mask, labels = self._get_predict_data(adata, dummy_labels=False)
        tf_x = to_dataset(x_data, mask, labels, shuffle=False, batch_size=batch_size)
        tf_x = tf.data.Dataset.zip((tf_x,))
        return tf_x

    def _get_dataset_dummybatchlabels(self, adata, batch_size=64):
        """ used with dummy labels"""

        x_data, mask, labels = self._get_predict_data(adata, dummy_labels=True)
        tf_x = to_dataset(x_data, mask, labels, shuffle=False, batch_size=batch_size)
        tf_x = tf.data.Dataset.zip((tf_x,))
        return tf_x

    def encode_full(self, adata, oadata, batch_size=64, skip_outliers=True):
        firstkey = list(adata['input'].keys())[0]
        dummy = True if self.conditional else False

        X, mask_x, label = self._get_input_data(adata, dummy_labels=dummy)
        tf_X = to_encoder_dataset(X, mask_x, label, shuffle=False, batch_size=batch_size)
        dfs = []
        for i, model in enumerate(self.models):
           
            out = model.encode(tf_X)
            #out = model.encoder.predict(tf_X)
            df = pd.DataFrame(out, index=adata['input'][firstkey].obs.index, columns=[f'D{i}-{n}' for n in range(out.shape[1])])
            df.to_csv(os.path.join(self.output, f'repeat_{i+1}', 'latent.csv'))
            oadata.obsm[f'scmaui-run_{i+1}'] = out
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        oadata.obsm['scmaui-ensemble'] = df.values
        df.to_csv(os.path.join(self.output, 'latent.csv'))
        return oadata

    def encode(self, adata, oadata, batch_size=64, skip_outliers=True):
        return self.encode_full(adata, oadata, batch_size, skip_outliers=skip_outliers)

    def impute(self, adata, batch_size=64):

        dummy = True if self.conditional else False

        X, mask_x, Y, mask_y, label = self._get_input_output_data(adata, dummy_labels=dummy)

        tf_X = to_dataset(X, mask_x, Y, mask_y, label, shuffle=False, batch_size=batch_size)

        firstkey = list(adata['input'].keys())[0]
        for i, model in enumerate(self.models):
            out = model.impute(tf_X)
            
            if not isinstance(out, list):
                out = [out]
            for j, k in enumerate(adata['input'].keys()):
                adata['input'][firstkey].obsm[f'scmaui-{k}_impute_prob_{i+1}'] = out[j][0]
                adata['input'][firstkey].obsm[f'scmaui-{k}_impute_score_{i+1}'] = out[j][1]
        N = len(self.models)
        for j, k in enumerate(adata['input'].keys()):
            adata['input'][firstkey].obsm[f'scmaui-{k}_impute_score'] = np.zeros_like(out[j][1])
            for i, _ in enumerate(self.models):
                adata['input'][firstkey].obsm[f'scmaui-{k}_impute_score'] += adata['input'][firstkey].obsm[f'scmaui-{k}_impute_score_{i+1}']/N
        return adata

    def combine_modalities(self, adata):
        
        print('combine-previous', adata['input'])
        #X = np.concatenate([adata['input'][k].X for k in adata['input']], axis=1)
        
        iadata = ad.concat([adata['input'][k] for k in adata['input']], axis=1)
        firstkey = list(adata['input'].keys())[0]
        iadata.obs = adata['input'][firstkey].obs
        iadata.uns = adata['input'][firstkey].uns
        #oadata = ad.concat([adata['output'][k] for k in adata['output']], axis=1)
        #iadata.obsm['output'] = oadata.X
        #iadata.obsm['output_var'] = oadata.var
        print('combine-output', iadata)
        return iadata
        #for k in adata['input']:
        #     adata['input'][firstkey].obsm['input_'+k] = adata['input'][k].X
        #     adata['input'][firstkey].obsm['input_mask_'+k] = adata['input'][k].obsm['mask']
        #     adata['input'][firstkey].uns['input_features_'+k] = adata['input'][k].var
        #     adata['input'][firstkey].obsm['output_'+k] = adata['output'][k].X
        #     adata['input'][firstkey].obsm['output_mask_'+k] = adata['output'][k].obsm['mask']
        #     adata['input'][firstkey].uns['output_features_'+k] = adata['output'][k].var
        #return adata

#class BatchEnsembleVAE(EnsembleVAE):
#    def __init__(self, name, params, repeats, output, overwrite, feature_fraction=1., batchnames=[],
#                 adversarial=True, conditional=False, ):
#        super().__init__(params=params,
#                         repeats=repeats,
#                         output=output,
#                         overwrite=overwrite,
#                         name=name,
#                         feature_fraction=feature_fraction)
#        self.batchnames = batchnames
#        #self.name = name
#        self.adversarial = True if 'scmaui' in name else False
#        self.conditional = True if name.startswith('cond') else False
#        print(f'using {self.name}')
#
#    def _get_dataset_truebatchlabels(self, adata, batch_size=64):
#        """ used without dummy labels"""
#        tf_x = super()._get_dataset_truebatchlabels(adata, batch_size=batch_size)
#        return tf_x
#
#    def _get_dataset_dummybatchlabels(self, adata, batch_size=64):
#        """ used with dummy labels"""
#        tf_x = super()._get_dataset_dummybatchlabels(adata, batch_size=batch_size)
#        return tf_x
#

