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
from scipy.sparse import vstack


class EnsembleVAE:
    """Ensemble of VAEs

    This class maintains an ensemble of VAE models
    and provides an interface to fit, save, load and evaluate the models.

    Parameters
    ----------
    params : dict
        Model parameters
    ensemble_size : int
        Ensemble size. Default=1
    model : str or None
        Model name. Currently only 'vae' is available.
    """

    def __init__(self, params, ensemble_size=1, model=None):
        self.ensemble_size = ensemble_size
        self.models = []

        self.space = params
        # self.feature_fraction = max(min(feature_fraction, 1.), 0.)
        if model is None:
            model = "vae"
        self.model = model
        print(f"using {self.model}")

    def _create(self, model, space):
        """Create VAE model"""
        if model == "vae":
            model = VAE.create(space, create_encoder_base, create_decoder_base)
        else:
            raise ValueError(f"Unknown model: {model}")
        return model

    def _load(self, path):
        """Reload VAE model"""
        if self.model == "vae":
            model = VAE.load(path)
        else:
            raise ValueError(f"Unknown model: {model}")
        return model

    def summary(self):
        assert len(self.models) > 0, "No models available yet."
        assert len(self.models) > 0, "No models available yet."
        self.models[0].summary()

    def save(self, path):
        """save the ensemble"""
        for r, model in enumerate(self.models):
            subpath = os.path.join(path, f"model_{r+1}")
            os.makedirs(subpath, exist_ok=True)
            model.save(os.path.join(subpath, "network", "vae.h5"))
            model.save_weights(os.path.join(subpath, "model", "vae.h5"))

    def load(self, path, learning_rate=0.001):
        """load a previously trained ensemble"""
        for r in range(self.ensemble_size):
            subpath = os.path.join(path, f"model_{r+1}")
            if (not os.path.exists(os.path.join(subpath, "network"))) or (
                not os.path.exists(os.path.join(subpath, "model"))
            ):
                print(f"no model in {subpath}")
                return

            model = self._load(os.path.join(subpath, "network", "vae.h5"))
            model.load_weights(os.path.join(subpath, "model", "vae.h5"))
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=learning_rate, amsgrad=True
                )
            )
            self.models.append(model)
        print(f"re-loaded models from {subpath}")

    def fit(
        self,
        dataset,
        shuffle=True,
        batch_size=64,
        learning_rate=0.001,
        epochs=1,
        validation_split=0.15,
    ):
        """Fit an ensemble of VAE models.

        Parameters
        ----------
        dataset : SCDataset
            An SCDataset.
        shuffle : bool
            Whether to shuffle the dataset during training. Default: True.
        batch_size : int
            Batch size. Default: 64
        epochs : int
            Number of epochs. Default: 1
        validation_split : float
            Validation split. Default: 0.15

        Returns
        -------
        list(history)
            Training loss history.
        """
        space = self.space
        histories = []

        for r in range(self.ensemble_size):
            tf_X, tf_X_test = dataset.training_data(
                batch_size=batch_size, validation_split=0.15
            )

            print(f"Run model {r+1}")
            space.update(dataset.shapes())
            model = self._create(self.model, space)

            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=learning_rate, amsgrad=True
                ),
            )
            history = model.fit(
                tf_X,
                epochs=epochs,
                validation_data=(tf_X_test,),
                callbacks=CSVLogger("train_summary.csv"),
                verbose=2,
            )
            histories.append(history)
            self.models.append(model)
        return histories

    def encode_full(self, dataset, batch_size=64):
        tf_X = dataset.embedding_data(batch_size=batch_size)
        dfs = []
        for i, model in enumerate(self.models):

            out = model.encoder_mean.predict(tf_X)
            df = pd.DataFrame(
                out,
                index=dataset.adata["input"][0].obs.index,
                columns=[f"D{i}-{n}" for n in range(out.shape[1])],
            )
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df, dfs

    def encode(self, dataset, batch_size=64):
        """Latent feature encoding

        This method determines the latent feature (e.g. the last layer of the encoder).

        Parameters
        ----------
        dataset : SCDataset
            An SCDataset.
        batch_size : int
            Batch size to use for evaluating the imputation.

        Returns
        -------
        tuple(np.array, list(np.array)):
            The first element of the returned tuple contains the combined (stacked) encoding
            across the ensemble of VAEs. Its dimension is cells by ensemble_size x latent dims.
            The second element of the returned tuple contains the individual encodings per model.
            Each encoding is of dimension cells by latent dims.
        """
        return self.encode_full(dataset, batch_size)

    def impute(self, dataset, batch_size=64):
        """Imputation

        This method performs imputation on the original input features.
        If more than one model is available in the ensemble, the average prediction
        across all individual models is returned.

        Parameters
        ----------
        dataset : SCDataset
            An SCDataset.
        batch_size : int
            Batch size to use for evaluating the imputation.

        Returns
        -------
        list(np.array)
            A list of numpy arrays containing the imputed feature.
            Each entry in the list reflects predictions for a modality
            with dimensions $C \times F_i$ where C denotes the
            number of cells and
            $F_i$ denotes the input feature dimensionality of the ith modality.
        """
        tf_X = dataset.imputation_data(batch_size=batch_size)

        output = [np.zeros((dataset.size(), m)) for m in dataset.shapes()["outputdims"]]

        for i, model in enumerate(self.models):
            out = model.predict(tf_X)
            if not isinstance(out, (list, tuple)):
                out = [out]
            for i, o in enumerate(out):
                output[i] += o / len(self.models)

        return output

    def explain(self, dataset, cellids, baselineids=None, modelid=0):
        """Explain latent features based on input features.

        This method makes use of integrated gradients in order to
        attibute the input feature importances on the latent features.
        Integrated gradients evalutes the path integral between a baseline
        input and the target datapoint.
        This method evaluates the average feature attribution across multiple
        cells/samples assuming that the target cells/samples are inherently similar
        to one another (e.g. cells within one cluster).
        As baseline, a random selection of negative cells/samples are chosen (as opposed
        to an all-zeros input). It is possible to specify the baseline cells/samples
        through the baselineids argument.


        Parameters
        ----------
        dataset : SCDataset
            An SCDataset object.
        cellids : list(str)
            A list of cell/sample ids for which the feature attribution should be determined.
            cellids should be indices in the dataset (e.g. contained in adata.obs.index).
        baselineids : list(str) or None
            A list of negative cell/sample ids used as contrast for the feature attribution.
            If None, negative cell ids are randomly selected from the remaining cells in the dataset.
            Default: None.
        modelid : int
            Model index referencing the model in the ensemble. Feature explanation is only determined
            based on a single-model in the ensemble. Default: 0

        Returns
        -------
        list(np.array)
            A list of numpy arrays containing the feature attributions.
            Each entry in the list reflects attributions for a modality
            with dimensions $F_i \times L$ where
            $F_i$ denotes the input feature dimensionality of the ith modality and
            L the latent feature dimensions.
        """
        # get positive and baseline examples
        posdata = dataset.subset(cellids)
        if baselineids is None:
            negall = dataset.exclude(cellids)
            negdata = negall.sample(len(cellids))
        else:
            negdata = dataset.subset(baselineids)
            negdata = negdata.sample(len(cellids))

        posdata = posdata.embedding_data(as_tf_data=False)
        negdata = negdata.embedding_data(as_tf_data=False)

        S = 50  # integration steps
        # prepare new dataset for the integration
        alpha = np.linspace(0, 1, S).reshape(-1, 1, 1)

        X = [
            tf.convert_to_tensor(
                (
                    np.expand_dims(np.nan_to_num(p.toarray()), 0) * alpha
                    + np.expand_dims(np.nan_to_num(n.toarray()), 0) * (1 - alpha)
                ).reshape(-1, n.shape[-1]),
                dtype=tf.float32,
            )
            for p, n in zip(posdata[0][0], negdata[0][0])
        ]

        deltaX = [
            tf.convert_to_tensor(
                (np.nan_to_num(p.toarray()) - np.nan_to_num(n.toarray())).reshape(
                    -1, n.shape[-1]
                ),
                dtype=tf.float32,
            )
            for p, n in zip(posdata[0][0], negdata[0][0])
        ]

        mask = [
            tf.convert_to_tensor(
                vstack([m1.multiply(m2)] * S).toarray(), dtype=tf.float32
            )
            for m1, m2 in zip(posdata[0][1], negdata[0][1])
        ]
        cond = [
            tf.convert_to_tensor(np.kron(c, np.ones((S, 1))), dtype=tf.float32)
            for c in posdata[1]
        ]
        adv = [
            tf.convert_to_tensor(np.kron(a, np.ones((S, 1))), dtype=tf.float32)
            for a in posdata[1]
        ]

        ds = [X, mask], adv, cond

        def compute_gradients(model, X, mask, cond, fidx):
            with tf.GradientTape() as tape:
                tape.watch(X)
                latent = model([X, mask, cond])
                f = tf.math.reduce_sum(latent[:, fidx])
            G = tape.gradient(f, X)
            return [g * m for g, m in zip(G, mask)]

        model = self.models[modelid].encoder_mean.model
        fidx = 0
        ig_total = [
            np.zeros((d, self.space["nlatent"])) for d in self.space["inputdims"]
        ]
        for fidx in range(self.space["nlatent"]):
            grads = compute_gradients(model, X, mask, cond, fidx)
            # grads dims are equal to input dims
            igrads = [
                dx * 1 / S * tf.reduce_sum(tf.reshape(g, (S, -1, n.shape[-1])), axis=0)
                for m, dx, n, g in zip(mask, deltaX, negdata[0][0], grads)
            ]

            mean_igrads = [
                tf.reduce_sum(ig, axis=0)
                / tf.reduce_sum(tf.convert_to_tensor(m.toarray(), dtype=tf.float32))
                for ig, m in zip(igrads, posdata[0][1])
            ]
            for i, ig in enumerate(mean_igrads):
                ig_total[i][:, fidx] = ig.numpy()

        return ig_total
