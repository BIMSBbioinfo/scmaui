"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mscmaui` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``scmaui.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``scmaui.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from scmaui import __version__
from scmaui.ensembles import EnsembleVAE
from scmaui.data import load_data
from scmaui.data import SCDataset
from scmaui.data import combine_modalities
from scmaui.utils import get_model_params
from scmaui.utils import get_variable_regions
import scanpy as sc
import logging


def main(args=None):

    parser = argparse.ArgumentParser('scmaui',
                                     description=f'Single-cell multi-omics integration using variational auto-encoders - v{__version__}')

    # input data
    parser.add_argument('-data', dest='data', type=str,
                        nargs='+', help="One or more h5ad datasets containing one input modality each. "
                                        "If no -outdata is provided the input data serves as output data for an auto-encoding model.",
                        required=True)
    parser.add_argument('-datanames', dest='datanames', nargs='*',
                        help='Associated names of the input modalities.')

    # optional output data
    parser.add_argument('-outdata', dest='outdata', type=str,
                        nargs='*', help="(optional) One or more h5ad datasets containing one output modality each."
                         "If outdata is not specified, then the input datasets serve as output data as well for an auto-encoder."
                         "Otherwise, cross-modality prediction can be trained using the separate output datasets.")
    parser.add_argument('-outdatanames', dest='outdatanames', nargs='*',
                        help='Associated names of the output modalities.')

    # reconstruction loss
    parser.add_argument('-loss', dest='loss', type=str, nargs='+',
                        choices=['mul', 'mse', 'binary', 'poisson',
                                                'negmul', 'negmul2', 'zinb',
                                                'dirmul',
                                                'mixgaussian', 'negbinom', 'gamma', 'mixgamma',
                                               ],
                        help="Available reconstruction losses.", required=True)

    # output directory
    parser.add_argument('-output', dest='output', type=str,
                        help="Output directory", required=True)


    # optional model parameters and hyper-parameters
    parser.add_argument('-nlatent', dest='nlatent', type=int, default=10,
                        help="Latent dimensions. Default: 10")

    parser.add_argument('-epochs', dest='epochs', type=int, default=100,
                        help="Number of epochs. Default: 100.")
    parser.add_argument('-ensemble_size', dest='ensemble_size', type=int, default=1,
                        help="Ensemble size. "
                             "Default: 1.")
    parser.add_argument('-batch_size', dest='batch_size', type=int,
                        default=128,
                        help='Batch size. Default: 128.')
    #parser.add_argument('-overwrite', dest='overwrite',
    #                    action='store_true', default=False)
    parser.add_argument('-evaluate', dest='evaluate',
                        action='store_true', default=False,
                        help="Whether to evaluate an existing model. Default: False.")
    parser.add_argument('-skip_outliers', dest='skip_outliers',
                        action='store_true', default=False,
                        help='Skip models with outlier loss (aka poor local optimum)')

    parser.add_argument('-nhidden_e', dest='nhidden_e', type=int,
                        default=32,
                        help="Number of neurons per encoder layer. "
                             "This number is constant across the encoder. Default: 512.")
    parser.add_argument('-nlayers_e', dest='nlayers_e', type=int,
                        default=2,
                        help="Number of residual blocks for the encoder. Default: 20.")
    parser.add_argument('-nhidden_d', dest='nhidden_d', type=int,
                        default=16,
                        help="Number of neurons per decoder hidden layer. "
                             "Usually it is important to set nhidden_d as well as nlatent to relatively small values. "
                             " Too high numbers for these parameters degrades the quality of the latent features. "
                             "Default: 16.")
    parser.add_argument('-nlayers_d', dest='nlayers_d', type=int,
                        default=1,
                        help="Number of decoder hidden layers. Default: 1.")
    parser.add_argument('-inputdropout', dest='inputdropout', type=float,
                        default=0.0,
                        help="Dropout rate applied at the inital layer (e.g. input accessibility profile). Default=0.15")
    parser.add_argument("-hidden_e_dropout", dest="hidden_e_dropout", type=float,
                        default=0.0,
                        help="Dropout applied in each residual block of the encoder. Default=0.3")
    parser.add_argument("-hidden_d_dropout", dest="hidden_d_dropout", type=float,
                        default=0.0,
                        help="Dropout applied after each decoder hidden layer. Default=0.3")
    parser.add_argument("-feature_fraction", dest="feature_fraction", type=float, default=1.,
                        help="(currently unused) Whether to use a random subset of features. feature_fraction determines the proportion of features to use. Default=1.")

    parser.add_argument("-adversarial", dest="adversarial", type=str, nargs='+', default=[],
                        help="Adversarial labels. This should be a sample/cell-annotation column in one of the input dataset. "
                         "Adversarial model uses a mean-squared error against numerical labels and a categorical cross-entropy "
                         "error against categorical features.")

    parser.add_argument("-conditional", dest="conditional", type=str, nargs='+', default=[],
                        help="Conditional covariates. This should be a sample/cell-annotation column in one of the input dataset. "
                        "Categorical or numerical covariates are supported.")

    # for testing other model architectures
    parser.add_argument("-modelname", dest="modelname", type=str, default='vae', choices=[
                                                                                          #'scmaui-0', 'scmaui',
                                                                                          #'bcvae', 'bcvae2',
                                                                                           'vae',
                                                                                          # 'cond-vae',
                                                                                          # 'regout-vae',
                                                                                          # 'vae-ml',
                                                                                         ],
                        help="Model architectures. Default: vae")
    parser.add_argument('-resolution', dest='resolution', type=float, default=1.,
                        help="Resolution for Louvain clustering analysis.")


    parser.add_argument('-nhidden_b', dest='nhidden_b', type=int,
                        default=32,
                        help="Number of hidden neurons for batch predictor. "
                             "Default: 32.")
    parser.add_argument('-nmixcomp', dest='nmixcomp', type=int,
                        default=2,
                        help="Number of components for mixture models. Default: 2.")

    args = parser.parse_args()

    data = args.data
    datanames = args.datanames
    outdata = args.outdata
    outdatanames = args.outdatanames

    # load the dataset
    adatas = load_data(data, datanames, outdata, outdatanames)

    dataset = SCDataset(adatas, args.adversarial, args.conditional)
    print(dataset)

    params = get_model_params(dataset, args=args)

    metamodel = EnsembleVAE(params,
                            args.ensemble_size, 
                            args.modelname,
                            args.feature_fraction,)

    if not args.evaluate:
        metamodel.fit(dataset,
                      epochs=args.epochs,
                      batch_size=args.batch_size)
        metamodel.save(args.output)
    else:
        metamodel.load(args.output)

    latent, latent_list = metamodel.encode(dataset, skip_outliers=args.skip_outliers)

    for i, ldf in enumerate(latent_list):
        ldf.to_csv(os.path.join(args.output, f'model_{i+1}', 'latent.csv'))
    latent.to_csv(os.path.join(args.output, 'latent.csv'))

    oadata = combine_modalities(dataset.adata['input'])
    adata = oadata
    adata.obsm['scmaui-ensemble'] = latent.values
    for i, lat in enumerate(latent_list):
        adata.obsm[f'scmaui-{i+1}'] = lat.values

    sc.pp.neighbors(adata, n_neighbors=15, use_rep="scmaui-ensemble")
    sc.tl.louvain(adata, resolution=args.resolution)
    sc.tl.umap(adata)

    if 'batchnames' not in params:
        params['batchnames'] = None
    adata.write(os.path.join(args.output, "analysis.h5ad"), compression='gzip')
    print('saved to ' + os.path.join(args.output, "analysis.h5ad"))
