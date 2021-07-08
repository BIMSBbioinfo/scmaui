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
#from scmaui.ensembles import EnsembleVAE
from scmaui.ensembles import BatchEnsembleVAE
from scmaui.data import load_data
from scmaui.data import load_batch_labels
from scmaui.data import one_hot_encode_batches
from scmaui.utils import resnet_vae_params
from scmaui.utils import resnet_vae_batch_params
from scmaui.utils import get_variable_regions
import scanpy as sc
import logging


def main(args=None):

    parser = argparse.ArgumentParser('scmaui',
                                     description=f'Negative multinomial variational auto-encoders - v{__version__}')

    parser.add_argument('-data', dest='data', type=str,
                        nargs='+', help="One or more h5ad datasets each one containing one input modality. If no -outdata is provided the input data serves as output data for an auto-encoding model.", required=True)
    parser.add_argument('-datanames', dest='datanames', nargs='*',
                        help='Names of the modalities.')
    parser.add_argument('-outdata', dest='outdata', type=str,
                        nargs='*', help="One or more h5ad datasets each one containing one output modality.")
    parser.add_argument('-outdatanames', dest='outdatanames', nargs='*',
                        help='Names of the output modalities.')

    parser.add_argument('-loss', dest='loss', type=str, nargs='+',
                        default='mse', choices=['mul', 'mse', 'binary', 'binom',
                                                'negmul', 'negmul2', 'zinb', 'mixgaussian', 'negbinom', 'gamma', 'mixgamma',
                                               ],
                        help="Loss associated with each modality.")
    parser.add_argument('-output', dest='output', type=str,
                        help="Output directory", required=True)


    parser.add_argument('-nlatent', dest='nlatent', type=int, default=10,
                        help="Latent dimensions. Default: 10")

    parser.add_argument('-epochs', dest='epochs', type=int, default=100,
                        help="Number of epochs. Default: 100.")
    parser.add_argument('-nrepeat', '-nmodels', '-ensemblesize', dest='nrepeat', type=int, default=1,
                        help="Number of repeatedly fitted models. "
                             "Default: 1.")
    parser.add_argument('-batch_size', dest='batch_size', type=int,
                        default=128,
                        help='Batch size. Default: 128.')
    parser.add_argument('-overwrite', dest='overwrite',
                        action='store_true', default=False)
    parser.add_argument('-skip_outliers', dest='skip_outliers',
                        action='store_true', default=False,
                        help='Skip models with outlier loss (aka poor local optimum)')

    parser.add_argument('-nlayers_d', dest='nlayers_d', type=int,
                        default=1,
                        help="Number of decoder hidden layers. Default: 1.")
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
                        help="Whether to use a random subset of features. feature_fraction determines the proportion of features to use. Default=1.")
    parser.add_argument("-batches", dest="batches", type=str, default=None,
                        help="Table in tsv format defining the cell batches. "
                             "The first columns should represent the barcode "
                             "while the remaining columns represent the batches as categorical labels.")

    parser.add_argument("-batchnames", dest="batchnames", type=str, nargs='+', default=[],
                        help="Batch names in the anndata dataset. ")
    parser.add_argument("-modelname", dest="modelname", type=str, default='vae', choices=[
                                                                                          'scmaui-0', 'scmaui',
                                                                                          'bcvae', 'bcvae2',
                                                                                           'vae',
                                                                                           'cond-vae',
                                                                                           'regout-vae',
                                                                                           'vae-ml',
                                                                                         ],
                        help="Model name for batch correction. Default: vae")
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
    outdatanames = args.outdatanames
    outdata = args.outdata
    batches = args.batches

    batchnames = ['basebatch']
    batchnames += args.batchnames
    # load the dataset
    adatas = load_data(data, outdata)
    #adatas = dict()
    #adatas['input'] = load_data(data, datanames)
    #if outdata is None:
    #    adatas['output'] = adatas['input']
    #else:
    #    adatas['output'] = load_data(outdata, outdatanames)

    params = resnet_vae_params(args)
    params['inputmodality'] = list(adatas['input'].keys())
    params['outputmodality'] = list(adatas['output'].keys())

    adatas['input'][list(adatas['input'].keys())[0]].obs.loc[:,'basebatch'] = 'basebatch'

    adatas['input'] = one_hot_encode_batches(adatas['input'], batchnames)
    params.update(resnet_vae_batch_params(adatas['input'], batchnames))

    metamodel = BatchEnsembleVAE(args.modelname, params,
                            args.nrepeat, args.output,
                            args.overwrite,
                            args.feature_fraction,
                            params['batchnames'])

    metamodel.fit(adatas, epochs=args.epochs, batch_size=args.batch_size)

    print(adatas)
    oadata = metamodel.combine_modalities(adatas)
    oadata = metamodel.encode(adatas, oadata, skip_outliers=args.skip_outliers)
    #adata = metamodel.impute(adatas)

    #firstkey = list(oadata.keys())[0]
    adata = oadata
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="scmaui-ensemble")
    sc.tl.louvain(adata, resolution=args.resolution)
    sc.tl.umap(adata)

    if 'batchnames' not in params:
        params['batchnames'] = None
    #adata = get_variable_regions(adata, batches=params['batchnames'])
    adata.write(os.path.join(args.output, "analysis.h5ad"), compression='gzip')
    print(adata)
    print('saved to ' + os.path.join(args.output, "analysis.h5ad"))
