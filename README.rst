========
BAVARIA
========

BAVARIA is python package that implements a
Batch-adversarial Variational auto-encoder with Negative Multinomial reconstruction loss for single-cell ATAC-seq analysis.

.. image:: scmaui_scheme.svg
  :width: 600

In particular, the model can be used to extract a latent feature representation of
a cell which can be used for downstream analysis tasks, including cell cluster,
cell identification, etc.
The package is freely available under a GNU Lesser General Public License v3 or later (LGPLv3+)

Installation
============

::

    pip install https://github.com/BIMSBbioinfo/scmaui/archive/v0.1.0.zip


Documentation
=============

BAVARIA offers a command line interface that fits an ensemble of BAVARIA models
given a raw count matrix (-data)
Subsequently, the model parameters and latent features
are stored in the output directory (-output)

::

   scmaui -data adata.h5ad \
         -names gtx \
         -output <outputdir> \
         -epochs 200 \
         -ensemble_size 10 \
         -nlatent 15 \
         -adversarial label1 label2 \
         -conditional covariate1 covariate2
 
Additional information on available hyper-parameters are available through

::

  scmaui -h

