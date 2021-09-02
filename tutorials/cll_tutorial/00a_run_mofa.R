library(MOFA2)
library(MOFAdata)
library(data.table)
library(ggplot2)
library(tidyverse)
library(umap)

utils::data("CLL_data")       
lapply(CLL_data,dim)
MOFAobject <- create_mofa(CLL_data)

model_opts <- get_default_model_options(MOFAobject)
model_opts$num_factors <- 10

model_opts

data_opts <- get_default_data_options(MOFAobject)
data_opts

train_opts <- get_default_training_options(MOFAobject)
train_opts$convergence_mode <- "slow"
train_opts$seed <- 42

train_opts

MOFAobject <- prepare_mofa(MOFAobject,
  data_options = data_opts,
  model_options = model_opts,
  training_options = train_opts
)

MOFAobject <- run_mofa(MOFAobject, outfile="./MOFA2_CLL.hdf5")

write.table(as.data.frame(MOFAobject@expectations[[1]][[1]]), 
            "mofa_latent_features.csv", sep=",", quote=F)
