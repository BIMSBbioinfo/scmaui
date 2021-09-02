mkdir -p data
# preprocessed scrna + scatac data (multimodal)
wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5 -O data/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_per_barcode_metrics.csv -O data/pbmc_granulocyte_sorted_3k_per_barcode_metrics.csv
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz -O data/pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz.tbi -O data/pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz.tbi
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/ -O data/pbmc_granulocyte_sorted_3k_singlecell.csv
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.tar.gz -O pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.tar.gz

#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_gex_molecule_info.h5 -O pbmc_granulocyte_sorted_3k_gex_molecule_info.h5
#wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_raw_feature_bc_matrix.h5 -O pbmc_granulocyte_sorted_3k_raw_feature_bc_matrix.h5

# fragment counts of the atac portion of the multimodal data


# scrna-seq
wget https://cf.10xgenomics.com/samples/cell-exp/3.0.2/5k_pbmc_v3/5k_pbmc_v3_filtered_feature_bc_matrix.h5 -O data/scrna_5k_pbmc_v3_filtered_feature_bc_matrix.h5

