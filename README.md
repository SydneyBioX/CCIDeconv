### CCIDeconv: Hierarchical model for deconvolution of subcellular cell-cell interactions in single-cell data

#### Description
A computational model for inferring subcellular cell–cell interactions (CCI) from high-resolution spatial transcriptomics data and single-cell transcriptomics.
This repository provides the model, analysis scripts, and example workflows to:

Deconvolute CCI to subcellular regions from traditional cell-resolved transcriptomics data using a hierarchical machine learning framework.

Reproduce analyses and generate figures from publicly available 10X Xenium datasets across multiple human tissues.

The model enables researchers to pinpoint where signaling is initiated and better interpret downstream pathway activity, offering insights into development, homeostasis, and disease progression.

<img src="figures/schematic_final.png" alt="Description" width="500">
Figure 1: Schematic of the subcellular CCI.

#### Model Architecture

The model distinguishes nuclear vs cytoplasmic CCI signals using a hierarchical neural network. Inputs are communication scores; outputs are deconvoluted CCI scores.

![Model Architecture](figures/Model_outline.png)

#### Installation / Requirements

To run the model on example data or your own dataset:
1. Clone the repository and install python dependencies
```bash
git clone https://github.com/SydneyBioX/CCIDeconv.git
cd CCIDeconv
pip install -r requirements.txt
```
Additional R Dependencies: If you plan on running the downstream analysis scripts and reproducing the figures in the manuscript, you may need to manually install additional R packages.
Depending on your environment, additional packages or dependencies may need to be installed manually.

2. Run the model on your own data or with the example data. Model weights can be downloaded from https://doi.org/10.5281/zenodo.20518900 

``` bash
 cd CCIDeconv
 python evaluate.py \
     --data_path example/example_train_data.csv \
     --categorical_columns lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane \
     --exclude_columns cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor\
     --test_data_path example/example_test_data.csv
```
3. (Optional) Explore the notebook:

If you prefer an interactive workflow, you can open example/example_run.ipynb in Jupyter Notebook. It reproduces the same steps as evaluate.py and shows example outputs.

```{r session-info}
sessionInfo()
```## Session Information

```text
R version 4.5.1 (2025-06-13)
Platform: aarch64-apple-darwin20
Running under: macOS Tahoe 26.7.2

Matrix products: default
BLAS:   /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib 
LAPACK: /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.1

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

time zone: Australia/Sydney
tzcode source: internal

attached base packages:
[1] grid      stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] lubridate_1.9.5        forcats_1.0.1          stringr_1.6.0          dplyr_1.2.1            purrr_1.2.2            readr_2.2.0           
 [7] tidyr_1.3.2            tibble_3.3.1           tidyverse_2.0.0        ggbiplot_0.6.5         patchwork_1.3.2        enrichplot_1.30.5     
[13] clusterProfiler_4.18.4 ggpubr_1.0.0           msigdbr_26.1.1         fgsea_1.36.2           ComplexHeatmap_2.26.1  cowplot_1.2.0         
[19] pheatmap_1.0.13        ggvenn_0.1.19          ggplot2_4.0.3          scales_1.4.0           circlize_0.4.18        data.table_1.18.6.1   

loaded via a namespace (and not attached):
  [1] splines_4.5.1               ggplotify_0.1.4             R.oo_1.27.1                 polyclip_1.10-7             lifecycle_1.0.5            
  [6] rstatix_1.1.0               doParallel_1.0.17           rprojroot_2.1.1             globals_0.20.0              lattice_0.22-7             
 [11] MASS_7.3-65                 backports_1.5.1             magrittr_2.0.5              rmarkdown_2.32              yaml_2.3.12                
 [16] otel_0.2.0                  ggtangle_0.1.4              reticulate_1.47.0           DBI_1.3.0                   RColorBrewer_1.1-3         
 [21] abind_1.4-8                 GenomicRanges_1.62.1        R.utils_2.13.0              BiocGenerics_0.56.0         yulab.utils_0.2.5          
 [26] tweenr_2.0.3                rappdirs_0.3.4              gdtools_0.5.1               IRanges_2.44.0              S4Vectors_0.48.1           
 [31] ggrepel_0.9.8               listenv_1.1.0               tidytree_0.4.8              parallelly_1.48.0           codetools_0.2-20           
 [36] DelayedArray_0.36.1         DOSE_4.4.0                  scuttle_1.20.0              ggforce_0.5.0               tidyselect_1.2.1           
 [41] shape_1.4.6.1               aplot_0.3.2                 farver_2.1.2                matrixStats_1.5.0           stats4_4.5.1               
 [46] Seqinfo_1.0.0               jsonlite_2.0.0              GetoptLong_1.1.1            Formula_1.2-6               iterators_1.0.14           
 [51] systemfonts_1.3.2           foreach_1.5.2               tools_4.5.1                 ggnewscale_0.5.2            treeio_1.34.0              
 [56] Rcpp_1.1.2                  glue_1.8.1                  SparseArray_1.10.10         xfun_0.61                   here_1.0.2                 
 [61] qvalue_2.42.0               MatrixGenerics_1.22.0       withr_3.0.3                 fastmap_1.2.0               digest_0.6.39              
 [66] timechange_0.4.0            R6_2.6.1                    gridGraphics_0.5-1          colorspace_2.1-4            GO.db_3.22.0               
 [71] anndata_0.8.0               RSQLite_3.53.3              R.methodsS3_1.8.2           generics_0.1.4              fontLiberation_0.1.0       
 [76] httr_1.4.9                  htmlwidgets_1.6.4           S4Arrays_1.10.1             scatterpie_0.2.6            pkgconfig_2.0.3            
 [81] gtable_0.3.6                blob_1.3.0                  S7_0.2.2                    SingleCellExperiment_1.32.0 XVector_0.50.0             
 [86] htmltools_0.5.9             fontBitstreamVera_0.1.1     carData_3.0-6               clue_0.3-68                 Biobase_2.70.0             
 [91] png_0.1-9                   ggfun_0.2.2                 knitr_1.52                  rstudioapi_0.19.0           tzdb_0.5.0                 
 [96] reshape2_1.4.5              rjson_0.2.23                nlme_3.1-171                cachem_1.1.0                GlobalOptions_0.1.4        
[101] parallel_4.5.1              AnnotationDbi_1.72.0        pillar_1.11.1               vctrs_0.7.3                 car_3.1-5                  
[106] tidydr_0.0.7                beachmat_2.26.0             cluster_2.1.8.3             evaluate_1.0.5              cli_3.6.6                  
[111] compiler_4.5.1              rlang_1.3.0                 crayon_1.5.3                ggsignif_0.6.4              labeling_0.4.3             
[116] plyr_1.8.9                  fs_2.1.0                    ggiraph_0.9.6               stringi_1.8.9               BiocParallel_1.44.0        
[121] assertthat_0.2.1            Biostrings_2.78.0           lazyeval_0.2.3              GOSemSim_2.36.0             fontquiver_0.2.1           
[126] Matrix_1.7-3                hms_1.1.4                   bit64_4.8.6                 future_1.76.0               KEGGREST_1.50.0            
[131] SummarizedExperiment_1.40.0 igraph_2.3.4                broom_1.0.13                memoise_2.0.1               ggtree_4.0.5               
[136] fastmatch_1.1-8             bit_4.6.0                   ape_5.8-1                   gson_0.2.2                 
```
