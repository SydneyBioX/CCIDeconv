suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

rds_dir <- "/Users/rjay0256/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney/PhD projects/main/Cell Cell Communication/server/cell_cell_commuication/data/processed/cellchat_ablated_significant"
training_data_path <- "data/training_data/training_data.csv"

# Load training data to get model genes
train_df <- fread(training_data_path)
model_ligands <- unique(train_df$ligand[!is.na(train_df$ligand)])
model_receptors <- unique(train_df$receptor[!is.na(train_df$receptor)])
all_model_genes <- unique(c(model_ligands, model_receptors))
cat(sprintf("Loaded %d unique CCIDeconv model genes from training_data.csv\n", length(all_model_genes)))

# List of the 9 datasets
dataset_files <- list(
  "spe_list_brainad_cellchat" = "spe_list_brainad_cellchat.rds",
  "spe_list_braingl_cellchat" = "spe_list_braingl_cellchat.rds",
  "spe_list_breast_cancer_cellchat" = "spe_list_breast_cancer_cellchat.rds",
  "spe_list_lung_cellchat" = "spe_list_lung_cellchat.rds",
  "spe_list_lung_high_res_cellchat" = "spe_list_lung_high_res_cellchat.rds",
  "spe_list_lymphnode_cellchat" = "spe_list_lymphnode_cellchat.rds",
  "spe_list_ovarian_cellchat" = "spe_list_ovarian_cellchat.rds",
  "spe_list_panc_cellchat" = "spe_list_panc_cellchat.rds",
  "spe_list_skin_cellchat" = "spe_list_skin_cellchat.rds"
)

dataset_labels <- list(
  "spe_list_brainad_cellchat" = "Brain (AD)",
  "spe_list_braingl_cellchat" = "Brain (GL)",
  "spe_list_breast_cancer_cellchat" = "Breast Cancer",
  "spe_list_lung_cellchat" = "Lung",
  "spe_list_lung_high_res_cellchat" = "Lung (High-Res)",
  "spe_list_lymphnode_cellchat" = "Lymph Node",
  "spe_list_ovarian_cellchat" = "Ovarian",
  "spe_list_panc_cellchat" = "Pancreas",
  "spe_list_skin_cellchat" = "Skin"
)

summary_rows <- list()
gene_rows <- list()
cell_subsample_rows <- list()

set.seed(42)

for (sample_key in names(dataset_files)) {
  f <- file.path(rds_dir, dataset_files[[sample_key]])
  label <- dataset_labels[[sample_key]]
  cat(sprintf("\n>>> Processing [%s] (%s) from %s <<<\n", label, sample_key, basename(f)))
  
  t0 <- proc.time()
  obj <- readRDS(f)
  cat(sprintf("  Loaded in %.1f seconds\n", (proc.time() - t0)[3]))
  
  cc_cyt <- obj[["cyt"]]
  cc_nuc <- obj[["nuc"]]
  cc_cell <- obj[["cell"]]
  
  mat_cyt <- cc_cyt@data
  mat_nuc <- cc_nuc@data
  mat_cell <- if (!is.null(cc_cell)) cc_cell@data else NULL
  
  n_genes_panel <- nrow(mat_cyt)
  n_cells_cyt <- ncol(mat_cyt)
  n_cells_nuc <- ncol(mat_nuc)
  n_cells_cell <- if (!is.null(mat_cell)) ncol(mat_cell) else NA_integer_
  
  cat(sprintf("  Genes: %d | Cells: Cell=%s, Cyt=%d, Nuc=%d\n", 
              n_genes_panel, as.character(n_cells_cell), n_cells_cyt, n_cells_nuc))
  
  # 1. Cell-level detected genes and total expression
  det_cyt <- Matrix::colSums(mat_cyt > 0)
  det_nuc <- Matrix::colSums(mat_nuc > 0)
  
  tot_cyt <- Matrix::colSums(mat_cyt)
  tot_nuc <- Matrix::colSums(mat_nuc)
  
  zero_frac_cyt <- 1 - (det_cyt / n_genes_panel)
  zero_frac_nuc <- 1 - (det_nuc / n_genes_panel)
  
  # Subsample cells for plotting (up to 3000 cells per compartment)
  n_sub_cyt <- min(3000, length(det_cyt))
  sub_idx_cyt <- sample(seq_along(det_cyt), n_sub_cyt)
  n_sub_nuc <- min(3000, length(det_nuc))
  sub_idx_nuc <- sample(seq_along(det_nuc), n_sub_nuc)
  
  df_sub_cyt <- data.frame(
    sample_key = sample_key,
    dataset = label,
    compartment = "Cytoplasm",
    detected_genes = det_cyt[sub_idx_cyt],
    zero_fraction = zero_frac_cyt[sub_idx_cyt],
    total_expression = tot_cyt[sub_idx_cyt]
  )
  df_sub_nuc <- data.frame(
    sample_key = sample_key,
    dataset = label,
    compartment = "Nucleus",
    detected_genes = det_nuc[sub_idx_nuc],
    zero_fraction = zero_frac_nuc[sub_idx_nuc],
    total_expression = tot_nuc[sub_idx_nuc]
  )
  cell_subsample_rows[[paste0(sample_key, "_cyt")]] <- df_sub_cyt
  cell_subsample_rows[[paste0(sample_key, "_nuc")]] <- df_sub_nuc
  
  # Matched cells for cell-level N/C ratio
  shared_cells <- intersect(colnames(mat_cyt), colnames(mat_nuc))
  n_shared_cells <- length(shared_cells)
  cat(sprintf("  Shared cells with both cyt and nuc: %d\n", n_shared_cells))
  
  if (n_shared_cells > 0) {
    # cell-level N/C of total expression
    tot_cyt_shared <- tot_cyt[shared_cells]
    tot_nuc_shared <- tot_nuc[shared_cells]
    cell_nc_ratio <- tot_nuc_shared / (tot_cyt_shared + 1e-4)
    cell_nc_median <- median(cell_nc_ratio)
    cell_nc_iqr <- IQR(cell_nc_ratio)
  } else {
    cell_nc_median <- NA_real_
    cell_nc_iqr <- NA_real_
  }
  
  # 2. Gene-level metrics
  # Detection rate per gene across cells
  gene_det_cells_cyt <- Matrix::rowSums(mat_cyt > 0)
  gene_det_cells_nuc <- Matrix::rowSums(mat_nuc > 0)
  
  gene_det_rate_cyt <- gene_det_cells_cyt / n_cells_cyt
  gene_det_rate_nuc <- gene_det_cells_nuc / n_cells_nuc
  
  gene_mean_expr_cyt <- Matrix::rowMeans(mat_cyt)
  gene_mean_expr_nuc <- Matrix::rowMeans(mat_nuc)
  
  # Gene N/C ratio using mean expression with pseudocount 1e-4
  gene_nc_ratio <- (gene_mean_expr_nuc + 1e-4) / (gene_mean_expr_cyt + 1e-4)
  
  genes_all <- rownames(mat_cyt)
  is_model_gene <- genes_all %in% all_model_genes
  
  df_gene_ds <- data.frame(
    sample_key = sample_key,
    dataset = label,
    gene = genes_all,
    is_model_gene = is_model_gene,
    det_cells_cyt = as.numeric(gene_det_cells_cyt),
    det_cells_nuc = as.numeric(gene_det_cells_nuc),
    det_rate_cyt = as.numeric(gene_det_rate_cyt),
    det_rate_nuc = as.numeric(gene_det_rate_nuc),
    dropout_rate_cyt = 1 - as.numeric(gene_det_rate_cyt),
    dropout_rate_nuc = 1 - as.numeric(gene_det_rate_nuc),
    mean_expr_cyt = as.numeric(gene_mean_expr_cyt),
    mean_expr_nuc = as.numeric(gene_mean_expr_nuc),
    nc_ratio = as.numeric(gene_nc_ratio),
    log2_nc_ratio = log2(as.numeric(gene_nc_ratio))
  )
  gene_rows[[sample_key]] <- df_gene_ds
  
  # 3. Overall matrix dropout (fraction of zero entries)
  nnz_cyt <- Matrix::nnzero(mat_cyt)
  tot_entries_cyt <- as.numeric(n_genes_panel) * as.numeric(n_cells_cyt)
  matrix_dropout_cyt <- 1 - (nnz_cyt / tot_entries_cyt)
  
  nnz_nuc <- Matrix::nnzero(mat_nuc)
  tot_entries_nuc <- as.numeric(n_genes_panel) * as.numeric(n_cells_nuc)
  matrix_dropout_nuc <- 1 - (nnz_nuc / tot_entries_nuc)
  
  # 4. Model gene coverage
  n_model_genes_panel <- sum(is_model_gene)
  # detected in at least 1 cell
  n_model_genes_det_cyt_1cell <- sum(is_model_gene & gene_det_cells_cyt > 0)
  n_model_genes_det_nuc_1cell <- sum(is_model_gene & gene_det_cells_nuc > 0)
  # detected in >= 1% of cells
  n_model_genes_det_cyt_1pct <- sum(is_model_gene & gene_det_rate_cyt >= 0.01)
  n_model_genes_det_nuc_1pct <- sum(is_model_gene & gene_det_rate_nuc >= 0.01)
  
  model_det_rate_cyt_median <- if (n_model_genes_panel > 0) median(gene_det_rate_cyt[is_model_gene]) else NA_real_
  model_det_rate_nuc_median <- if (n_model_genes_panel > 0) median(gene_det_rate_nuc[is_model_gene]) else NA_real_
  
  # 5. Compile dataset summary
  row <- data.frame(
    sample_key = sample_key,
    dataset = label,
    n_genes_panel = n_genes_panel,
    n_cells_cell = n_cells_cell,
    n_cells_cyt = n_cells_cyt,
    n_cells_nuc = n_cells_nuc,
    n_shared_cells = n_shared_cells,
    nuc_to_cyt_cell_ratio = n_cells_nuc / n_cells_cyt,
    
    # Sparsity: Detected genes per cell
    cyt_detected_median = median(det_cyt),
    cyt_detected_mean = mean(det_cyt),
    cyt_detected_iqr = IQR(det_cyt),
    nuc_detected_median = median(det_nuc),
    nuc_detected_mean = mean(det_nuc),
    nuc_detected_iqr = IQR(det_nuc),
    detected_ratio_nuc_to_cyt = median(det_nuc) / median(det_cyt),
    detected_diff_cyt_minus_nuc = median(det_cyt) - median(det_nuc),
    
    # Sparsity: Zero fraction per cell
    cyt_zero_frac_median = median(zero_frac_cyt),
    cyt_zero_frac_mean = mean(zero_frac_cyt),
    cyt_zero_frac_iqr = IQR(zero_frac_cyt),
    nuc_zero_frac_median = median(zero_frac_nuc),
    nuc_zero_frac_mean = mean(zero_frac_nuc),
    nuc_zero_frac_iqr = IQR(zero_frac_nuc),
    zero_frac_diff_nuc_minus_cyt = median(zero_frac_nuc) - median(zero_frac_cyt),
    
    # Expression sum per cell
    cyt_tot_expr_median = median(tot_cyt),
    cyt_tot_expr_mean = mean(tot_cyt),
    cyt_tot_expr_iqr = IQR(tot_cyt),
    nuc_tot_expr_median = median(tot_nuc),
    nuc_tot_expr_mean = mean(tot_nuc),
    nuc_tot_expr_iqr = IQR(tot_nuc),
    tot_expr_ratio_nuc_to_cyt = median(tot_nuc) / median(tot_cyt),
    
    # Technical dropout: Matrix level
    cyt_matrix_dropout = matrix_dropout_cyt,
    nuc_matrix_dropout = matrix_dropout_nuc,
    delta_matrix_dropout_nuc_minus_cyt = matrix_dropout_nuc - matrix_dropout_cyt,
    
    # Technical dropout: Gene level
    cyt_gene_dropout_median = median(1 - gene_det_rate_cyt),
    nuc_gene_dropout_median = median(1 - gene_det_rate_nuc),
    cyt_gene_det_rate_median = median(gene_det_rate_cyt),
    nuc_gene_det_rate_median = median(gene_det_rate_nuc),
    delta_gene_dropout_nuc_minus_cyt = median(1 - gene_det_rate_nuc) - median(1 - gene_det_rate_cyt),
    
    # N/C ratio distributions (gene-level)
    gene_nc_ratio_median = median(gene_nc_ratio),
    gene_nc_ratio_mean = mean(gene_nc_ratio),
    gene_nc_ratio_iqr = IQR(gene_nc_ratio),
    frac_genes_predom_nuc = mean(gene_nc_ratio > 1.0),
    frac_genes_predom_cyt = mean(gene_nc_ratio < 1.0),
    
    # Cell-level N/C ratio
    cell_nc_ratio_median = cell_nc_median,
    cell_nc_ratio_iqr = cell_nc_iqr,
    
    # Gene-panel & model coverage
    n_model_genes_panel = n_model_genes_panel,
    n_model_genes_det_cyt_1cell = n_model_genes_det_cyt_1cell,
    n_model_genes_det_nuc_1cell = n_model_genes_det_nuc_1cell,
    n_model_genes_det_cyt_1pct = n_model_genes_det_cyt_1pct,
    n_model_genes_det_nuc_1pct = n_model_genes_det_nuc_1pct,
    model_gene_coverage_ratio_1pct = n_model_genes_det_nuc_1pct / max(1, n_model_genes_det_cyt_1pct),
    model_det_rate_cyt_median = model_det_rate_cyt_median,
    model_det_rate_nuc_median = model_det_rate_nuc_median,
    delta_model_det_rate = model_det_rate_cyt_median - model_det_rate_nuc_median
  )
  
  summary_rows[[sample_key]] <- row
  rm(obj, cc_cyt, cc_nuc, cc_cell, mat_cyt, mat_nuc, mat_cell)
  gc()
}

df_summary <- do.call(rbind, summary_rows)
df_genes <- do.call(rbind, gene_rows)
df_subcells <- do.call(rbind, cell_subsample_rows)

out_dir <- "/Users/rjay0256/.gemini/antigravity-ide/brain/c1dfd39f-7c93-4072-91e5-563151ee2bec/scratch"
fwrite(df_summary, file.path(out_dir, "dataset_data_quality_metrics.csv"))
fwrite(df_genes, file.path(out_dir, "gene_level_data_quality_metrics.csv"))
fwrite(df_subcells, file.path(out_dir, "cell_level_subsample_metrics.csv"))

cat("\nDone! Saved outputs to scratch directory.\n")
