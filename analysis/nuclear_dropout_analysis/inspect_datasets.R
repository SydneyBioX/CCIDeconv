suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
})

rds_dir <- "/Users/rjay0256/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney/PhD projects/main/Cell Cell Communication/server/cell_cell_commuication/data/processed/cellchat_ablated_significant"
files <- list.files(rds_dir, pattern = "spe_list_.*rds$", full.names = TRUE)

cat("Found", length(files), "files\n")
for (f in files) {
  obj <- readRDS(f)
  bname <- basename(f)
  cat(sprintf("=== Dataset: %s ===\n", bname))
  for (comp in c("cell", "cyt", "nuc")) {
    cc <- obj[[comp]]
    if (!is.null(cc)) {
      cat(sprintf("  %-4s: n_genes=%d, n_cells=%d, data_class=%s\n", 
                  comp, nrow(cc@data), ncol(cc@data), class(cc@data)[1]))
    } else {
      cat(sprintf("  %-4s: NULL\n", comp))
    }
  }
}
