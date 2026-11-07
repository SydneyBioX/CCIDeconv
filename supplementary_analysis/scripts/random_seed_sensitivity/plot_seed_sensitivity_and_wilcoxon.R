#!/usr/bin/env Rscript
# ==============================================================================
# plot_seed_sensitivity_and_wilcoxon.R
# 
# 1. Performs paired Wilcoxon signed-rank tests between random seeds (42, 112, 123)
#    across the 9 spatial transcriptomics datasets.
# 2. Generates publication-ready boxplots with individual dataset paired trajectories.
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(rstatix)
  library(ggpubr)
})

# ------------------------------------------------------------------------------
# 1. Paths and Data Loading
# ------------------------------------------------------------------------------
base_dir <- getwd()
csv_path <- file.path(base_dir, "supplementary_analysis", "results", "random_seed_sensitivity", "seed_sensitivity_per_dataset.csv")

if (!file.exists(csv_path)) {
  stop(paste("File not found at:", csv_path))
}

df <- read_csv(csv_path, show_col_types = FALSE) %>%
  mutate(
    Seed = factor(Seed, levels = c("42", "112", "123"), labels = c("Seed 42", "Seed 112", "Seed 123")),
    Dataset = factor(Dataset)
  )

# Define metrics of interest and clean display labels
metric_mapping <- c(
  "AUC"          = "AUROC",
  "macro_recall" = "Macro Recall",
  "macro_f1"     = "Macro F1",
  "cyt_R2"       = "Cytoplasm R²",
  "nuc_R2"       = "Nuclear R²",
  "avg_R2"       = "Average R²",
  "cyt_NRMSE"    = "Cytoplasm NRMSE",
  "nuc_NRMSE"    = "Nuclear NRMSE"
)

# ------------------------------------------------------------------------------
# 2. Paired Wilcoxon Signed-Rank Tests
# ------------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("PAIRED WILCOXON SIGNED-RANK TESTS ACROSS 9 DATASETS (SEEDS: 42, 112, 123)\n")
cat("==============================================================================\n")

wilcox_results_list <- list()

for (m in names(metric_mapping)) {
  test_res <- df %>%
    pairwise_wilcox_test(
      as.formula(paste(m, "~ Seed")),
      paired = TRUE,
      p.adjust.method = "none"
    ) %>%
    mutate(
      metric_code = m,
      Metric = metric_mapping[m]
    )
  wilcox_results_list[[m]] <- test_res
}

wilcox_summary <- bind_rows(wilcox_results_list) %>%
  group_by(Metric) %>%
  adjust_pvalue(method = "BH") %>%
  add_significance("p.adj") %>%
  ungroup() %>%
  select(Metric, group1, group2, n1, n2, statistic, p, p.adj, p.adj.signif)

print(as.data.frame(wilcox_summary), row.names = FALSE)

# Save Wilcoxon test results table
out_dir <- file.path(base_dir, "supplementary_analysis", "results", "random_seed_sensitivity")
figures_dir <- file.path(out_dir, "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

wilcox_table_path <- file.path(out_dir, "paired_wilcoxon_seed_tests.csv")
write_csv(wilcox_summary, wilcox_table_path)
cat(paste0("\nSaved paired Wilcoxon test results to: ", wilcox_table_path, "\n"))

# ------------------------------------------------------------------------------
# 3. Reshape Data for Plotting
# ------------------------------------------------------------------------------
plot_df <- df %>%
  select(Seed, Dataset, all_of(names(metric_mapping))) %>%
  pivot_longer(
    cols = all_of(names(metric_mapping)),
    names_to = "metric_code",
    values_to = "Value"
  ) %>%
  mutate(Metric = factor(metric_mapping[metric_code], levels = unname(metric_mapping)))

# Key 4 Primary Metrics for compact main-text figure
primary_metrics <- c("AUROC", "Macro Recall", "Cytoplasm R²", "Nuclear R²")
plot_df_primary <- plot_df %>% filter(Metric %in% primary_metrics)

# ------------------------------------------------------------------------------
# 4. Generate Publication-Quality Boxplots with "ns" Statistical Brackets
# ------------------------------------------------------------------------------
seed_palette <- c("Seed 42" = "#2B5C8F", "Seed 112" = "#D95F02", "Seed 123" = "#7570B3")

# Compute xy positions for statistical brackets
stat_test_primary <- plot_df_primary %>%
  group_by(Metric) %>%
  pairwise_wilcox_test(Value ~ Seed, paired = TRUE, p.adjust.method = "BH") %>%
  add_significance("p.adj") %>%
  add_xy_position(x = "Seed", step.increase = 0.10)

stat_test_all <- plot_df %>%
  group_by(Metric) %>%
  pairwise_wilcox_test(Value ~ Seed, paired = TRUE, p.adjust.method = "BH") %>%
  add_significance("p.adj") %>%
  add_xy_position(x = "Seed", step.increase = 0.10)

# Plot A: 4 Primary Metrics (Compact 2x2 layout with "ns" displayed on boxes)
p_primary <- ggplot(plot_df_primary, aes(x = Seed, y = Value, fill = Seed)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.75, color = "black", linewidth = 0.5) +
  # Paired gray lines tracking individual datasets across the 3 seeds
  geom_line(aes(group = Dataset), color = "gray60", alpha = 0.5, linewidth = 0.4) +
  geom_point(aes(color = Seed), size = 2.2, shape = 21, stroke = 0.4, fill = "white") +
  # Statistical brackets with 'ns' displayed directly above boxes
  stat_pvalue_manual(
    stat_test_primary,
    label = "p.adj.signif",
    tip.length = 0.015,
    bracket.size = 0.4,
    size = 3.6,
    hide.ns = FALSE
  ) +
  facet_wrap(~ Metric, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = seed_palette) +
  scale_color_manual(values = seed_palette) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  theme_classic(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "#F2F4F7", color = NA),
    strip.text = element_text(face = "bold", size = 11, color = "#1A1A1A"),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = "bold", size = 11),
    axis.text.x = element_text(face = "bold", size = 10, color = "black"),
    legend.position = "none",
    panel.grid.major.y = element_line(color = "#E9ECEF", linetype = "dashed", linewidth = 0.4)
  ) +
  labs(
    y = "Evaluation Metric Value",
    title = "CCIDeconv Performance Across Random Seeds (LOGO-CV, N = 9 Datasets)",
    subtitle = "Pairwise Paired Wilcoxon Signed-Rank Tests (BH-adjusted p-values; ns = non-significant)"
  )

# Plot B: All 8 Comprehensive Metrics (Faceted 4x2 layout with "ns" displayed on boxes)
p_all <- ggplot(plot_df, aes(x = Seed, y = Value, fill = Seed)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.75, color = "black", linewidth = 0.5) +
  geom_line(aes(group = Dataset), color = "gray60", alpha = 0.5, linewidth = 0.4) +
  geom_point(aes(color = Seed), size = 2.0, shape = 21, stroke = 0.4, fill = "white") +
  stat_pvalue_manual(
    stat_test_all,
    label = "p.adj.signif",
    tip.length = 0.012,
    bracket.size = 0.35,
    size = 3.2,
    hide.ns = FALSE
  ) +
  facet_wrap(~ Metric, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = seed_palette) +
  scale_color_manual(values = seed_palette) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  theme_classic(base_size = 11) +
  theme(
    strip.background = element_rect(fill = "#F2F4F7", color = NA),
    strip.text = element_text(face = "bold", size = 10, color = "#1A1A1A"),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = "bold", size = 10),
    axis.text.x = element_text(face = "bold", size = 9, color = "black"),
    legend.position = "bottom",
    panel.grid.major.y = element_line(color = "#E9ECEF", linetype = "dashed", linewidth = 0.4)
  ) +
  labs(
    y = "Metric Score",
    title = "Comprehensive Random Seed Sensitivity Benchmark Across 9 Spatial Datasets",
    subtitle = "Pairwise Paired Wilcoxon Signed-Rank Tests (BH-adjusted p-values; ns = non-significant)"
  )

# Plot C: Non-Faceted Grouped Boxplot (Composite Metric & NRMSE, colored by Seed)
target_metrics_grouped <- c(
  "composite_metric" = "Composite Metric",
  "cyt_NRMSE"        = "Cytoplasm NRMSE",
  "nuc_NRMSE"        = "Nuclear NRMSE",
  "avg_NRMSE"        = "Average NRMSE"
)

plot_df_grouped <- df %>%
  select(Seed, Dataset, all_of(names(target_metrics_grouped))) %>%
  pivot_longer(cols = all_of(names(target_metrics_grouped)), names_to = "metric_code", values_to = "Value") %>%
  mutate(Metric = factor(target_metrics_grouped[metric_code], levels = unname(target_metrics_grouped)))

stat_test_grouped <- plot_df_grouped %>%
  group_by(Metric) %>%
  pairwise_wilcox_test(Value ~ Seed, paired = TRUE, p.adjust.method = "BH") %>%
  add_significance("p.adj") %>%
  add_xy_position(x = "Metric", dodge = 0.8)

p_non_faceted <- ggplot(plot_df_grouped, aes(x = Metric, y = Value, fill = Seed)) +
  geom_boxplot(width = 0.65, outlier.shape = NA, alpha = 0.75, color = "black", linewidth = 0.5,
               position = position_dodge(0.8)) +
  geom_point(aes(color = Seed), size = 1.8, shape = 21, stroke = 0.3, fill = "white",
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8)) +
  stat_pvalue_manual(
    stat_test_grouped,
    label = "p.adj.signif",
    tip.length = 0.015,
    bracket.size = 0.35,
    size = 3.2,
    hide.ns = FALSE
  ) +
  scale_fill_manual(values = seed_palette) +
  scale_color_manual(values = seed_palette) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  theme_classic(base_size = 12) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = "bold", size = 11),
    axis.text.x = element_text(face = "bold", size = 11, color = "black"),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    panel.grid.major.y = element_line(color = "#E9ECEF", linetype = "dashed", linewidth = 0.4)
  ) +
  labs(
    y = "Metric Score",
    fill = "Sensitivity Seed:",
    color = "Sensitivity Seed:",
    title = "Composite Metric and NRMSE Across 9 Spatial Datasets",
    subtitle = "Non-faceted grouped boxplot colored by sensitivity seed with pairwise paired Wilcoxon 'ns' brackets"
  )

# ------------------------------------------------------------------------------
# 5. Export Figures (Vector PDF and high-res PNG)
# ------------------------------------------------------------------------------
pdf_primary_path <- file.path(figures_dir, "seed_sensitivity_boxplots_primary.pdf")
png_primary_path <- file.path(figures_dir, "seed_sensitivity_boxplots_primary.png")
ggsave(pdf_primary_path, plot = p_primary, width = 7.5, height = 6.0, device = "pdf")
ggsave(png_primary_path, plot = p_primary, width = 7.5, height = 6.0, dpi = 300)

pdf_all_path <- file.path(figures_dir, "seed_sensitivity_boxplots_all.pdf")
png_all_path <- file.path(figures_dir, "seed_sensitivity_boxplots_all.png")
ggsave(pdf_all_path, plot = p_all, width = 12.0, height = 6.5, device = "pdf")
ggsave(png_all_path, plot = p_all, width = 12.0, height = 6.5, dpi = 300)

pdf_non_faceted_path <- file.path(figures_dir, "composite_and_nrmse_non_faceted.pdf")
png_non_faceted_path <- file.path(figures_dir, "composite_and_nrmse_non_faceted.png")
ggsave(pdf_non_faceted_path, plot = p_non_faceted, width = 8.5, height = 5.5, device = "pdf")
ggsave(png_non_faceted_path, plot = p_non_faceted, width = 8.5, height = 5.5, dpi = 300)

cat(paste0("Saved primary figure to:      ", pdf_primary_path, "\n"))
cat(paste0("Saved full figure to:         ", pdf_all_path, "\n"))
cat(paste0("Saved non-faceted figure to:  ", pdf_non_faceted_path, "\n"))
cat("Plotting and statistical testing completed successfully!\n")
