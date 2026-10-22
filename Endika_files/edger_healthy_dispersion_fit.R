## ---------------------------------------------------------------------------
## edgeR NB dispersion fit on healthy-donor TCR repertoire data
##
## Goal: independent edgeR estimate of the shared NB dispersion (phi) on the
## FULLY-OBSERVED subset of healthy-donor clonotypes (detected at every
## timepoint for that donor -> no missingness to interpret), to compare
## against the Python pipeline's sparse-fit phi (~4.0-4.3) and the
## complete-case phi (~3.86) reported in fit_nb_dispersion_mle().
##
## Recall: edgeR's `dispersion` and the pipeline's `phi` are reciprocals,
##   phi = 1 / dispersion
## and clonotype IDs (clono_1, clono_2, ...) are locally scoped PER DONOR,
## so every step below (pivot, DGEList, normalization, dispersion fit) is
## done strictly within one donor at a time -- never pooled by raw ID.
## ---------------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(edgeR)

## --- 1. Load data -----------------------------------------------------------
## Expected: a long-format data.frame with columns patient, day, clono, count
## (the pipeline's canonical format), saved as a CSV. This is a SPARSE
## representation: a row exists only if that clonotype was detected in that
## donor at that timepoint (absent rows = not detected, not a true zero).
##
## Adjust the path below (use an absolute path, or setwd() first, to avoid
## "cannot open the connection" errors from a relative path).
data_path <- "/home/lgavira/VHIO-colab/data/Healthy-Data-long.csv"
healthy_long <- read.csv(data_path, stringsAsFactors = FALSE)

stopifnot(all(c("patient", "day", "clono", "count") %in% colnames(healthy_long)))

## --- 2. Per-donor library size (from the FULL sparse data, before any
##        subsetting -- matches add_library_sizes() in nb_glm.py) -----------
lib_sizes <- healthy_long %>%
  group_by(patient, day) %>%
  summarise(lib = sum(count), .groups = "drop")

## --- 3. Per-donor fit ---------------------------------------------------
donors <- unique(healthy_long$patient)
donor_results <- list()

for (donor in donors) {

  sub <- healthy_long %>% filter(patient == donor)
  donor_days <- sort(unique(sub$day))
  n_days <- length(donor_days)
  if (n_days < 2) next

  ## Pivot to clonotype x timepoint count matrix. tidyr::pivot_wider on a
  ## sparse long df fills un-observed (clono, day) combinations with NA --
  ## exactly what we want, since NA here means "not detected", not "zero".
  mat_df <- sub %>%
    select(clono, day, count) %>%
    pivot_wider(names_from = day, values_from = count) %>%
    arrange(clono)

  count_mat <- as.matrix(mat_df[, -1])
  rownames(count_mat) <- mat_df$clono
  colnames(count_mat) <- as.character(donor_days)

  ## Fully-observed subset: clonotypes with a real (non-NA) count at every
  ## timepoint for this donor -- the complete-case restriction that gave
  ## phi ~= 3.86 in the Python fit.
  fully_observed <- rowSums(is.na(count_mat)) == 0
  n_fo <- sum(fully_observed)
  if (n_fo < 5) {
    message(sprintf("Donor %s: only %d fully-observed clonotypes, skipping.", donor, n_fo))
    next
  }
  count_mat <- count_mat[fully_observed, , drop = FALSE]
  storage.mode(count_mat) <- "double"

  ## Library sizes for this donor's timepoints, from the FULL sparse data
  ## (not just the fully-observed subset) -- an external, fixed offset.
  donor_lib <- lib_sizes %>%
    filter(patient == donor, day %in% donor_days) %>%
    arrange(day) %>%
    pull(lib)

  ## --- edgeR fit, strictly within this donor ---
  y <- DGEList(counts = count_mat, lib.size = donor_lib)
  y <- calcNormFactors(y, method = "TMM")           # per-donor, across its own timepoints
  design <- matrix(1, ncol = 1, nrow = n_days)       # intercept-only: no covariate of interest here
  y <- estimateDisp(y, design)

  common_dispersion <- y$common.dispersion
  phi_edger <- 1 / common_dispersion

  donor_results[[donor]] <- data.frame(
    patient = donor,
    n_timepoints = n_days,
    n_clonotypes_fully_observed = n_fo,
    common_dispersion = common_dispersion,
    phi_edger = phi_edger
  )

  message(sprintf("Donor %-20s  n_tp=%d  n_clono=%-5d  dispersion=%.5f  phi=%.3f",
                   donor, n_days, n_fo, common_dispersion, phi_edger))
}

## --- 4. Summarize across donors (equal-per-donor weighting, NOT naive
##        concatenation -- per the pooling bug noted in the Python pipeline) -
results <- bind_rows(donor_results)
print(results)

cat("\n--- Pooled comparison figures ---\n")
cat(sprintf("Mean   phi across donors (unweighted): %.3f\n", mean(results$phi_edger)))
cat(sprintf("Median phi across donors (unweighted): %.3f\n", median(results$phi_edger)))
cat("Compare against Python fit_nb_dispersion_mle() on the fully-observed\n")
cat("subset (expected ~3.86) and the sparse fit (~4.0-4.3).\n")

## --- 5. Save for downstream comparison in the notebook ---------------------
out_path <- "/home/lgavira/notebook/edger_healthy_dispersion_results.rds"
saveRDS(results, out_path)