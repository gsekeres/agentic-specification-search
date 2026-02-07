#!/usr/bin/env Rscript
#
# Specification Search for Paper 207766-V1
# "Organized Voters: Evidence from Governmental Transfers" by Camille Urvoy (AER 2025)
#
# This paper uses a regression discontinuity design to study how government transfers
# to nonprofit organizations differ based on political alignment between local and
# national governments in France.
#

# Load required packages
required_packages <- c("haven", "rdrobust", "dplyr", "tidyr", "jsonlite")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# Paths
DATA_PATH <- "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/207766-V1/urvoy_organized_replication/processed_data/data_ready.dta"
OUTPUT_DIR <- "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/207766-V1/urvoy_organized_replication"

# Paper metadata
PAPER_ID <- "207766-V1"
JOURNAL <- "AER"
PAPER_TITLE <- "Organized Voters: Evidence from Governmental Transfers"

# Load data
cat("Loading data...\n")
df <- haven::read_dta(DATA_PATH)
cat(sprintf("Loaded %d observations\n", nrow(df)))

# Filter to analysis sample
df_analysis <- df %>% filter(insample == 1)
cat(sprintf("Analysis sample: %d observations\n", nrow(df_analysis)))

# Key variables
RUNNING_VAR <- "alignm"
CUTOFF <- 0
CLUSTER_VAR <- "comn"

# Main outcomes
MAIN_OUTCOMES <- c("amount_congr1_3_pcap", "amount_congr2_3_pcap", "amount_congr3_3_pcap", "amount_pcap")
PRIMARY_OUTCOME <- "amount_congr1_3_pcap"

# Initialize results
results <- list()

run_rd_spec <- function(data, outcome_var, spec_id, spec_tree_path, sample_desc,
                        controls_desc = "None", p = 1, kernel = "triangular",
                        bw = NULL, cluster = TRUE) {

  # Prepare data
  y <- data[[outcome_var]]
  x <- data[[RUNNING_VAR]]
  cluster_var <- if (cluster) data[[CLUSTER_VAR]] else NULL

  # Remove NA
  valid <- !is.na(y) & !is.na(x)
  if (cluster) valid <- valid & !is.na(cluster_var)

  y <- y[valid]
  x <- x[valid]
  if (cluster) cluster_var <- cluster_var[valid]

  if (length(y) < 50) {
    cat(sprintf("  Skipping %s: too few observations (%d)\n", spec_id, length(y)))
    return(NULL)
  }

  tryCatch({
    # Run rdrobust
    if (is.null(bw)) {
      if (cluster) {
        rd <- rdrobust(y, x, c = CUTOFF, p = p, kernel = kernel, cluster = cluster_var)
      } else {
        rd <- rdrobust(y, x, c = CUTOFF, p = p, kernel = kernel)
      }
    } else {
      if (cluster) {
        rd <- rdrobust(y, x, c = CUTOFF, p = p, kernel = kernel, h = bw, cluster = cluster_var)
      } else {
        rd <- rdrobust(y, x, c = CUTOFF, p = p, kernel = kernel, h = bw)
      }
    }

    # Extract results
    coef <- rd$coef[1]  # Conventional
    se <- rd$se[1]
    pval <- rd$pv[1]
    pval_rb <- rd$pv[3]  # Robust bias-corrected
    ci_l <- rd$ci[1, 1]
    ci_u <- rd$ci[1, 2]
    bw_l <- rd$bws[1, 1]
    bw_r <- rd$bws[1, 2]
    n_l <- rd$N_h[1]
    n_r <- rd$N_h[2]
    n_obs <- n_l + n_r

    # Coefficient vector JSON
    coef_vector <- list(
      treatment = list(
        var = "above_cutoff",
        coef = coef,
        se = se,
        pval = pval,
        ci_lower = ci_l,
        ci_upper = ci_u,
        robust_pval = pval_rb
      ),
      running_variable = list(
        var = RUNNING_VAR,
        cutoff = CUTOFF,
        bandwidth_left = bw_l,
        bandwidth_right = bw_r
      ),
      diagnostics = list(
        polynomial_order = p,
        kernel = kernel,
        n_left = n_l,
        n_right = n_r
      ),
      n_obs = n_obs
    )

    t_stat <- if (se > 0) coef / se else NA

    result <- data.frame(
      paper_id = PAPER_ID,
      journal = JOURNAL,
      paper_title = PAPER_TITLE,
      spec_id = spec_id,
      spec_tree_path = spec_tree_path,
      outcome_var = outcome_var,
      treatment_var = paste0(RUNNING_VAR, " > ", CUTOFF),
      coefficient = coef,
      std_error = se,
      t_stat = t_stat,
      p_value = pval,
      p_value_robust = pval_rb,
      ci_lower = ci_l,
      ci_upper = ci_u,
      n_obs = n_obs,
      r_squared = NA,
      coefficient_vector_json = as.character(toJSON(coef_vector, auto_unbox = TRUE)),
      sample_desc = sample_desc,
      fixed_effects = "None",
      controls_desc = controls_desc,
      cluster_var = if (cluster) CLUSTER_VAR else "None",
      model_type = sprintf("RD_p%d_%s", p, kernel),
      estimation_script = sprintf("scripts/paper_analyses/%s.R", PAPER_ID),
      bandwidth = bw_l,
      stringsAsFactors = FALSE
    )

    cat(sprintf("  %s: coef=%.4f, se=%.4f, p=%.4f, n=%d\n", spec_id, coef, se, pval, n_obs))
    return(result)

  }, error = function(e) {
    cat(sprintf("  Error in %s: %s\n", spec_id, e$message))
    return(NULL)
  })
}

cat("\n================================================================================\n")
cat("SPECIFICATION SEARCH FOR PAPER 207766-V1\n")
cat("================================================================================\n")

# -----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATIONS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("1. BASELINE SPECIFICATIONS\n")
cat("--------------------------------------------------------------------------------\n")

for (outcome in MAIN_OUTCOMES) {
  res <- run_rd_spec(df_analysis, outcome, "baseline",
                     "methods/regression_discontinuity.md#baseline",
                     "Full analysis sample (insample==1)")
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# -----------------------------------------------------------------------------
# 2. BANDWIDTH VARIATIONS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("2. BANDWIDTH VARIATIONS\n")
cat("--------------------------------------------------------------------------------\n")

# Get optimal bandwidth
bw_sel <- rdbwselect(df_analysis[[PRIMARY_OUTCOME]], df_analysis[[RUNNING_VAR]], c = CUTOFF)
opt_bw <- bw_sel$bws[1, 1]
cat(sprintf("Optimal bandwidth: %.2f\n", opt_bw))

# Bandwidth multipliers
bw_multipliers <- c(0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5)

for (mult in bw_multipliers) {
  bw <- opt_bw * mult
  res <- run_rd_spec(df_analysis, PRIMARY_OUTCOME,
                     sprintf("rd/bandwidth/mult_%.2f", mult),
                     "methods/regression_discontinuity.md#bandwidth-selection",
                     sprintf("Bandwidth = %.2fx optimal (%.2f)", mult, bw),
                     bw = bw)
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# Fixed bandwidths
for (bw_fixed in c(5, 7, 10, 15, 20)) {
  res <- run_rd_spec(df_analysis, PRIMARY_OUTCOME,
                     sprintf("rd/bandwidth/fixed_%d", bw_fixed),
                     "methods/regression_discontinuity.md#bandwidth-selection",
                     sprintf("Fixed bandwidth = %d", bw_fixed),
                     bw = bw_fixed)
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# -----------------------------------------------------------------------------
# 3. POLYNOMIAL ORDER VARIATIONS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("3. POLYNOMIAL ORDER VARIATIONS\n")
cat("--------------------------------------------------------------------------------\n")

for (p_order in c(1, 2, 3)) {
  for (outcome in MAIN_OUTCOMES) {
    res <- run_rd_spec(df_analysis, outcome,
                       sprintf("rd/poly/order_%d", p_order),
                       "methods/regression_discontinuity.md#polynomial-order",
                       "Full analysis sample",
                       p = p_order)
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# -----------------------------------------------------------------------------
# 4. KERNEL VARIATIONS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("4. KERNEL VARIATIONS\n")
cat("--------------------------------------------------------------------------------\n")

for (kernel in c("triangular", "uniform", "epanechnikov")) {
  for (outcome in MAIN_OUTCOMES) {
    res <- run_rd_spec(df_analysis, outcome,
                       sprintf("rd/kernel/%s", kernel),
                       "methods/regression_discontinuity.md#kernel-function",
                       "Full analysis sample",
                       kernel = kernel)
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# -----------------------------------------------------------------------------
# 5. SAMPLE RESTRICTIONS - LEFT-RIGHT RACES
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("5. SAMPLE RESTRICTIONS - LEFT-RIGHT RACES\n")
cat("--------------------------------------------------------------------------------\n")

df_lr <- df_analysis %>% filter(insample_leftright == 1)

for (outcome in MAIN_OUTCOMES) {
  res <- run_rd_spec(df_lr, outcome,
                     "rd/sample/leftright_only",
                     "robustness/sample_restrictions.md",
                     "Left-right races only (insample_leftright==1)")
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# -----------------------------------------------------------------------------
# 6. SAMPLE RESTRICTIONS - BY YEAR
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("6. SAMPLE RESTRICTIONS - BY YEAR\n")
cat("--------------------------------------------------------------------------------\n")

years <- unique(na.omit(df_analysis$year))

for (yr in years) {
  df_year <- df_analysis %>% filter(year == yr)
  res <- run_rd_spec(df_year, PRIMARY_OUTCOME,
                     sprintf("rd/sample/year_%d", yr),
                     "robustness/sample_restrictions.md",
                     sprintf("Year = %d", yr))
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# Drop each year
for (yr in years) {
  df_noyear <- df_analysis %>% filter(year != yr)
  res <- run_rd_spec(df_noyear, PRIMARY_OUTCOME,
                     sprintf("rd/sample/drop_year_%d", yr),
                     "robustness/sample_restrictions.md",
                     sprintf("Excluding year %d", yr))
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# Pre-election years
df_preelec <- df_analysis %>% filter(year %in% c(2006, 2007, 2012, 2013))
res <- run_rd_spec(df_preelec, PRIMARY_OUTCOME,
                   "rd/sample/pre_election_years",
                   "robustness/sample_restrictions.md",
                   "Pre-election years (2006-2007, 2012-2013)")
if (!is.null(res)) results[[length(results) + 1]] <- res

# Non-pre-election years
df_nonpreelec <- df_analysis %>% filter(!year %in% c(2006, 2007, 2012, 2013))
res <- run_rd_spec(df_nonpreelec, PRIMARY_OUTCOME,
                   "rd/sample/non_pre_election_years",
                   "robustness/sample_restrictions.md",
                   "Non-pre-election years")
if (!is.null(res)) results[[length(results) + 1]] <- res

# -----------------------------------------------------------------------------
# 7. SAMPLE RESTRICTIONS - BY POPULATION
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("7. SAMPLE RESTRICTIONS - BY POPULATION\n")
cat("--------------------------------------------------------------------------------\n")

if ("pop_all" %in% names(df_analysis)) {
  pop_median <- median(df_analysis$pop_all, na.rm = TRUE)

  df_large <- df_analysis %>% filter(pop_all >= pop_median)
  df_small <- df_analysis %>% filter(pop_all < pop_median)

  res <- run_rd_spec(df_large, PRIMARY_OUTCOME,
                     "rd/sample/large_municipalities",
                     "robustness/sample_restrictions.md",
                     "Above-median population")
  if (!is.null(res)) results[[length(results) + 1]] <- res

  res <- run_rd_spec(df_small, PRIMARY_OUTCOME,
                     "rd/sample/small_municipalities",
                     "robustness/sample_restrictions.md",
                     "Below-median population")
  if (!is.null(res)) results[[length(results) + 1]] <- res

  # Campaign spending cap threshold
  df_capped <- df_analysis %>% filter(pop_all >= 9000)
  df_uncapped <- df_analysis %>% filter(pop_all < 9000)

  res <- run_rd_spec(df_capped, PRIMARY_OUTCOME,
                     "rd/sample/campaign_spending_capped",
                     "robustness/sample_restrictions.md",
                     "Pop >= 9000 (campaign spending capped)")
  if (!is.null(res)) results[[length(results) + 1]] <- res

  res <- run_rd_spec(df_uncapped, PRIMARY_OUTCOME,
                     "rd/sample/campaign_spending_uncapped",
                     "robustness/sample_restrictions.md",
                     "Pop < 9000 (campaign spending uncapped)")
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# Two-candidate races
if ("nb_candi_1rd_all" %in% names(df_analysis)) {
  df_two <- df_analysis %>% filter(nb_candi_1rd_all == 2)
  for (outcome in MAIN_OUTCOMES[1:2]) {
    res <- run_rd_spec(df_two, outcome,
                       "rd/sample/two_candidates",
                       "robustness/sample_restrictions.md",
                       "Two-candidate races only")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# -----------------------------------------------------------------------------
# 8. DONUT HOLE SPECIFICATIONS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("8. DONUT HOLE SPECIFICATIONS\n")
cat("--------------------------------------------------------------------------------\n")

for (donut in c(0.5, 1, 2, 3, 5)) {
  df_donut <- df_analysis %>% filter(abs(.data[[RUNNING_VAR]]) > donut)
  res <- run_rd_spec(df_donut, PRIMARY_OUTCOME,
                     sprintf("rd/donut/exclude_%.1fpp", donut),
                     "methods/regression_discontinuity.md#donut-hole-specifications",
                     sprintf("Donut hole: exclude |alignm| <= %.1f", donut))
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# -----------------------------------------------------------------------------
# 9. ALTERNATIVE OUTCOMES - SPLIT GROUPS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("9. ALTERNATIVE OUTCOMES - SPLIT GROUPS\n")
cat("--------------------------------------------------------------------------------\n")

# 2-group split
alt_outcomes_2 <- c("amount_congr1_2_pcap", "amount_congr2_2_pcap")
for (outcome in alt_outcomes_2) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/split_2groups",
                       "robustness/measurement.md",
                       "Organizations split into 2 groups")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# 4-group split
alt_outcomes_4 <- c("amount_congr1_4_pcap", "amount_congr2_4_pcap", "amount_congr3_4_pcap", "amount_congr4_4_pcap")
for (outcome in alt_outcomes_4) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/split_4groups",
                       "robustness/measurement.md",
                       "Organizations split into 4 groups")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Extensive margin
extensive_outcomes <- c("ntr_congr1_3_pcap", "ntr_congr2_3_pcap", "ntr_congr3_3_pcap", "ntr_pcap")
for (outcome in extensive_outcomes) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/extensive_margin",
                       "robustness/measurement.md",
                       "Extensive margin (number of orgs)")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Young organizations
young_outcomes <- c("amount_congr1_3_yng_pcap", "amount_congr2_3_yng_pcap", "amount_congr3_3_yng_pcap", "amount_yng_pcap")
for (outcome in young_outcomes) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/young_orgs",
                       "robustness/measurement.md",
                       "Young organizations (<= 6 years old)")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Old organizations
old_outcomes <- c("amount_congr1_3_old_pcap", "amount_congr2_3_old_pcap", "amount_congr3_3_old_pcap", "amount_old_pcap")
for (outcome in old_outcomes) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/old_orgs",
                       "robustness/measurement.md",
                       "Old organizations (> 6 years old)")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Winsorized 95%
wins_95 <- c("amount_w95_congr1_3_pcap", "amount_w95_congr2_3_pcap", "amount_w95_congr3_3_pcap", "amount_w95_pcap")
for (outcome in wins_95) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/winsorized_95",
                       "robustness/sample_restrictions.md#outlier-treatment",
                       "Winsorized at 95th percentile")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Winsorized 99%
wins_99 <- c("amount_w99_congr1_3_pcap", "amount_w99_congr2_3_pcap", "amount_w99_congr3_3_pcap", "amount_w99_pcap")
for (outcome in wins_99) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/winsorized_99",
                       "robustness/sample_restrictions.md#outlier-treatment",
                       "Winsorized at 99th percentile")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Multiple ministries
multimin <- c("amount_congr1_3_multimin_pcap", "amount_congr2_3_multimin_pcap", "amount_congr3_3_multimin_pcap", "amount_multimin_pcap")
for (outcome in multimin) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/multiple_ministries",
                       "robustness/measurement.md",
                       "Organizations receiving from multiple ministries")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Single ministry
onemin <- c("amount_congr1_3_onemin_pcap", "amount_congr2_3_onemin_pcap", "amount_congr3_3_onemin_pcap", "amount_onemin_pcap")
for (outcome in onemin) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/single_ministry",
                       "robustness/measurement.md",
                       "Organizations receiving from single ministry")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# Non-local organizations
nonlocal <- c("amount_hq_congr1_3_noloc_pcap", "amount_hq_congr2_3_noloc_pcap", "amount_hq_congr3_3_noloc_pcap", "amount_hq_noloc_pcap")
for (outcome in nonlocal) {
  if (outcome %in% names(df_analysis)) {
    res <- run_rd_spec(df_analysis, outcome,
                       "rd/outcome/nonlocal_orgs",
                       "robustness/measurement.md",
                       "Non-local organizations (HQ outside municipality)")
    if (!is.null(res)) results[[length(results) + 1]] <- res
  }
}

# -----------------------------------------------------------------------------
# 10. HETEROGENEITY ANALYSES
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("10. HETEROGENEITY ANALYSES\n")
cat("--------------------------------------------------------------------------------\n")

# By campaign spending
if ("campexp_aligned_cand_past_high" %in% names(df_analysis)) {
  df_high <- df_analysis %>% filter(campexp_aligned_cand_past_high == 1)
  df_low <- df_analysis %>% filter(campexp_aligned_cand_past_high == 0)

  res <- run_rd_spec(df_high, PRIMARY_OUTCOME,
                     "rd/het/campaign_spending_high",
                     "robustness/heterogeneity.md",
                     "High campaign spending municipalities")
  if (!is.null(res)) results[[length(results) + 1]] <- res

  res <- run_rd_spec(df_low, PRIMARY_OUTCOME,
                     "rd/het/campaign_spending_low",
                     "robustness/heterogeneity.md",
                     "Low campaign spending municipalities")
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# By government party popularity
if ("pres_score_high" %in% names(df_analysis)) {
  for (level in c(1, 0)) {
    label <- if (level == 1) "popular" else "unpopular"
    df_het <- df_analysis %>% filter(pres_score_high == level)
    for (outcome in MAIN_OUTCOMES[1:2]) {
      res <- run_rd_spec(df_het, outcome,
                         sprintf("rd/het/govt_party_%s", label),
                         "robustness/heterogeneity.md",
                         sprintf("Govt party %s locally", label))
      if (!is.null(res)) results[[length(results) + 1]] <- res
    }
  }
}

# By incumbent status
if ("wasinpower" %in% names(df_analysis)) {
  for (level in c(1, 0)) {
    label <- if (level == 1) "incumbent" else "new"
    df_het <- df_analysis %>% filter(wasinpower == level)
    for (outcome in MAIN_OUTCOMES[1:2]) {
      res <- run_rd_spec(df_het, outcome,
                         sprintf("rd/het/incumbent_%s", label),
                         "robustness/heterogeneity.md",
                         sprintf("Govt party %s in power last term", if(level==1) "was" else "was not"))
      if (!is.null(res)) results[[length(results) + 1]] <- res
    }
  }
}

# By turnout
if ("low_turnout" %in% names(df_analysis)) {
  for (level in c(1, 0)) {
    label <- if (level == 1) "low" else "high"
    df_het <- df_analysis %>% filter(low_turnout == level)
    for (outcome in MAIN_OUTCOMES[1:2]) {
      res <- run_rd_spec(df_het, outcome,
                         sprintf("rd/het/turnout_%s", label),
                         "robustness/heterogeneity.md",
                         sprintf("%s turnout municipalities", if(level==1) "Low" else "High"))
      if (!is.null(res)) results[[length(results) + 1]] <- res
    }
  }
}

# -----------------------------------------------------------------------------
# 11. PLACEBO CUTOFF TESTS
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("11. PLACEBO CUTOFF TESTS\n")
cat("--------------------------------------------------------------------------------\n")

placebo_cutoffs <- c(-15, -10, -5, 5, 10, 15)

for (pc in placebo_cutoffs) {
  # Filter to appropriate side
  if (pc < 0) {
    df_placebo <- df_analysis %>% filter(.data[[RUNNING_VAR]] < 0)
  } else {
    df_placebo <- df_analysis %>% filter(.data[[RUNNING_VAR]] > 0)
  }

  if (nrow(df_placebo) > 100) {
    tryCatch({
      y <- df_placebo[[PRIMARY_OUTCOME]]
      x <- df_placebo[[RUNNING_VAR]]
      cluster_var <- df_placebo[[CLUSTER_VAR]]

      valid <- !is.na(y) & !is.na(x) & !is.na(cluster_var)
      y <- y[valid]
      x <- x[valid]
      cluster_var <- cluster_var[valid]

      rd <- rdrobust(y, x, c = pc, cluster = cluster_var)

      coef <- rd$coef[1]
      se <- rd$se[1]
      pval <- rd$pv[1]
      bw_l <- rd$bws[1, 1]
      n_obs <- sum(rd$N_h)

      coef_vector <- list(
        treatment = list(var = sprintf("placebo_cutoff_%d", pc), coef = coef, se = se, pval = pval),
        running_variable = list(var = RUNNING_VAR, placebo_cutoff = pc),
        n_obs = n_obs
      )

      result <- data.frame(
        paper_id = PAPER_ID,
        journal = JOURNAL,
        paper_title = PAPER_TITLE,
        spec_id = sprintf("rd/placebo/cutoff_%d", pc),
        spec_tree_path = "methods/regression_discontinuity.md#placebo-cutoff-tests",
        outcome_var = PRIMARY_OUTCOME,
        treatment_var = sprintf("%s > %d", RUNNING_VAR, pc),
        coefficient = coef,
        std_error = se,
        t_stat = if (se > 0) coef / se else NA,
        p_value = pval,
        p_value_robust = rd$pv[3],
        ci_lower = rd$ci[1, 1],
        ci_upper = rd$ci[1, 2],
        n_obs = n_obs,
        r_squared = NA,
        coefficient_vector_json = as.character(toJSON(coef_vector, auto_unbox = TRUE)),
        sample_desc = sprintf("Placebo cutoff at %d", pc),
        fixed_effects = "None",
        controls_desc = "None",
        cluster_var = CLUSTER_VAR,
        model_type = "RD_placebo",
        estimation_script = sprintf("scripts/paper_analyses/%s.R", PAPER_ID),
        bandwidth = bw_l,
        stringsAsFactors = FALSE
      )

      results[[length(results) + 1]] <- result
      cat(sprintf("  rd/placebo/cutoff_%d: coef=%.4f, se=%.4f, p=%.4f\n", pc, coef, se, pval))

    }, error = function(e) {
      cat(sprintf("  Error in placebo cutoff %d: %s\n", pc, e$message))
    })
  }
}

# -----------------------------------------------------------------------------
# 12. INFERENCE VARIATIONS (no clustering)
# -----------------------------------------------------------------------------
cat("\n--------------------------------------------------------------------------------\n")
cat("12. INFERENCE VARIATIONS (no clustering)\n")
cat("--------------------------------------------------------------------------------\n")

for (outcome in MAIN_OUTCOMES) {
  res <- run_rd_spec(df_analysis, outcome,
                     "rd/inference/no_cluster",
                     "robustness/clustering_variations.md",
                     "Full analysis sample",
                     cluster = FALSE)
  if (!is.null(res)) results[[length(results) + 1]] <- res
}

# =============================================================================
# SAVE RESULTS
# =============================================================================

cat("\n================================================================================\n")
cat("SAVING RESULTS\n")
cat("================================================================================\n")

# Combine results
results_df <- bind_rows(results)

# Summary statistics
cat(sprintf("\nTotal specifications: %d\n", nrow(results_df)))
cat(sprintf("Positive coefficients: %d (%.1f%%)\n",
            sum(results_df$coefficient > 0),
            mean(results_df$coefficient > 0) * 100))
cat(sprintf("Significant at 5%%: %d (%.1f%%)\n",
            sum(results_df$p_value < 0.05),
            mean(results_df$p_value < 0.05) * 100))
cat(sprintf("Significant at 1%%: %d (%.1f%%)\n",
            sum(results_df$p_value < 0.01),
            mean(results_df$p_value < 0.01) * 100))

# For main outcome
main_results <- results_df %>% filter(outcome_var == PRIMARY_OUTCOME)
cat(sprintf("\nFor main outcome (%s):\n", PRIMARY_OUTCOME))
cat(sprintf("  Specifications: %d\n", nrow(main_results)))
cat(sprintf("  Positive: %d (%.1f%%)\n",
            sum(main_results$coefficient > 0),
            mean(main_results$coefficient > 0) * 100))
cat(sprintf("  Sig at 5%%: %d (%.1f%%)\n",
            sum(main_results$p_value < 0.05),
            mean(main_results$p_value < 0.05) * 100))

# Save
output_path <- file.path(OUTPUT_DIR, "specification_results.csv")
write.csv(results_df, output_path, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_path))

cat("\nDone!\n")
