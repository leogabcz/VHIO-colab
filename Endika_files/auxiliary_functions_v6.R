### function ::load_data::
### Loads data from adaptive or hsd outputs, recalculates frequencies for "in-frame" sequences and computes fold-changes relative to baseline.
load_data <- function(samples,
                      datapath.pbl,
                      output.list.name,
                      technology = "adaptive",
                      min_freq = 0.0001) {
  min_freq <<- min_freq
  if (!exists(output.list.name, envir = .GlobalEnv)) {
    assign(output.list.name, list(), envir = .GlobalEnv)
  }
  output_list <- get(output.list.name, envir = .GlobalEnv)
  fc_output_list_name <- paste0(output.list.name, "_fc")
  if (!exists(fc_output_list_name, envir = .GlobalEnv)) {
    assign(fc_output_list_name, list(), envir = .GlobalEnv)
  }
  fc_output_list <- get(fc_output_list_name, envir = .GlobalEnv)
  message("⏳ loading data...")
  for (i in samples) {
    if (technology == "adaptive") {
      datapath_i <- file.path(datapath.pbl, i)
      files <- list.files(path = datapath_i, pattern = "\\.tsv$", full.names = TRUE)
      files <- files[order(as.numeric(sub(".*d(\\d+)\\.tsv$", "\\1", files)))]
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read.table(file = x, sep = "\t", header = TRUE)
        tcr_data <- subset(tcr_data, subset = tcr_data$sequenceStatus == "In")
        if ("count" %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count / sum(tcr_data$count) * 100
        } else if ("count..reads." %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count..reads. / sum(tcr_data$count..reads.) * 100
        } else if ("count..templates.reads." %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count..templates.reads. / sum(tcr_data$count..templates.reads.) * 100
        } else {
          stop("invalid format")
        }
        tcr_data <- tcr_data %>% filter(freq_in >= min_freq)
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        if ("count" %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count", "freq_in", "vGeneName", "jGeneName"))
        } else if ("count..reads." %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count..reads.", "freq_in", "vGeneName", "jGeneName"))
        } else if ("count..templates.reads." %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count..templates.reads.", "freq_in", "vGeneName", "jGeneName"))
        } else {
          stop("invalid format")
        }
        tcr_data$nucleotide <- paste0(tcr_data$nucleotide, "_", tcr_data$vGeneName, "_", tcr_data$jGeneName)
        tcr_data$vGeneName <- NULL
        tcr_data$jGeneName <- NULL
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        return(tcr_data)
      })
    } else if (technology == "hsd") {
      datapath_i <- file.path(datapath.pbl, i)
      files <- list.files(path = datapath_i, pattern = "\\.xlsx$", full.names = TRUE)
      files <- files[order(as.numeric(sub(".*d(\\d+)\\.xlsx$", "\\1", files)))]
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read_excel(x, col_names = TRUE)
        tcr_data <- as.data.frame(tcr_data)
        tcr_data <- subset(tcr_data, subset = tcr_data$`Frame-info` == "ok")
        tcr_data$freq_in <- tcr_data$"total.nr-reads" / sum(tcr_data$"total.nr-reads") * 100
        # Filter by min_freq_in
        tcr_data <- tcr_data %>% filter(freq_in >= min_freq)
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        tcr_data$`DNA-seq (AA2)` <- paste0(tcr_data$`DNA-seq (AA2)`, "_", tcr_data$`Vsegment-ID`, "_", tcr_data$`Jsegment-ID`)
        tcr_data <- tcr_data %>% select("DNA-seq (AA2)", "AA1: pept-seq Insert", "time", "total.nr-reads", "freq_in")
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        return(tcr_data)
      })
    } else if (technology == "SEQTR") {
      datapath_i <- file.path(datapath.pbl, i)
      files <- list.files(path = datapath_i, pattern = "\\.txt$", full.names = TRUE)
      files <- files[order(as.numeric(sub(".*d(\\d+)\\.txt$", "\\1", files)))]
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read.table(x, sep = "\t", fill = TRUE, na.strings = c("", "NA"))
        colnames(tcr_data) <- c("CDR3_sequence", "Count", "TRBV",	"TRBJ",	"Frame", "CDR3_aaseq", "CDR3_length",	"UMI", "CDR3_nt_nbr")
        tcr_data <- as.data.frame(tcr_data)
        tcr_data <- subset(tcr_data, subset = tcr_data$Frame == "IN")
        tcr_data <- tcr_data %>% filter(Count > 1)
        tcr_data$freq_in <- tcr_data$Count / sum(tcr_data$Count) * 100
        # Filter by min_freq_in
        tcr_data <- tcr_data %>% filter(freq_in >= min_freq)
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        tcr_data$CDR3_sequence <- paste0(tcr_data$CDR3_sequence, "_", tcr_data$TRBV, "_", tcr_data$TRBJ)
        tcr_data$aminoAcid_sequence <- paste0(tcr_data$CDR3_aaseq, "_", tcr_data$TRBV, "_", tcr_data$TRBJ)
        tcr_data <- tcr_data %>% select("CDR3_sequence", "aminoAcid_sequence", "time", "Count", "freq_in")
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        tcr_data <- tcr_data[!is.na(rownames(tcr_data)), ]
        return(tcr_data)
      })
    } else if (technology == "MiXCR") {
      datapath_i <- file.path(datapath.pbl, i)
      files <- list.files(path = datapath_i, pattern = "\\.tsv$", full.names = TRUE)
      files <- files[order(as.numeric(sub(".*d(\\d+)\\.tsv$", "\\1", files)))]
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read.table(file = x, sep = "\t", header = TRUE)
        colnames(tcr_data) <- c("aaSeqCDR3", "vGene", "jGene", "nSamples", "readCountAggregated", "readFractionAggregated", "uniqueUMICountAggregated", "uniqueUMIFractionAggregated", "tagCounts", "bestVGene", "bestJGene")
        tcr_data <- as.data.frame(tcr_data)
        tcr_data <- tcr_data %>% filter(readCountAggregated > 10)
        tcr_data$freq_in <- tcr_data$readFractionAggregated / sum(tcr_data$readFractionAggregated) * 100
        # Filter by min_freq_in
        tcr_data <- tcr_data %>% filter(freq_in >= min_freq)
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        tcr_data$CDR3_sequence <- paste0(tcr_data$aaSeqCDR3, "_", tcr_data$bestVGene, "_", tcr_data$bestJGene)
        tcr_data$aminoAcid_sequence <- paste0(tcr_data$aaSeqCDR3, "_", tcr_data$bestVGene, "_", tcr_data$bestJGene)
        tcr_data <- tcr_data %>% select("CDR3_sequence", "aminoAcid_sequence", "time", "readCountAggregated", "freq_in")
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        return(tcr_data)
      })
    } else {
      message("🚧 invalid technology; only adaptive, hsd, SEQTR or MiXCR data are supported...")
      return(NULL)
    }
    # check if data has been loaded
    if (length(tcr_data_i) == 0) {
      message("🚧 invalid technology or input format; only only adaptive, hsd, SEQTR or MiXCR data are supported...")
      return(NULL)
    }
    names(tcr_data_i) <- tools::file_path_sans_ext(basename(files))
    output_list[[i]] <- tcr_data_i
    data <- Reduce(rbind, tcr_data_i)
    data$count <- NULL
    data <- aggregate(freq_in ~ nucleotide + time, data, sum)
    data <- as.data.frame(tidyr::pivot_wider(data, names_from = time, values_from = freq_in))
    data[is.na(data)] <- min_freq
    freq_data <- data
    timepoint_values <- sub("d", "", colnames(data)[2:ncol(data)])
    timepoint_numeric <- as.numeric(timepoint_values)
    min_timepoint_index <- which.min(timepoint_numeric)
    fc <- data[, 2:ncol(data)] / data[, min_timepoint_index + 1]
    fc <- log2(fc)
    fc$nucleotide <- data$nucleotide
    timepoints <- colnames(fc)[1:(ncol(fc) - 1)]
    fc_long <- as.data.frame(tidyr::pivot_longer(fc, cols = all_of(timepoints), names_to = "timepoints", values_to = "fc"))
    freq_long <- tidyr::pivot_longer(freq_data, cols = all_of(timepoints), names_to = "timepoints", values_to = "freq_in")
    fc_long <- dplyr::left_join(fc_long, freq_long, by = c("nucleotide", "timepoints"))
    fc_output_list[[i]] <- fc_long
    message(paste0(grep(i, samples), "/", length(samples), " ", i))
  }
  assign(output.list.name, output_list, envir = .GlobalEnv)
  assign(fc_output_list_name, fc_output_list, envir = .GlobalEnv)
  message("✅ done!")
}

### function ::plot_mean_counts::
###
plot_mean_counts <- function(data_list,
                             count_type = "count") {
  mean_values <- map(data_list, function(data_i) {
    map(data_i, ~ mean(.x[[count_type]], na.rm = TRUE))
  })
  mean_df <- map_df(names(mean_values), function(hd) {
    tibble(
      HD = hd,
      Timepoint = names(mean_values[[hd]]),
      Value = unlist(mean_values[[hd]])
    )
  }) %>%
    filter(!is.na(Value))
  ggplot(mean_df, aes(x = HD, y = Value)) +
    geom_boxplot(outlier.shape = NA, fill = "lightgray", alpha = 0.3) +
    geom_point(color = "red3", alpha = 0.5, size = 3) +
    theme_light() +
    #scale_y_log10(limits = c(1, 120)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(y = "value", title = paste("Average", count_type, "per sample"))
}

### function ::normalize_data_edgeR::
### 
normalize_data_edgeR <- function(healthy_data = "healthy",
                                 patients_data = "patients") {
  data_hd <- get(healthy_data, envir = .GlobalEnv)
  data_pt <- get(patients_data, envir = .GlobalEnv)
  df_list <- c(data_hd, data_pt)
  # Data normalization using edgeR TMM
  message("Normalizing data using edgeR TMM approach...")
  message("Please, be patient, it could take several minutes... just take a break ☕️")
  all_timepoints <- unique(unlist(lapply(df_list, names)))
  for (i in all_timepoints) {
    union_nucs <- unique(unlist(lapply(df_list, function(patient) {
      if (i %in% names(patient)) {
        patient[[i]]$nucleotide
      } else {
        NULL
      }
    })))
    union_nucs <- sort(union_nucs)
    aligned_counts <- list()
    patient_names <- c()
    for (patient_name in names(df_list)) {
      patient_data <- df_list[[patient_name]]
      if (i %in% names(patient_data)) {
        df <- patient_data[[i]] %>% select(nucleotide, count)
        count_vec <- setNames(rep(0, length(union_nucs)), union_nucs)
        count_vec[df$nucleotide] <- df$count
        aligned_counts[[patient_name]] <- count_vec
        patient_names <- c(patient_names, patient_name)
      }
    }
    count_matrix <- do.call(cbind, aligned_counts)
    dge <- DGEList(counts = count_matrix)
    dge <- calcNormFactors(dge, method = "TMM")
    norm_matrix <- cpm(dge, normalized.lib.sizes = TRUE)
    for (patient_name in patient_names) {
      orig_df <- df_list[[patient_name]][[i]]
      norm_counts <- norm_matrix[match(orig_df$nucleotide, union_nucs), patient_name]
      orig_df$count.norm <- norm_counts
      df_list[[patient_name]][[i]] <- orig_df
    }
  }
  message("✅ Done!")
  return(df_list)
}

### function ::complete_data::
###
complete_data <- function(df_list) {
  message("Completing data for each nucleotide x timepoint...")
  for (i in names(df_list)) {
    clono_data <- Reduce(x = df_list[[i]], f = rbind) %>%
      select(nucleotide, aminoAcid) %>%
      distinct()
    aminoAcid_lookup <- setNames(clono_data$aminoAcid, clono_data$nucleotide)
    clonotypes <- unique(clono_data$nucleotide)
    for (j in names(df_list[[i]])) {
      data_j <- df_list[[i]][[j]]
      missing_clonotypes <- setdiff(clonotypes, data_j$nucleotide)
      if (length(missing_clonotypes) > 0) {
        missing_aminoAcids <- aminoAcid_lookup[missing_clonotypes]
        missing_aminoAcids <- missing_aminoAcids[!is.na(missing_aminoAcids)]
        new_rows <- data.frame(
          nucleotide = names(missing_aminoAcids),
          aminoAcid = missing_aminoAcids,
          time = j,
          count = 0,
          count.norm = as.numeric(0.5),
          freq_in = as.numeric(0)
        )
        df_list[[i]][[j]] <- bind_rows(df_list[[i]][[j]], new_rows)
      }
    }
    message(paste0(grep(i, names(df_list)), "/", length(names(df_list)), " ", i))
  }
  return(df_list)
}

### function ::calculate_variability::
### Removes natural variability in pt based on fold-change dynamics from hd, with three selectable modes (mean, 90th, extreme).
calculate_variability <- function(input.data) {
  message("⏳ calculating natural variability...")
  input <- get(input.data, envir = .GlobalEnv)
  input <- complete_data(input)
  if (exists("natural_var", envir = .GlobalEnv)) {
    natural_var <<- get("natural_var", envir = .GlobalEnv)
  } else {
    natural_var <<- list()
  }
  # calculate the rate of the triangle (difference in time x difference in freq).
  ls_slope <- list()
  for (i in names(input)) {
    df <- Reduce(rbind, input[[i]]) %>% mutate(time = as.numeric(gsub("d", "", time)))
    df <- df %>% mutate(pseudotime = time/max(time))
    df <- df %>%
      group_by(nucleotide) %>%
      arrange(pseudotime, .by_group = TRUE) %>%
      mutate(comparison = paste(lag(time), time, sep = "-"),
             base = pseudotime - lag(pseudotime),
             height = count.norm / lag(count.norm),
             log_height = log2(height),
             rate = log_height/base) %>%
      filter(!is.na(rate), rate != -Inf)
    
    ls_slope[[i]] <- df
    message(paste0("📐 slope calculated for each clonotype in ", i))
  }
  # merge all patient data.
  distr <- Reduce(rbind, ls_slope)
  natural_var[["data"]] <<- distr
  # fit normal distribution to rate values.
  mean_slope <- mean(distr$rate)
  sd_slope <- sd(distr$rate)
  # store normality parameters.
  natural_var[["mu"]] <<- mean_slope
  natural_var[["sigma"]] <<- sd_slope
  # plot
  plot <- ggplot(distr, aes(x = rate)) +
    geom_histogram(aes(y = ..density..), 
                   bins = 500, 
                   fill = "lightgray", 
                   color = "black") +
    stat_function(fun = dnorm, 
                  args = list(mean = mean(distr$rate, na.rm = TRUE), 
                              sd = sd(distr$rate, na.rm = TRUE)),
                  color = "red3",
                  linewidth = 0.5) +
    labs(title = "slope distribution",
         x = "slope",
         y = "density") +
    theme_light() + 
    theme(aspect.ratio = 0.5)
  natural_var[["distribution_plot"]] <<- plot
  message("✅ done!")
  return(natural_var[["distribution_plot"]])
}

### function ::calculate_statistics::
### Computes p-values for trajectory jumps between time points, identifies significant trajectories and categorized trajectory complexity into low, int, high or ext.
calculate_statistics <- function(input.data,
                                 day.limit = NULL,
                                 p.val = 0.05,
                                 plot = FALSE) {
  # create statistics to store data.
  if (exists("statistics", envir = .GlobalEnv)) {
    statistics <<- get("statistics", envir = .GlobalEnv)
  } else {
    statistics <<- list()
  }
  # start the analysis.
  message("⏳ calculating statistics...")
  input <- get(input.data, envir = .GlobalEnv)
  input <- complete_data(input)
  ls_slope <- list()
  ls_number <- list()
  # calculate the rate of the triangle (difference in time x difference in freq).
  for (i in names(input)) {
    df <- Reduce(rbind, input[[i]]) %>% mutate(time = as.numeric(gsub("d", "", time)))
    # apply day.limit if specified.
    if (!is.null(day.limit)) {
      message(paste0("✂️ applying day limit to ", i, "..."))
      df <- df %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
    }
    # calculate mean, without considering the imputed min values
    filter_counts <- mean(df$count.norm[df$count.norm != 0.5])
    filter_counts <- 2*filter_counts
    #filter_counts <- (df$count.norm[df$count.norm != 0.5])
    #filter_counts <- filter_counts[[4]]
    df <- df %>%
      group_by(nucleotide) %>%
      filter(sum(count.norm > filter_counts) >= 2) %>%
      ungroup()
    message("removing clonotypes with normalized counts below twice the mean at two time points...")
    df <- df %>% mutate(pseudotime = time/max(time))
    df <- df %>%
      group_by(nucleotide) %>%
      arrange(pseudotime, .by_group = TRUE) %>%
      mutate(comparison = paste(lag(time), time, sep = "-"),
             base = pseudotime - lag(pseudotime),
             height = count.norm / lag(count.norm),
             log_height = log2(height),
             rate = log_height/base) %>%
      filter(!is.na(rate), rate != -Inf)
    ls_slope[[i]] <- df
    message(paste0("📐 slope calculate for each clonotype in ", i, " "))
  }
  message("🔍 identifying significant expansions/contractions based on ref data...")
  for (i in names(input)) {
    df <- ls_slope[[i]] %>%
      mutate(pval = 2 * (1 - pnorm(rate, mean = natural_var$mu, sd = natural_var$sigma))) %>%
      mutate(pval = ifelse(pval > 1, 1, pval)) %>%
      mutate(pval_adj = p.adjust(pval, method = "BH"))
    # merge both data.
    df <- df %>% mutate(sig = ifelse(pval_adj <= p.val, "yes", "no"))
    df$count.norm <- NULL
    ls_slope[[i]] <- df
    ls_number[[i]] <- length(unique(subset(df, pval_adj <= p.val)$nucleotide))
    # plot adj p-value distribution.
    if (plot) {
      hist(df$pval_adj, prob = TRUE, main = i, xlab = "adjusted p-value", col = "lightgray", border = "black", breaks = 50)
      abline(v = p.val, col = "red3", lty = 2)
    }
    # store data on statistics.
    statistics[[i]] <<- ls_slope[[i]]
  }
  # print number of clonotypes.
  for (i in names(input)) {
    message(paste0("🔦 A total of ", ls_number[[i]], " clonotypes identified in ", i, " for p-value < ", p.val))
  }
  for (i in names(patients)) {
    for (j in names(patients[[i]])) {
      patients[[i]][[j]]$count.norm <<- NULL
    }
  }
  message("✅ done!")
}

### function ::identify_trajectories::
### 
identify_trajectories <- function(
    input.data,
    day.limit = NULL,
    true.window = FALSE,
    min.counts = 2,
    min.window = 2,
    filter.dropouts = FALSE,
    filter.na.valley = FALSE,
    na.valley = 2,
    traj.plot = TRUE,
    traj.filter = FALSE
) {
  if (is.null(natural_var)) {
    if (exists("natural_var", envir = .GlobalEnv)) {
      natural_var <- get("natural_var", envir = .GlobalEnv)
      natural_var$data$count.norm <- NULL
    } else {
      stop("🚧 please, run calculate_variability() function before...")
    }
  }
  if (is.null(natural_var)) {
    if (exists("statistics", envir = .GlobalEnv)) {
      statistics <- get("statistics", envir = .GlobalEnv)
    } else {
      stop("🚧 please, run calculate_statistics() function before...")
    }
  }
  # Filter 0: natural variability.
  message("⏳ identifying non-neutral clonotypes...")
  input <- get(paste0(input.data, "_fc"))
  
  sig_list <- list()
  
  for (i in names(input)) {
    data_i <- input[[i]]
    if (!is.null(day.limit)) {
      message(paste0("✂️ applying day limit to ", i, "..."))
      data_i <- data_i %>% filter(as.numeric(gsub("d", "", timepoints)) <= day.limit)
    }
    # statistically significant.
    if (!is.null(statistics[[i]]) && "sig" %in% colnames(statistics[[i]])) {
      sig_list[[i]] <- unique(subset(statistics[[i]], subset = sig == "yes")$nucleotide)
      input[[i]]$non_neutral <- ifelse(input[[i]]$nucleotide %in% sig_list[[i]], "Y", "N")
      input[[i]] <- subset(input[[i]], subset = non_neutral == "Y")
      } else {
        stop(paste0("🚨 error: 'sig' column not found in ", input.data, "; please run calculate_statistics()."))
      }
    }
  # Filter 1: true.window.
  if (true.window) {
    message(paste0("🪟 applying true.window for ", min.counts, " counts x ", min.window, " timepoints..."))
    # function ::apply_true_window:::
    #apply_true_window <- function(x) {
    #  sum(rle(x)$lengths[rle(x)$values & rle(x)$lengths >= min.window]) >= 1
    #}
    apply_true_window <- function(x) {
      return(as.numeric(sum(rle(x)$lengths[rle(x)$values & rle(x)$lengths >= min.window]) >= 1))
    }
    filtered_tcrb <- list()
    data <- list()
    raw_data <- get(input.data)
    for (i in names(input)) {
      data_i <- Reduce(rbind, raw_data[[i]])
      if (!is.null(day.limit)) {
        data_i <- data_i %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
      }
      data_i$freq_in <- NULL
      data_i$aminoAcid <- NULL
      data_i <- subset(data_i, nucleotide %in% unique(input[[i]]$nucleotide))
      # new
      data_i <- data_i |> 
        distinct(nucleotide, time, .keep_all = TRUE) |> 
        pivot_wider(names_from = time, values_from = count)
      # old
      #data_i <- as.data.frame(pivot_wider(data_i,
      #                                    names_from = time,
      #                                    values_from = count))
      # reorder columns in ascending order to allow negative days, eg d-9.
      time_columns <- grep("^d", names(data_i), value = TRUE)
      ordered_time_columns <- time_columns[order(as.numeric(gsub("d", "", time_columns)))]
      data_i <- data_i[, c("nucleotide", ordered_time_columns)]
      
      data_i[is.na(data_i)] <- 0
      data[[i]] <- data_i
      consec_true <- data_i[, 2:ncol(data_i)] >= min.counts
      consec_true <- t(rollapply(t(consec_true), width = min.window, FUN = apply_true_window, align = "center", partial = TRUE))
      consec_true <- as.data.frame(consec_true)
      consec_true$filter <- ifelse(rowSums(consec_true) > 0, "Y", "N")
      consec_true$nucleotide <- data_i$nucleotide
      consec_true <- subset(consec_true, filter == "Y")
      filtered_tcrb[[i]] <- unique(consec_true$nucleotide)
    }
    for (i in names(input)) {
      input[[i]]$true.window <- ifelse(input[[i]]$nucleotide %in% filtered_tcrb[[i]], "Y", "N")
    }
  }
  # Filter 2: filter.dropouts.
  if (filter.dropouts) {
    message("🕳 applying consecutive dropouts filter...")
    has_few_timepoints <- function(df) {
      num_timepoints <- ncol(df) - 1
      return(num_timepoints <= 5)
    }
    # function ::apply_consec_drops::
    apply_consec_drops <- function(df) {
      df %>%
        rowwise() %>%
        mutate(
          consec_drop = {
            values <- c_across(starts_with("d"))
            n <- length(values)
            is_value <- !is.na(values)
            has_alternating <- FALSE
            for (i in 3:(n - 3)) {
              if (
                is_value[i - 2] & is_value[i - 1] & !is_value[i] &
                is_value[i + 1] & !is_value[i + 2] & is_value[i + 3]
              ) {
                has_alternating <- TRUE
                break
              }
            }
            if (has_alternating) "Y" else "N"
          }
        ) %>%
        ungroup()
    }
    for (i in names(input)) {
      data_i <- subset(data[[i]], subset = nucleotide %in% filtered_tcrb[[i]])
      if (has_few_timepoints(data_i)) {
        message(paste0("⚠️ skipping dropouts filter for ", i, " due to low number of timepoints"))
        next
      }
      data_i <- data_i %>% mutate(across(starts_with("d"), ~ na_if(., 0)))
      consec_drop <- apply_consec_drops(data_i)
      consec_drop <- as.data.frame(consec_drop)
      consec_drop <- subset(consec_drop, subset = consec_drop == "N")
      input[[i]]$consec_drop <- ifelse(input[[i]]$nucleotide %in% consec_drop$nucleotide, "N", "Y")
    }
  }
  # Filter 3: filter.na.valley(s).
  if (filter.na.valley) {
    message(paste0("🏜 applying NA valley filter..."))
    # function ::apply_filter_valley::
    apply_filter_valley <- function(df) {
      df %>%
        rowwise() %>%
        mutate(
          max_na_gap = {
            values <- c_across(starts_with("d"))
            max_gap <- 0
            current_gap <- 0
            counting_na <- FALSE
            for (value in values) {
              if (!is.na(value)) {
                if (counting_na) {
                  max_gap <- max(max_gap, current_gap)
                }
                current_gap <- 0
                counting_na <- TRUE
              } else if (counting_na) {
                current_gap <- current_gap + 1
              }
            }
            max_gap
          }
        ) %>%
        ungroup()
    }
    for (i in names(input)) {
      data_i <- subset(data[[i]], subset = nucleotide %in% filtered_tcrb[[i]])
      num_timepoints <- ncol(data_i) - 1
      if (num_timepoints <= na.valley) {
        message(paste0("⚠️ skipping dropouts filter for ", i, " due to low number of timepoints"))
        next
      }
      data_i <- data_i %>% mutate(across(starts_with("d"), ~ na_if(., 0)))
      filtered_valley <- apply_filter_valley(data_i)
      filtered_valley <- subset(filtered_valley, max_na_gap <= na.valley)
      input[[i]]$valley_filter <- ifelse(input[[i]]$nucleotide %in% filtered_valley$nucleotide, "N", "Y")
    }
  }
  # Plot.
  if(traj.plot) {
    traj_fc <<- list()
    for (i in names(input)) {
      data_i <- input[[i]]
      if (!is.null(day.limit)) {
        data_i <- data_i %>% filter(as.numeric(gsub("d", "", timepoints)) <= day.limit)
      }
      if (!"true.window" %in% colnames(data_i)) data_i$true.window <- "x"
      if (!"consec_drop" %in% colnames(data_i)) data_i$consec_drop <- "x"
      if (!"valley_filter" %in% colnames(data_i)) data_i$valley_filter <- "x"
      data_i <- data_i %>%
        mutate(facet_category = paste(true.window, consec_drop, valley_filter, sep = "_")) %>%
        select(nucleotide, timepoints, fc, freq_in, non_neutral, true.window, consec_drop, valley_filter, facet_category)
      data_i <- subset(data_i, subset = facet_category != "N_Y_Y")
      data_i <- subset(data_i, subset = facet_category != "N_Y_x")
      data_i <- subset(data_i, subset = facet_category != "N_x_Y")
      data_i <- subset(data_i, subset = facet_category != "N_N_Y")
      data_i <- subset(data_i, subset = facet_category != "N_x_x")
      data_i <- data_i %>%
        mutate(facet_category = recode(facet_category,
                                       "Y_N_N" = "Non-neutral clonotypes",
                                       "Y_x_N" = "Non-neutral clonotypes",
                                       "Y_N_x" = "Non-neutral clonotypes",
                                       "Y_x_x" = "Non-neutral clonotypes",
                                       "Y_N_Y" = "Eliminated due to NA valleys",
                                       "Y_x_Y" = "Eliminated due to NA valleys",
                                       "Y_Y_N" = "Eliminated due to dropouts",
                                       "Y_Y_x" = "Eliminated due to dropouts",
                                       "Y_Y_Y" = "Eliminated due to NA valley & dropouts",
                                       "x_x_x" = "Non-neutral based on natural variability removal"))
      # labels
      facet_labels <- data_i %>% filter(non_neutral == "Y") %>% group_by(facet_category) %>% summarise(sig = n() / length(unique(data_i$timepoints)))
      facet_labels <- facet_labels %>% mutate(label = paste0(sig, "/", length(sig_list[[i]])))
      statistics_i <- statistics[[i]] %>% select(nucleotide, time, pval_adj) 
      statistics_i <- statistics_i %>%
        group_by(nucleotide) %>%
        summarise(min_pval_adj = min(pval_adj, na.rm = TRUE)) %>%
        ungroup() %>% filter(nucleotide %in% sig_list[[i]])
      data_i <- data_i %>% left_join(statistics_i, by = "nucleotide")
      plot_i <- ggplot(data_i,
                       aes(x = as.numeric(gsub("d", "", timepoints)),
                           y = log10(freq_in),
                           group = nucleotide,
                           color = min_pval_adj)
      ) +
        geom_line(alpha = 0.8, linewidth = 0.2) +
        facet_grid(~facet_category, scales = "free") + 
        theme_minimal() +
        scale_color_gradientn(
          colors = c("#B2182B", "#EF8A62", "#F7F7F7", "#67A9CF", "#2166AC"),
          values = scales::rescale(c(0, 0.005, 0.025, 0.045, 0.05)),
          name = "adj pval",
          limits = c(0, 0.05),
          breaks = seq(0, 0.05, by = 0.01),
          labels = seq(0, 0.05, by = 0.01)
        ) +
        scale_x_continuous() +
        scale_y_continuous(
          breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, 100))),
          labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", "100")),
          limits = c(log10(min_freq), log10(100))
        ) +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.background = element_blank(),
              panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
              aspect.ratio = 0.5,
              axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "right") +
        geom_text(data = facet_labels,
                  aes(x = 25,
                      y = log10(5),
                      label = label),
                  inherit.aes = FALSE,
                  hjust = 0,
                  position = position_nudge(y = 0), size = 3) +
        labs(
          title = i,
          x = "days",
          y = "frequency"
        )
      traj_fc[[i]] <<- plot_i
    }
  }
  # Filter trajectories
  pbl_filt <<- list()
  if (!traj.filter) {
    message("📁 storing trajectories in pbl_fil...")
    for (i in names(input)) {
      pbl_filt[[i]] <<- input[[i]]
    }
    message("✅ done!")
    print(traj_fc)
  } else {
    message("📁 storing filtered trajectories in pbl_fil...")
    for (i in names(input)) {
      existing_filters <- c("non_neutral", "true.window", "consec_drop", "valley_filter")
      available_filters <- existing_filters[existing_filters %in% colnames(input[[i]])]
      filter_condition <- paste0(
        c("non_neutral == 'Y'", "true.window == 'Y'", "consec_drop == 'N'", "valley_filter == 'N'")[
          c("non_neutral", "true.window", "consec_drop", "valley_filter") %in% available_filters
        ], collapse = " & "
      )
      if (length(available_filters) > 0) {
        if (all(c("non_neutral", "true.window") %in% available_filters) & !("consec_drop" %in% available_filters) & !("valley_filter" %in% available_filters)) {
          message(paste0(i, ": only non_neutral and true.window filters applied."))
        } else if (all(c("non_neutral", "true.window", "consec_drop") %in% available_filters) & !("valley_filter" %in% available_filters)) {
          message(paste0(i, ": only non_neutral, true.window, and consec_drop filters applied."))
        } else if (all(c("non_neutral", "true.window", "consec_drop", "valley_filter") %in% available_filters)) {
          message(paste0(i, ": all filters successfully applied."))
        }
        filtered_data <- input[[i]]
        if (!is.null(day.limit)) {
          filtered_data <- filtered_data %>% filter(as.numeric(gsub("d", "", timepoints)) <= day.limit)
        }
        filtered_data <- filtered_data %>% filter(eval(parse(text = filter_condition)))
        # Add the count info to pbl_filt data.
        timepoints <- unique(filtered_data$timepoints)
        patient_data <- data.frame()
        for (j in timepoints) {
          if (j %in% names(get(paste0(input.data))[[i]])) {
            temp_df <- get(paste0(input.data))[[i]][[j]]
            patient_data <- bind_rows(patient_data, temp_df)
          }
        }
        colnames(patient_data) <- c("nucleotide", "aminoAcid", "timepoints", "count", "freq_in")
        filtered_data <- filtered_data %>%
          left_join(patient_data %>% select(nucleotide, timepoints, count), 
                    by = c("nucleotide", "timepoints")) %>%
          mutate(count = ifelse(is.na(count), 0, count))
        if (nrow(filtered_data) > 0) {
          pbl_filt[[i]] <<- filtered_data
        } else {
          message(paste0("☹️ No non-neutral clonotypes identified for ", i))
        }
      } else {
        warning(paste0("☹️ No valid filters found for ", i, "; skipping dataset."))
      }
    }
    message("✅ done!")
    print(traj_fc)
  }
}

# function ::calculate_k_means::
calculate_k_means <- function(input.data,
                              sample,
                              day.limit = NULL,
                              mode = "fc",
                              weight.frequency = 0, 
                              weight.jumps = 0.5,
                              weight.na.pattern = 0.25, 
                              weight.contractions = 0.25,
                              biopsy = NULL,
                              max.biopsy = 50,
                              K.max = 20,
                              min_freq = 0.0001) {
  raw_data_i <- get(input.data, envir = .GlobalEnv)
  message("⏳ calculating k...")
  # tumor traj clustering.
  if(!is.null(biopsy)){
    raw_data_i <- raw_data_i[[sample]][[biopsy]]
    # check if time column starts with "d".
    if (!all(grepl("^d", raw_data_i$time))) {
      raw_data_i$time <- ifelse(grepl("^d", raw_data_i$time), raw_data_i$time, paste0("d", raw_data_i$time))
    }
    # apply day limit.
    if (!is.null(day.limit)) {
      raw_data_i <- raw_data_i %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
    }
    # filter by top n.
    top.n.clonotypes <<- raw_data_i %>% 
      group_by(nucleotide) %>%
      summarise(max_log10value = max(log10value)) %>%
      arrange(desc(max_log10value)) %>%
      slice_head(n = max.biopsy) %>%
      pull(nucleotide)
    raw_data_i <- raw_data_i %>% filter(nucleotide %in% top.n.clonotypes)
    raw_data_i <- raw_data_i %>% select(nucleotide, freq_in, time)
    raw_freq_i <- raw_data_i %>% tidyr::pivot_wider(names_from = c(time),
                                                    values_from = freq_in,
                                                    values_fill = min_freq) %>% as.data.frame()
  } else {
    # non_neutral traj clustering.
    raw_data_i <- raw_data_i[[sample]]
    timepoints <- names(raw_data_i)
    raw_data_i <- Reduce(rbind, raw_data_i) %>% filter(nucleotide %in% as.character(pbl_filt[[sample]]$nucleotide))
    
    # completing missing time points in data with freq_in = min_freq.
    all_combinations <- expand.grid(nucleotide = unique(raw_data_i$nucleotide), time = timepoints)
    raw_data_i <- full_join(raw_data_i, all_combinations, by = c("nucleotide", "time"))
    raw_data_i <- raw_data_i %>% mutate(count = ifelse(is.na(count), 0, count),
                                        freq_in = ifelse(is.na(freq_in), min_freq, freq_in)
    )
    # check for day limit.
    if (!is.null(day.limit)) {
      raw_data_i <- raw_data_i %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
    }
    raw_freq_i <- raw_data_i %>% select(-aminoAcid, -count) %>%
      tidyr::pivot_wider(names_from = time,
                         values_from = freq_in,
                         values_fill = min_freq) %>%
      as.data.frame()
  }
  # reorder columns by time to manage cases where the first sample is, e.g. d-9.
  raw_freq_i <- raw_freq_i %>%
    select(-matches("^d")) %>%
    bind_cols(raw_freq_i %>%
                select(matches("^d")) %>%
                select(order(as.numeric(gsub("d", "", names(.))))))
  d0 <- raw_freq_i %>% select(matches("^d")) %>% .[, 1]
  # Step 1: distance matrix from raw_freq_i.
  dist_raw_data <- raw_freq_i %>% select(matches("^d")) %>% scale()
  dist_raw_data <- dist(dist_raw_data, method = "euclidean")
  dist_raw_data_weighted <- dist_raw_data * weight.frequency
  # Step 2: jumps between time-points.
  message(paste0("Ⓜ️ clustering based on ", mode, "..."))
  if (mode == "slope") {
    unique_times <- raw_freq_i %>%
      select(starts_with("d")) %>%
      names() %>%
      str_remove("d") %>%
      as.numeric() %>%
      sort()
    jump_data_diff <- raw_freq_i %>%
      pivot_longer(cols = starts_with("d"), names_to = "time", values_to = "freq_in") %>%
      mutate(time = as.numeric(str_remove(time, "d"))) %>%
      group_by(nucleotide) %>%
      arrange(time, .by_group = TRUE) %>%
      mutate(comparison = paste(lag(time), time, sep = "-"),
             base = time - lag(time),
             height = lag(freq_in) - freq_in,
             jump = height/base) %>%
      filter(!is.na(jump))
    jump_data_diff <- jump_data_diff %>%
      mutate(time = paste0("d", time)) %>%
      select(nucleotide, time, jump) %>%
      pivot_wider(names_from = time, values_from = jump, values_fill = 0)
    for (t in paste0("d", unique_times)) {
      if (!(t %in% names(jump_data_diff))) {
        jump_data_diff[[t]] <- 0
      }
    }
    jump_data_diff <- jump_data_diff %>%
      select(nucleotide, all_of(paste0("d", unique_times))) %>% as.data.frame()
    dist_jump_diff <- jump_data_diff %>% select(matches("^d")) %>% scale()
    dist_jump_diff <- dist(dist_jump_diff, method = "euclidean")
    dist_jump_diff_weighted <- dist_jump_diff * weight.jumps
  } else {
    log2_data_diff <- raw_freq_i %>%
      mutate(across(matches("^d"), ~ log2(. / d0), .names = "log2_{col}"))
    log2_columns <- names(log2_data_diff)[grepl("^log2_d", names(log2_data_diff))]
    for (i in 2:length(log2_columns)) {
      log2_data_diff <- log2_data_diff %>%
        mutate(
          !!paste0("diff_", gsub("log2_", "", log2_columns[i])) := 
            .[[log2_columns[i]]] - .[[log2_columns[i - 1]]]
        )
    }
    log2_diff_scaled <- scale(log2_data_diff %>% select(starts_with("diff_")))
    diff_d0 <- rep(0, nrow(log2_diff_scaled))
    dist_jump_diff <- cbind(diff_d0, log2_diff_scaled)
    dist_jump_diff <- dist(dist_jump_diff, method = "euclidean")
    dist_jump_diff_weighted <- dist_jump_diff * weight.jumps
  }
  # Step 3: NA pattern. Codification of NA matrix: 0 = non-NA, 1 first NA, 2 second NA, and so on. 
  binary_matrix <- raw_freq_i %>% select(matches("^d"))
  counter_matrix <- matrix(0, nrow = nrow(binary_matrix), ncol = ncol(binary_matrix))
  for (i in 1:nrow(binary_matrix)) {
    counter <- 0
    for (j in 1:ncol(binary_matrix)) {
      if (binary_matrix[i, j] == min_freq) {
        counter <- counter + 1
        counter_matrix[i, j] <- counter
      } else {
        counter <- 0
        counter_matrix[i, j] <- 0
      }
    }
  }
  colnames(counter_matrix) <- colnames(binary_matrix)
  binary_matrix <- as.data.frame(counter_matrix)
  dist_binary_matrix <- dist(binary_matrix, method = "euclidean")
  dist_binary_matrix_weighted <- dist_binary_matrix * weight.na.pattern
  # Step 4: Identification of contracting trajectories compared to baseline.
  binary_contractions <- raw_freq_i %>% select(-nucleotide)
  for (i in 1:nrow(binary_contractions)) {
    if (binary_contractions[i, 1] > binary_contractions[i, 2]) {
      binary_contractions[i, 1] <- 1
      binary_contractions[i, 2:ncol(binary_contractions)] <- 0
    } else {
      binary_contractions[i, ] <- 0
    }
  }
  dist_binary_contractions <- dist(binary_contractions, method = "euclidean")
  dist_binary_contractions_weighted <- dist_binary_contractions * weight.contractions
  # Combine all.
  dist_matrix_combined <<- dist_raw_data_weighted + 
    dist_jump_diff_weighted + 
    dist_binary_matrix_weighted + 
    dist_binary_contractions_weighted
  # K.max-means plot.
  set.seed(123)
  gap_stat <- cluster::clusGap(as.matrix(dist_matrix_combined), FUN = kmeans, K.max = (min(K.max, nrow(as.matrix(as.dist(dist_matrix_combined))))-1), B = 50)
  message("✅ done!")
  message("⏩ now, run separate_trajectories() for desired k...")
  print(fviz_gap_stat(gap_stat, linecolor = "black") + ggtitle(sample))
}

# function ::generate_trajectory_plots::
separate_trajectories <- function(input.data,
                                  k,
                                  n.col = 5,
                                  max.freq = 100,
                                  sample,
                                  day.limit = NULL,
                                  biopsy = NULL
) {
  message("⏳ separating trajectories...")
  if (!exists("dist_matrix_combined", envir = .GlobalEnv)) {
    message("⚠️ please, run the calculate_k_means() function before...")
    return(NULL)
  } else {
    dist_matrix_combined <- get("dist_matrix_combined", envir = .GlobalEnv)
  }
  
  # If biopsy provided.
  if(!is.null(biopsy)){
    data <- get(input.data, envir = .GlobalEnv)[[sample]][[biopsy]]
    if (!all(grepl("^d", data$time))) {
      data$time <- ifelse(grepl("^d", data$time), data$time, paste0("d", data$time))
    }
    if (!is.null(day.limit)) {
      data <- data %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
    }
    top_n_nucleotides <- get("top.n.clonotypes", envir = .GlobalEnv)
    data <- data %>% filter(nucleotide %in% top_n_nucleotides)
    data <- data %>% select(-aminoAcid, -log10value)
    metadata <- data %>% select(-time, -count, -freq_in, -traj) %>% distinct()
    data <- data %>% select(nucleotide, time, freq_in)
    data <- data %>% tidyr::pivot_wider(names_from = c(time),
                                        values_from = freq_in,
                                        values_fill = min_freq) %>% as.data.frame()
    data <- data %>% left_join(metadata, by = "nucleotide")
    
  } else {
    data <- get(input.data, envir = .GlobalEnv)[[sample]]
    data <- Reduce(rbind, data) %>% filter(nucleotide %in% as.character(pbl_filt[[sample]]$nucleotide))
    
    # filter day limit
    if (!is.null(day.limit)) {
      data <- data %>% filter(as.numeric(gsub("d", "", time)) <= day.limit)
    }
    data <- data %>%
      select(-aminoAcid, -count) %>%
      pivot_wider(
        names_from = time,
        values_from = freq_in,
        values_fill = min_freq
      ) %>%
      as.data.frame()
  }
  hc <- hclust(dist(dist_matrix_combined), method = "ward.D2")
  clusters <- cutree(hc, k = k)
  data$cluster <- as.factor(clusters)
  if (!is.null(biopsy)) {
    data_tumor_traj <- data %>% select(nucleotide, biopsy, starts_with("d"), cluster)
    if (!exists("pbl_tumor_traj", envir = .GlobalEnv)) {
      assign("pbl_tumor_traj", list(), envir = .GlobalEnv)
    }
    pbl_tumor_traj <- get("pbl_tumor_traj", envir = .GlobalEnv)
    if (is.null(pbl_tumor_traj[[sample]])) {
      pbl_tumor_traj[[sample]] <- list()
    }
    pbl_tumor_traj[[sample]][[biopsy]] <- data_tumor_traj
    assign("pbl_tumor_traj", pbl_tumor_traj, envir = .GlobalEnv)
  }
  # plot ::plot_dend::
  plot_dend <- fviz_dend(hc,
                         k = k,
                         show_labels = FALSE,
                         rect = TRUE,
                         rect_fill = TRUE,
                         cex = 2,
                         color_labels_by_k = T,
                         k_colors = "Set1",
                         type = "phylogenic",
                         main = sample) +
    theme(aspect.ratio = 1)
  
  if (k <= 9) {
    cols <- brewer.pal(k, "Set1")
  } else {
    cols <- colorRampPalette(brewer.pal(9, "Set1"))(k)
  }
  
  if (k <= 9) {
    cols <- brewer.pal(k, "Set1")
  } else {
    cols <- colorRampPalette(brewer.pal(9, "Set1"))(k)
  }
  dend_data <- attributes(plot_dend)$dendrogram
  tree_order <- order.dendrogram(dend_data)
  clusters <- factor(clusters, levels = unique(clusters[tree_order]))
  names(cols) = unique(clusters[tree_order])
  # plot ::plot_traj::
  long_data <- data %>%
    pivot_longer(
      cols = matches("^d"),
      names_to = "time",
      values_to = "value"
    ) %>%
    mutate(
      time = as.numeric(gsub("^d", "", time)),
      log10value = log10(value)
    )
  # update ::input.data:: with clusters
  if (!is.null(day.limit)) {
    long_data <- long_data %>% mutate(time = as.numeric(gsub("^d", "", time))) %>% filter(time <= day.limit)
  }
  if(is.null(biopsy)){
    nucleotide_cluster <- long_data %>% select(nucleotide, cluster) %>% distinct()
    colnames(nucleotide_cluster) <- c("nucleotide", "traj")
    data_update <- get(input.data, envir = .GlobalEnv)[[sample]]
    for (i in seq_along(data_update)) {
      data_update[[i]] <- data_update[[i]] %>% left_join(nucleotide_cluster, by = "nucleotide")
    }
    output_data <<- list()
    output_data[[sample]] <<- data_update
  }
  # plot traj by cluster.
  plot_traj <- ggplot() +
    geom_line(data = long_data, 
              aes(x = time, y = log10value, group = nucleotide, color = cluster), 
              linewidth = 0.4, alpha = 0.6) +
    theme_minimal() +
    scale_x_continuous() +
    scale_y_continuous(
      breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, max.freq))),
      labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", as.character(max.freq))),
      limits = c(log10(min_freq), log10(max.freq))
    ) +
    labs(y = "frequency") +
    scale_color_manual(values = cols) +
    facet_wrap(~cluster, ncol = n.col) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
      aspect.ratio = 0.5
    ) +
    ggtitle(ifelse(!is.null(biopsy), paste0(sample, " / ", biopsy), paste0(sample, " / non_neutral"))) +
    geom_vline(data = long_data, aes(xintercept = time), linetype = "dashed", color = "gray", alpha = 0.5)
  # save the plots.
  if(!is.null(biopsy)){
    if (!exists("traj_tumor_pbl", envir = .GlobalEnv)) {
      assign("traj_tumor_pbl", list(), envir = .GlobalEnv)
    }
    traj_tumor_pbl <- get("traj_tumor_pbl", envir = .GlobalEnv)
    traj_tumor_pbl[[sample]][[biopsy]] <- wrap_plots(plot_dend,
                                                     plot_traj,
                                                     ncol = 2,
                                                     widths = c(1, 3))
    assign("traj_tumor_pbl", traj_tumor_pbl, envir = .GlobalEnv)
    message("✅ done!")
    return(traj_tumor_pbl[[sample]][[biopsy]])
  } else {
    if (!exists("traj_non_neutral", envir = .GlobalEnv)) {
      assign("traj_non_neutral", list(), envir = .GlobalEnv)
    }
    traj_non_neutral <- get("traj_non_neutral", envir = .GlobalEnv)
    traj_non_neutral[[sample]] <- wrap_plots(plot_dend,
                                             plot_traj,
                                             ncol = 2,
                                             widths = c(1, 3))
    assign("traj_non_neutral", traj_non_neutral, envir = .GlobalEnv)
    message("✅ done!")
    return(traj_non_neutral[[sample]])
  }
}

# function ::identify_trajectories::
highligth_trajectories <- function(input.data,
                                   sample,
                                   biopsy = NULL,
                                   max.biopsy = 100,
                                   max.freq = 100,
                                   day.limit = NULL,
                                   filter.traj = TRUE,
                                   top.n.clonotypes = 250,
                                   technology = "adaptive",
                                   datapath.annot, 
                                   plot.time = TRUE,
                                   icb.1 = NULL,
                                   icb.2 = NULL,
                                   Tx.1 = NULL,
                                   Tx.2 = NULL,
                                   Tx.3 = NULL,
                                   biopsy.day = NULL
) {
  message("🎨 painting trajectories...")
  # MODULE 1: highlight non_neutral clonotypes found in biopsy.
  if(is.null(biopsy)) {
    
    if (!exists("output_data", envir = .GlobalEnv)) {
      message("⚠️ please run separate_trajectories() before...")
      return(NULL)
    } else {
      long_data <- get("output_data", envir = .GlobalEnv)[[sample]]
      #rm("output_data", envir = .GlobalEnv)
    }
    long_data <- Reduce(rbind, long_data) %>% distinct()
    
    # MODULE 1 --> load tumor data.
    
    if (technology == "adaptive") {
      datapath_i <- file.path(datapath.annot, sample)
      files <- list.files(path = datapath_i, pattern = "\\.tsv$", full.names = TRUE)
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read.table(file = x, sep = "\t", header = TRUE)
        tcr_data <- subset(tcr_data, subset = tcr_data$sequenceStatus == "In")
        if ("count" %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count / sum(tcr_data$count) * 100
        } else if ("count..reads." %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count..reads. / sum(tcr_data$count..reads.) * 100
        } else if ("count..templates.reads." %in% colnames(tcr_data)) {
          tcr_data$freq_in <- tcr_data$count..templates.reads. / sum(tcr_data$count..templates.reads.) * 100
        } else {
          stop("invalid format")
        }
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        if ("count" %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count", "freq_in", "vGeneName", "jGeneName"))
        } else if ("count..reads." %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count..reads.", "freq_in", "vGeneName", "jGeneName"))
        } else if ("count..templates.reads." %in% colnames(tcr_data)) {
          tcr_data <- subset(tcr_data, select = c("nucleotide", "aminoAcid", "time", "count..templates.reads.", "freq_in", "vGeneName", "jGeneName"))
        } else {
          stop("invalid format")
        }
        tcr_data$nucleotide <- paste0(tcr_data$nucleotide, "_", tcr_data$vGeneName, "_", tcr_data$jGeneName)
        tcr_data$vGeneName <- NULL
        tcr_data$jGeneName <- NULL
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        return(tcr_data)
      })
    } else if (technology == "hsd") {
      datapath_i <- file.path(datapath.annot, sample)
      files <- list.files(path = datapath_i, pattern = "\\.xlsx$", full.names = TRUE)
      tcr_data_i <- lapply(files, function(x) {
        tcr_data <- read_excel(x, col_names = TRUE)
        tcr_data <- as.data.frame(tcr_data)
        tcr_data <- subset(tcr_data, subset = tcr_data$`Frame-info` == "ok")
        tcr_data$freq_in <- tcr_data$`total.nr-reads` / sum(tcr_data$`total.nr-reads`) * 100
        tcr_data$time <- tools::file_path_sans_ext(basename(x))
        tcr_data$`DNA-seq (AA2)` <- paste0(tcr_data$`DNA-seq (AA2)`, "_", tcr_data$`Vsegment-ID`, "_", tcr_data$`Jsegment-ID`)
        tcr_data <- tcr_data %>% select("DNA-seq (AA2)", "AA1: pept-seq Insert", "time", "total.nr-reads", "freq_in")
        colnames(tcr_data) <- c("nucleotide", "aminoAcid", "time", "count", "freq_in")
        return(tcr_data)
      })
      message("✅ reference tcrb repertoire data succesfully loaded...")
    } else {
      stop("🚧 invalid technology; only adaptive or hsd data are supported...")
    }
    # check if data has been properly loaded
    if (length(tcr_data_i) == 0) {
      message("🚧 invalid technology or input format; only only adaptive, hsd, SEQTR or MiXCR data are supported...")
      return(NULL)
    }
    names(tcr_data_i) <- tools::file_path_sans_ext(basename(files))
    biopsy_names <- names(tcr_data_i)
    
    # MODULE 2 --> annotate pbl by their presence in tumor.
    
    for (i in names(tcr_data_i)){
      long_data[[as.character(i)]] <- ifelse(long_data$nucleotide %in% unique(tcr_data_i[[i]]$nucleotide), "Y", "N")
      message(paste0("✅ tcrb repertoire succesfully annotated for ", i, "..."))
    }
    
    # filter day.limit (if specified).
    if (!is.null(day.limit)) {
      message("✂️ applying day.limit...")
      long_data <- long_data %>% mutate(time = as.numeric(gsub("^d", "", time))) %>% filter(time <= day.limit)
    } else {
      long_data <- long_data
    }
    
    # MODULE 3 --> calculate cumulative frequency for each trajectory.
    message("🧱 calculating cumulative frequencty for each trajectory...")
    cumulative <- long_data %>% group_by(traj, time) %>% summarise(cumulative_freq = sum(freq_in), .groups = "drop") %>% arrange(traj, time)
    cumulative <- cumulative %>% mutate(nucleotide = "cumulative_frequency",
                                        aminoAcid = "CUMULATIVE",
                                        count = NA,
                                        freq_in = cumulative$cumulative_freq) %>% as.data.frame()
    cumulative <- cumulative %>% select(nucleotide, aminoAcid, time, count, freq_in, traj)
    cumulative <- subset(cumulative, subset = !is.na(traj))
    cumulative[biopsy_names] <- "N"
    long_data <- rbind(long_data, cumulative)
    long_data$log10value <- log10(long_data$freq_in)
    
    # MODULE 4 --> save pbl_raw + pbl_tumor data with statistics.
    
    if (!exists("pbl", envir = .GlobalEnv)) {
      assign("pbl", list(), envir = .GlobalEnv)
    }
    pbl <- get("pbl", envir = .GlobalEnv)
    pbl[[sample]] <- long_data
    assign("pbl", pbl, envir = .GlobalEnv)
    
    # pbl_tumor data.
    tumor_pbl <- list()
    for (i in biopsy_names) {
      tumor_pbl_i <- long_data
      tumor_pbl_i <- subset(tumor_pbl_i, subset = tumor_pbl_i[[i]] == "Y")
      tumor_pbl[[i]] <- tumor_pbl_i
    }
    if (!exists("pbl_tumor", envir = .GlobalEnv)) {
      assign("pbl_tumor", list(), envir = .GlobalEnv)
    }
    pbl_tumor <- get("pbl_tumor", envir = .GlobalEnv)
    pbl_tumor[[sample]] <- tumor_pbl
    assign("pbl_tumor", pbl_tumor, envir = .GlobalEnv)
    
    # MODULE 5 --> plot top.n.clonotypes at baseline.
    
    if(!is.null(top.n.clonotypes)) {
      message("🔝 drawing trajectories of top clonotypes at baseline...")
      long_data$time <- gsub("d", "", long_data$time)
      min_time <- min(long_data$time)
      top_n_baseline <- long_data %>%
        filter(nucleotide != "cumulative_frequency") %>%
        group_by(nucleotide) %>%
        filter(time == min_time) %>%
        arrange(desc(freq_in)) %>%
        head(top.n.clonotypes) %>%
        mutate(!!paste0("top_", top.n.clonotypes) := "Y") %>%
        as.data.frame() %>%
        select(nucleotide, paste0("top_", top.n.clonotypes))
      long_data <- long_data %>% left_join(top_n_baseline, by = "nucleotide")
      long_data[[paste0("top_",top.n.clonotypes)]] <- ifelse(is.na(long_data[[paste0("top_",top.n.clonotypes)]]), "N", long_data[[paste0("top_",top.n.clonotypes)]])
      
      # prepare data.
      data_top_n <- long_data %>% filter(nucleotide %in% top_n_baseline$nucleotide)
      if (length(biopsy_names) == 1) {
        data_top_n[["biopsy"]] <- ifelse(data_top_n[[biopsy_names]] == "Y", biopsy_names, "not found in the biopsy")
      } else {
        data_top_n <- data_top_n %>%
          rowwise() %>%
          mutate(time = time,
                 log10value = -log10(freq_in),
                 biopsy = {
                   y_columns <- biopsy_names[which(c_across(all_of(biopsy_names)) == "Y")]
                   if (length(y_columns) > 0) {
                     paste(y_columns, collapse = " + ")
                   } else {
                     "not found in biopsies"
                   }
                 }) %>%
          ungroup()
      }
      
      # function ::complete_long_data::
      complete_long_data <- function(data,
                                     excluded_cols = NULL) {
        data$time <- paste0("d", data$time)
        metadata <- data %>% select(-freq_in) %>% distinct()
        
        # pivot wide format and fill NA with = min_freq
        data <- data %>%
          select(nucleotide, time, freq_in) %>%
          pivot_wider(id_cols = nucleotide, names_from = time, values_from = freq_in) %>%
          mutate(across(starts_with("d"), ~ as.numeric(as.character(.)))) %>%
          mutate(across(starts_with("d"), ~ ifelse(is.na(.), min_freq, .))) %>%
          as.data.frame()
        # pivot back to long format
        long_data_complete <- data %>%
          pivot_longer(cols = starts_with("d"), names_to = "time", values_to = "freq_in") %>%
          mutate(
            time = as.numeric(gsub("^d", "", time))
          ) %>%
          mutate(time = paste0("d", time)) %>%
          left_join(metadata, by = c("nucleotide", "time")) %>%
          mutate(count = ifelse(is.na(count), 0, count)) %>%
          distinct() %>%
          as.data.frame()
        # complete metadata with NA filled values
        excluded_cols <-  c("nucleotide", "time", "freq_in", "log10value", "count")
        process_cols <- setdiff(names(long_data_complete), excluded_cols)
        all_combinations <- long_data_complete %>% distinct(nucleotide, time)
        complete <- long_data_complete %>%
          left_join(all_combinations, by = c("nucleotide", "time")) %>%
          group_by(nucleotide) %>%
          mutate(
            aminoAcid = ifelse(is.na(aminoAcid), first(na.omit(aminoAcid)), aminoAcid),
            traj = ifelse(is.na(traj), first(na.omit(traj)), traj)
          ) %>%
          mutate(across(all_of(process_cols), ~ ifelse(is.na(.), first(na.omit(.)), .))) %>%
          ungroup() %>%
          mutate(
            aminoAcid = ifelse(is.na(aminoAcid), "unknown", aminoAcid),
            traj = ifelse(is.na(traj), "unknown", traj),
            #log10value = ifelse(is.na(log10value), -4, log10value),
            across(all_of(process_cols), ~ ifelse(is.na(.), "unclassified", .))
          ) %>%
          as.data.frame()
        return(complete)
      }
      
      # Complete data.
      data_top_n <- complete_long_data(data = data_top_n)
      data_top_n$time <- as.numeric(gsub("d", "", data_top_n$time))
      
      # Plot data.
      plot_top_n <- ggplot() +
        geom_line(data = data_top_n, 
                  aes(x = time,
                      y = log10(freq_in),
                      group = nucleotide),
                  color = "black",
                  alpha = 0.5,
                  linewidth = 0.2) +
        theme_minimal() +
        facet_wrap(~biopsy, ncol = 5) +
        scale_x_continuous() +
        scale_y_continuous(
          breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, max.freq))),
          labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", as.character(max.freq))),
          limits = c(log10(min_freq), log10(max.freq))
        ) +
        labs(y = "frequency") +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
          aspect.ratio = 0.5
        ) +
        labs(title = paste0(sample, " / top ", top.n.clonotypes, " clonotypes at baseline"),
             color = "presence"
        )
      if (plot.time){
        plot_top_n <- plot_top_n + geom_vline(data = long_data, aes(xintercept = as.numeric(time)), linetype = "dashed", color = "#D1D1D1", alpha = 0.5)
      }
      if (!is.null(icb.1)) {
        icb.1_df <- data.frame(day = icb.1, coord = 1.3)
        plot_top_n <- plot_top_n + geom_point(data = icb.1_df, aes(x = day, y = coord), size = 3, color = "black", fill = "white", shape = 25)
      }
      if (!is.null(icb.2)) {
        icb.2_df <- data.frame(day = icb.2, coord = 1.3)
        plot_top_n <- plot_top_n + geom_point(data = icb.2_df, aes(x = day, y = coord), size = 3, color = "black", fill = "gray30", shape = 25)
      }
      if (!is.null(Tx.1)) {
        Tx.1_df <- data.frame(day = Tx.1)
        plot_top_n <- plot_top_n + geom_vline(data = Tx.1_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#8DB6DF", alpha = 0.2)
      }
      if (!is.null(Tx.2)) {
        Tx.2_df <- data.frame(day = Tx.2)
        plot_top_n <- plot_top_n + geom_vline(data = Tx.2_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#B7460D", alpha = 0.2)
      }
      if (!is.null(Tx.3)) {
        Tx.3_df <- data.frame(day = Tx.3)
        plot_top_n <- plot_top_n + geom_vline(data = Tx.3_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "seagreen4", alpha = 0.2)
      }
      if (!is.null(biopsy.day)) {
        biopsy_day_df <- data.frame(day = biopsy.day, coord = 2)
        plot_top_n <- plot_top_n + geom_point(data = biopsy_day_df, aes(x = day, y = coord), size = 3, color = "black", shape = 13)
      }
      
      # save plots.
      if (!exists("traj_top_baseline", envir = .GlobalEnv)) {
        assign("traj_top_baseline", list(), envir = .GlobalEnv)
      }
      traj_top_baseline <- get("traj_top_baseline", envir = .GlobalEnv)
      if (is.null(traj_top_baseline[[sample]])) {
        traj_top_baseline[[sample]] <- list()
      }
      traj_top_baseline[[sample]] <- plot_top_n
      assign("traj_top_baseline", traj_top_baseline, envir = .GlobalEnv)
    }
    
    # MODULE 6 --> filter trajectories.
    
    if (filter.traj == TRUE) {
      filtered_traj <- subset(long_data, !is.na(traj))
      filtered_traj <- unique(filtered_traj$nucleotide)
      filtered_traj <- filtered_traj[!filtered_traj %in% "cumulative_frequency"]
    } else {
      filtered_traj <- unique(long_data$nucleotide)
      filtered_traj <- filtered_traj[!filtered_traj %in% "cumulative_frequency"]
    }
    
    if (filter.traj == TRUE) {
      filtered_data <- long_data %>% filter(nucleotide %in% filtered_traj)
      message(paste0("✅ filter.traj succesfully applied! ", length(unique(filtered_data$nucleotide)), " clonotypes are shown..."))
    } else {
      filtered_data <- long_data
      message(paste0("🚧 no filters applied; computational load could be high, please, be patient 💣💦🔫"))
    }
    
    filtered_data <- complete_long_data(data = filtered_data)
    filtered_data$time <- as.numeric(gsub("d", "", filtered_data$time))
    
    # save filtered_data.
    if (!exists("pbl_filt_traj_stat", envir = .GlobalEnv)) {
      assign("pbl_filt_traj_stat", list(), envir = .GlobalEnv)
    }
    pbl_filt <- get("pbl_filt_traj_stat", envir = .GlobalEnv)
    pbl_filt[[sample]] <- filtered_data
    assign("pbl_filt_traj_stat", pbl_filt, envir = .GlobalEnv)
    
    
    # MODULE 7 --> re-calculate cumulative frequency for filtered trajectories.
    
    cumulative <- filtered_data %>% group_by(traj, time) %>% summarise(cumulative_freq = sum(freq_in), .groups = "drop") %>% arrange(traj, time)
    cumulative <- cumulative %>% mutate(nucleotide = "cumulative_frequency",
                                        aminoAcid = "CUMULATIVE",
                                        count = NA,
                                        freq_in = cumulative$cumulative_freq) %>% as.data.frame()
    cumulative <- cumulative %>% select(nucleotide, aminoAcid, time, count, freq_in, traj)
    cumulative <- subset(cumulative, subset = !is.na(traj))
    all_combinations <- expand.grid(
      time = unique(cumulative$time),
      traj = unique(cumulative$traj)
    )
    cumulative <- all_combinations %>%
      left_join(cumulative, by = c("time", "traj"))
    cumulative <- cumulative %>%
      mutate(
        time = as.numeric(time),
        count = ifelse(is.na(count), 0, count),
        freq_in = ifelse(is.na(freq_in), min_freq, freq_in),
        aminoAcid = ifelse(is.na(aminoAcid), "CUMULATIVE", aminoAcid),
        nucleotide = ifelse(is.na(nucleotide), "cumulative_frequency", nucleotide),
      )
    
    # MODULE 8 --> highlight trajectories in absence of biopsies.
    
    if (length(biopsy_names) == 0) {
      plots_list <- list()
      # count traj.
      timepoints <- length(unique(filtered_data$time))
      count_traj <- filtered_data %>% select(nucleotide, traj) %>%
        group_by(traj) %>%
        summarise(
          count_total = n() / timepoints, # do not consider cumulative frequency.
          .groups = "drop"
        )
      message(paste0("no biopsy provided for ", sample))
      plot_traj_i <- ggplot() +
        geom_line(data = filtered_data,
                  aes(x = time,
                      y = log10(freq_in),
                      group = nucleotide),
                  color = "black",
                  alpha = 0.5,
                  linewidth = 0.2,
                  linetype = "solid") + 
        geom_line(data = cumulative,
                  aes(x = time,
                      y = log10(freq_in),
                      group = nucleotide),
                  color = "blue3",
                  alpha = 0.5,
                  linewidth = 0.5,
                  linetype = "dashed") +
        theme_minimal() +
        scale_x_continuous() +
        scale_y_continuous(
          breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, max.freq))),
          labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", as.character(max.freq))),
          limits = c(log10(min_freq), log10(max.freq))
        ) +
        labs(y = "frequency") +
        facet_wrap(~traj, ncol = 5) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
          aspect.ratio = 0.5
        ) +
        labs(
          title = paste0("no biopsy provided for ", sample),
          color = "presence"
        ) +
        geom_text(data = count_traj,
                  aes(x = 25, y = 0.5, label = count_total),
                  inherit.aes = FALSE,
                  position = position_nudge(y = 0), size = 4)
      if (plot.time){
        plot_traj_i <- plot_traj_i + geom_vline(data = filtered_data, aes(xintercept = as.numeric(time)), linetype = "dashed", color = "#D1D1D1", alpha = 0.5)
      }
      if (!is.null(icb.1)) {
        icb.1_df <- data.frame(day = icb.1, coord = 1.3)
        plot_traj_i <- plot_traj_i + geom_point(data = icb.1_df, aes(x = day, y = coord), size = 3, color = "black", fill = "white", shape = 25)
      }
      if (!is.null(icb.2)) {
        icb.2_df <- data.frame(day = icb.2, coord = 1.3)
        plot_traj_i <- plot_traj_i + geom_point(data = icb.2_df, aes(x = day, y = coord), size = 3, color = "black", fill = "gray30", shape = 25)
      }
      if (!is.null(Tx.1)) {
        Tx.1_df <- data.frame(day = Tx.1)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.1_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#8DB6DF", alpha = 0.2)
      }
      if (!is.null(Tx.2)) {
        Tx.2_df <- data.frame(day = Tx.2)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.2_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#B7460D", alpha = 0.2)
      }
      if (!is.null(Tx.3)) {
        Tx.3_df <- data.frame(day = Tx.3)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.3_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "seagreen4", alpha = 0.2)
      }
      if (!is.null(biopsy.day)) {
        biopsy_day_df <- data.frame(day = biopsy.day, coord = 2)
        plot_traj_i <- plot_traj_i + geom_point(data = biopsy_day_df, aes(x = day, y = coord), size = 3, color = "black", shape = 13)
      }
      plots_list[["traj"]] <- plot_traj_i
      
    } else {
      
      # MODULE 9 --> highlight trajectories when biopsies are available.
      
      plots_list <- list()
      
      for (i in seq_along(biopsy_names)) {
        
        # count trajectories found in biopsies.
        timepoints <- length(unique(filtered_data$time))
        count_traj <- filtered_data %>%
          select(nucleotide, traj, any_of(biopsy_names[i])) %>% distinct() %>%
          group_by(traj) %>%
          summarise(
            count_traj = sum(!!sym(biopsy_names[i]) == "Y", na.rm = TRUE),
            count_total = n_distinct(nucleotide),
            count_label = paste(count_traj, count_total, sep = "/"),
            .groups = "drop"
          )
        # plot by traj.
        filtered_data$biopsy_col <- filtered_data[[biopsy_names[[i]]]]
        plot_traj_i <- ggplot() +
          geom_line(data = filtered_data,
                    aes(x = time,
                        y = log10(freq_in),
                        group = nucleotide,
                        color = biopsy_col,
                        alpha = biopsy_col,
                        linewidth = biopsy_col)) + 
          geom_line(data = cumulative,
                    aes(x = time,
                        y = log10(freq_in),
                        group = nucleotide),
                    color = "blue3",
                    alpha = 0.5,
                    linewidth = 0.5,
                    linetype = "dashed") +
          theme_minimal() +
          scale_x_continuous() +
          scale_y_continuous(
            breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, max.freq))),
            labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", as.character(max.freq))),
            limits = c(log10(min_freq), log10(max.freq))
          ) +
          labs(y = "frequency") +
          scale_color_manual(values = c("Y" = "red3", "N" = "black")) +
          scale_alpha_manual(values = c("Y" = 1, "N" = 0.5)) +
          scale_linewidth_manual(values = c("Y" = 0.3, "N" = 0.2)) +
          facet_wrap(~traj, ncol = 5) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
            aspect.ratio = 0.5
          ) +
          labs(
            title = paste0(sample, " / non_neutral in ", biopsy_names[i]),
            color = "presence"
          ) +
          geom_text(data = count_traj,
                    aes(x = 25, y = 0.5, label = count_label),
                    inherit.aes = FALSE,
                    position = position_nudge(y = 0), size = 4)
        if (plot.time){
          plot_traj_i <- plot_traj_i + geom_vline(data = filtered_data, aes(xintercept = as.numeric(time)), linetype = "dashed", color = "#D1D1D1", alpha = 0.5)
        }
        if (!is.null(icb.1)) {
          icb.1_df <- data.frame(day = icb.1, coord = 1.3)
          plot_traj_i <- plot_traj_i + geom_point(data = icb.1_df, aes(x = day, y = coord), size = 3, color = "black", fill = "white", shape = 25)
        }
        if (!is.null(icb.2)) {
          icb.2_df <- data.frame(day = icb.2, coord = 1.3)
          plot_traj_i <- plot_traj_i + geom_point(data = icb.2_df, aes(x = day, y = coord), size = 3, color = "black", fill = "gray30", shape = 25)
        }
        if (!is.null(Tx.1)) {
          Tx.1_df <- data.frame(day = Tx.1)
          plot_traj_i <- plot_traj_i + geom_vline(data = Tx.1_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#8DB6DF", alpha = 0.2)
        }
        if (!is.null(Tx.2)) {
          Tx.2_df <- data.frame(day = Tx.2)
          plot_traj_i <- plot_traj_i + geom_vline(data = Tx.2_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#B7460D", alpha = 0.2)
        }
        if (!is.null(Tx.3)) {
          Tx.3_df <- data.frame(day = Tx.3)
          plot_traj_i <- plot_traj_i + geom_vline(data = Tx.3_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "seagreen4", alpha = 0.2)
        }
        if (!is.null(biopsy.day)) {
          biopsy_day_df <- data.frame(day = biopsy.day, coord = 2)
          plot_traj_i <- plot_traj_i + geom_point(data = biopsy_day_df, aes(x = day, y = coord), size = 3, color = "black", shape = 13)
        }
        plots_list[[biopsy_names[[i]]]] <- plot_traj_i
      }
    }
    # save plots.
    if (!exists("traj_non_neutral_highlighted", envir = .GlobalEnv)) {
      assign("traj_non_neutral_highlighted", list(), envir = .GlobalEnv)
    }
    traj_non_neutral_highlighted <- get("traj_non_neutral_highlighted", envir = .GlobalEnv)
    if (is.null(traj_non_neutral_highlighted[[sample]])) {
      traj_non_neutral_highlighted[[sample]] <- list()
    }
    traj_non_neutral_highlighted[[sample]] <- plots_list
    assign("traj_non_neutral_highlighted", traj_non_neutral_highlighted, envir = .GlobalEnv)
    # return plots.
    return(list(
      top_n = if (!is.null(top.n.clonotypes)) {
        traj_top_baseline[[sample]]
      } else { NULL },
      traj.plot = traj_non_neutral_highlighted[[sample]]
    ))
    
  } else {
    
    # MODULE 10 --> highlight trajectories when biopsy is specified.
    
    if (!exists(input.data, envir = .GlobalEnv)){
      message("⚠️ please, generate pbl_tumor object before by running highlight_trajectories()...")
      return(NULL)
    } else {
      # tumor clonotype traj.
      pbl_tumor_traj <- get("pbl_tumor_traj", envir = .GlobalEnv)[[sample]][[biopsy]]
      traj <- pbl_tumor_traj %>% select(nucleotide, cluster)
      traj_nucleotide <- unique(traj$nucleotide)
      
      # long_data
      long_data <- get(input.data, envir = .GlobalEnv)[[sample]][[biopsy]]
      long_data <- long_data %>% select(nucleotide, time, freq_in, log10value, aminoAcid, biopsy)
      if (!is.null(day.limit)) {
        long_data <- long_data %>% mutate(time = as.numeric(gsub("^d", "", time))) %>% filter(time <= day.limit)
      }
      top.n.clonotypes <- long_data %>% 
        group_by(nucleotide) %>%
        summarise(max_log10value = max(log10value)) %>%
        arrange(desc(max_log10value)) %>%
        slice_head(n = max.biopsy) %>%
        pull(nucleotide)
      long_data <- long_data %>% filter(nucleotide %in% top.n.clonotypes)
      long_data <- long_data %>% left_join(traj, by = "nucleotide")
      long_data$time <- gsub("d", "", long_data$time)
      metadata <- long_data %>% select(-time, -freq_in, -log10value) %>% distinct()
      long_data <- long_data %>% complete(nucleotide, time, fill = list(freq_in = min_freq))
      long_data$log10value <- ifelse(is.na(long_data$log10value), log10(min_freq), long_data$log10value)
      long_data <- long_data %>% select(nucleotide, time, freq_in, log10value) %>% left_join(metadata, by = "nucleotide")
      
      # cumulative.
      cumulative <- pbl_tumor_traj %>% pivot_longer(cols = starts_with("d"),
                                                    names_to = "time",
                                                    values_to = "freq_in") %>% as.data.frame()
      cumulative_freq <- cumulative %>% group_by(cluster, time) %>% summarise(cumulative_freq = sum(freq_in), .groups = "drop") %>% arrange(cluster, time)
      cumulative_freq$time <- gsub("d", "", cumulative_freq$time)
      cumulative_long <- cumulative_freq %>% mutate(nucleotide = "cumulative_frequency",
                                                    freq_in = cumulative_freq,
                                                    log10value = log10(cumulative_freq),
                                                    aminoAcid = "CUMULATIVE",
                                                    traj = NA)
      cumulative_long[biopsy] <- "N"
      cumulative_long <- cumulative_long %>% select(nucleotide, time, freq_in, log10value, aminoAcid, biopsy, cluster) %>% as.data.frame()
      long_data <- rbind(long_data, cumulative_long)
      long_data$time <- as.numeric(long_data$time)
      
      # count traj.
      timepoints <- length(unique(long_data$time))
      count_traj <- long_data %>%
        group_by(cluster) %>%
        summarise(
          count_total = n() / timepoints -1, # do not consider cumulative frequency.
          .groups = "drop"
        )
      plot_traj_i <- ggplot() +
        geom_line(data = long_data %>% filter(nucleotide != "cumulative_frequency"),
                  aes(x = time,
                      y = log10value,
                      group = nucleotide),
                  color = "black",
                  alpha = 0.5,
                  linewidth = 0.2,
                  linetype = "solid") + 
        geom_line(data = long_data %>% filter(nucleotide == "cumulative_frequency"),
                  aes(x = time,
                      y = log10value,
                      group = nucleotide),
                  color = "blue3",
                  alpha = 0.5,
                  linewidth = 0.5,
                  linetype = "dashed") +
        theme_minimal() +
        scale_x_continuous() +
        scale_y_continuous(
          breaks = log10(unique(c(min_freq, 0.001, 0.01, 0.1, 1, 10, max.freq))),
          labels = unique(c(as.character(min_freq), "0.001", "0.01", "0.1", "1", "10", as.character(max.freq))),
          limits = c(log10(min_freq), log10(max.freq))
        ) +
        labs(y = "frequency") +
        facet_wrap(~cluster, ncol = 5) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
          aspect.ratio = 0.5
        ) +
        labs(
          title = paste0(sample, " / ", biopsy, " infiltrating clonotypes in the blood")
        ) +
        geom_text(data = count_traj,
                  aes(x = 25, y = 0.5, label = count_total),
                  inherit.aes = FALSE,
                  position = position_nudge(y = 0), size = 4)
      if (plot.time){
        plot_traj_i <- plot_traj_i + geom_vline(data = long_data, aes(xintercept = as.numeric(time)), linetype = "dashed", color = "#D1D1D1", alpha = 0.5)
      }
      if (!is.null(icb.1)) {
        icb.1_df <- data.frame(day = icb.1, coord = 1.3)
        plot_traj_i <- plot_traj_i + geom_point(data = icb.1_df, aes(x = day, y = coord), size = 3, color = "black", fill = "white", shape = 25)
      }
      if (!is.null(icb.2)) {
        icb.2_df <- data.frame(day = icb.2, coord = 1.3)
        plot_traj_i <- plot_traj_i + geom_point(data = icb.2_df, aes(x = day, y = coord), size = 3, color = "black", fill = "gray30", shape = 25)
      }
      if (!is.null(Tx.1)) {
        Tx.1_df <- data.frame(day = Tx.1)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.1_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#8DB6DF", alpha = 0.2)
      }
      if (!is.null(Tx.2)) {
        Tx.2_df <- data.frame(day = Tx.2)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.2_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "#B7460D", alpha = 0.2)
      }
      if (!is.null(Tx.3)) {
        Tx.3_df <- data.frame(day = Tx.3)
        plot_traj_i <- plot_traj_i + geom_vline(data = Tx.3_df, aes(xintercept = day), linetype = "solid", linewidth = 2, color = "seagreen4", alpha = 0.2)
      }
      if (!is.null(biopsy.day)) {
        biopsy_day_df <- data.frame(day = biopsy.day, coord = 2)
        plot_traj_i <- plot_traj_i + geom_point(data = biopsy_day_df, aes(x = day, y = coord), size = 3, color = "black", shape = 13)
      }
    }
    message("✅ done!")
    # save plots.
    if (!exists("traj_tumor_pbl_highlighted", envir = .GlobalEnv)) {
      assign("traj_tumor_pbl_highlighted", list(), envir = .GlobalEnv)
    }
    traj_tumor_pbl_highlighted <- get("traj_tumor_pbl_highlighted", envir = .GlobalEnv)
    traj_tumor_pbl_highlighted[[sample]][[biopsy]] <- plot_traj_i
    assign("traj_tumor_pbl_highlighted", traj_tumor_pbl_highlighted, envir = .GlobalEnv)
    # return plots.
    return(traj_tumor_pbl_highlighted[[sample]])
  }
}

# function ::captured_clonotypes::
count_clonotypes <- function(sample, n_iter = 25) {
  if (!exists("captured_clonotypes", envir = .GlobalEnv)) {
    captured_clonotypes <<- list()
  }
  biopsies <- names(pbl_tumor[[sample]])
  plots_list <- lapply(biopsies, function(bio) {
    non_neutral <- unique(pbl_filt[[sample]]$nucleotide)
    count_non_neutral <- pbl[[sample]] %>%
      filter(nucleotide %in% non_neutral) %>%
      select(nucleotide, all_of(bio)) %>%
      distinct() %>%
      summarise(count = sum(.data[[bio]] == "Y", na.rm = TRUE)) %>%
      pull(count)
    random_counts <- numeric(n_iter)
    for (i in seq_len(n_iter)) {
      random_sample <- sample(pbl[[sample]]$nucleotide, length(non_neutral))
      count_random <- pbl[[sample]] %>%
        filter(nucleotide %in% random_sample) %>%
        select(nucleotide, all_of(bio)) %>%
        distinct() %>%
        summarise(count = sum(.data[[bio]] == "Y", na.rm = TRUE)) %>%
        pull(count)
      random_counts[i] <- count_random
    }
    mean_random <- mean(random_counts)
    sd_random <- sd(random_counts)
    df <- data.frame(
      Group = c("traj", "random"),
      Mean = c(count_non_neutral, mean_random),
      SD = c(0, sd_random),
      Label = c(as.character(count_non_neutral), as.character(round(mean_random, 2)))
    )
    max_value <- max(count_non_neutral, mean_random + sd_random)
    ylim_range <- c(0, max_value * 1.2)
    ggplot(df, aes(x = Group, y = Mean, fill = Group)) +
      geom_point(stat = "identity", position = position_dodge(), width = 0.6, size = 4) +
      geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD), width = 0.2) +
      geom_text(aes(label = Label), vjust = -2, size = 4) +
      labs(title = paste(sample, "|", bio),
           y = "captured non-neutral PBL-TILs") +
      scale_y_continuous(limits = ylim_range) +
      theme_light() +
      theme(aspect.ratio = 1.5)
  })
  wrapped_plot <- wrap_plots(plots_list)
  captured_clonotypes[[sample]] <<- wrapped_plot
  print(wrapped_plot)
}


# function ::save_data::
save_data <- function(datapath,
                      export.raw.data = FALSE,
                      dpi = 300) {
  results_path <- file.path(datapath, "results")
  if (!dir.exists(results_path)) {
    dir.create(results_path)
    message("✅ results directory succesfully created...")
  }
  for (i in names(pbl_filt)) {
    message(paste("📁 saving files... please, be patient and thank you for using ibilbide 😊"))
    subfolder_path <- file.path(results_path, i)
    if (!dir.exists(subfolder_path)) {
      dir.create(subfolder_path)
    }
    if (exists("pbl")) {
    if (export.raw.data) {
      write.table(
        pbl[[i]], 
        file = file.path(subfolder_path, paste0("pbl_raw_", i, ".tsv")),
        sep = "\t", 
        row.names = FALSE, 
        quote = FALSE
      )
    }}
    if (exists("pbl_filt_traj_stat")) {
      write.table(
        pbl_filt_traj_stat[[i]], 
        file = file.path(subfolder_path, paste0("pbl_non_neutral_filtered_", i, ".tsv")),
        sep = "\t", 
        row.names = FALSE, 
        quote = FALSE
      )}
    # Save pbl found in biopsies.
    if (exists("pbl_tumor_traj")) {
      for (j in names(pbl_tumor_traj[[i]])) {
        dictionary <- pbl_tumor_traj[[i]][[j]] %>% select(nucleotide, cluster)
        colnames(dictionary) <- c("nucleotide", "tumor_traj")
        if (!is.null(pbl_tumor[[i]][[j]])) {
          pbl_tumor_j <- pbl_tumor[[i]][[j]] %>% left_join(dictionary, by = "nucleotide")
          pbl_tumor_j <- pbl_tumor_j %>% select(nucleotide, time, freq_in, aminoAcid, count, everything())
          file_name <- file.path(subfolder_path, paste0("pbl_tumor_traj_", i, "_", j, ".tsv"))
          write.table(
            pbl_tumor_j,
            file = file_name,
            sep = "\t",
            row.names = FALSE,
            quote = FALSE
          )
        } else {
          message(paste("Skipping", i, "-", j, "because pbl_tumor_j is NULL"))
        }
      }
    }
    # Convert from cm to pixels (300 DPI)
    dpi <- dpi
    width_cm <- 297/7.5
    height_cm <- 210/7.5
    width_px <- (width_cm / 2.54) * dpi
    height_px <- (height_cm / 2.54) * dpi
    
    # Save the trajectory plots as TIFF files (maintaining proportions)
    if (exists("traj_fc")) {
    tiff(file.path(subfolder_path, paste0("traj_fc_plot.tiff")), 
         width = width_px, height = height_px, res = dpi)
    print(wrap_plots(traj_fc[[i]]))
    dev.off()
    }
    if (exists("traj_non_neutral")) {
    tiff(file.path(subfolder_path, paste0("traj_non_neutral.tiff")), 
         width = width_px, height = height_px, res = dpi)
    if (inherits(traj_non_neutral[[i]], "patchwork")) {
      grid::grid.draw(traj_non_neutral[[i]])
    } else {
      message(traj_non_neutral[[i]])
    }
    dev.off()
    }
    if (exists("traj_top_baseline")) {
      dev.new()
      tiff(file.path(subfolder_path, paste0("traj_top_baseline.tiff")), 
           width = width_px, height = height_px, res = dpi)
      if (inherits(traj_top_baseline[[i]], "patchwork")) {
        grid::grid.draw(traj_top_baseline[[i]])
      } else {
        message(traj_top_baseline[[i]])
      }
      dev.off()
    }
    if (exists("traj_non_neutral_highlighted")) {
    for (j in names(traj_non_neutral_highlighted[[i]])) {
      tiff(file.path(subfolder_path, paste0("traj_non_neutral_highlighted_", j, ".tiff")), 
           width = width_px, height = height_px, res = dpi)
      print(wrap_plots(traj_non_neutral_highlighted[[i]][[j]]))
      dev.off()
      }
    }
    if (exists("traj_tumor_pbl")) {
    for (j in names(traj_tumor_pbl[[i]])) {
        tiff(file.path(subfolder_path, paste0("traj_tumor_", j, ".tiff")), 
             width = width_px, height = height_px, res = dpi)
        if (inherits(traj_tumor_pbl[[i]][[j]], "patchwork")) {
          grid::grid.draw(traj_tumor_pbl[[i]][[j]])
        } else {
          message(traj_tumor_pbl[[i]][[j]])
        }
        dev.off()
    }}
    if (exists("traj_tumor_pbl_highlighted")) {
      for (j in names(traj_tumor_pbl_highlighted[[i]])) {
        tiff(file.path(subfolder_path, paste0("traj_tumor_highlighted_", j, ".tiff")), 
             width = width_px, height = height_px, res = dpi)
        print(patchwork::wrap_plots(traj_tumor_pbl_highlighted[[i]][[j]]))
        dev.off()
      }
    }
    if (exists("captured_clonotypes")) {
      tiff(file.path(subfolder_path, paste0("captured_clonotypes_", i, ".tiff")), 
           width = width_px, height = height_px, res = dpi)
      print(captured_clonotypes[[i]])
      dev.off()
    }
  }
}

# function ::initial_data_qc::
initial_data_qc <- function(data_list, ncol = 4,
                            headroom = 0.10,
                            show_checks = FALSE,
                            na_rm_counts = TRUE) {
  
  if (show_checks) {
    for (i in names(data_list)) {
      for (j in names(data_list[[i]])) {
        x <- data_list[[i]][[j]]
        message(paste0(i, "_", j, "_unique_", length(x$nucleotide)))
        message(paste0(i, "_", j, "_counts_", sum(x$count, na.rm = na_rm_counts)))
      }
    }
  }
  
  df <- purrr::imap_dfr(data_list, function(donor_list, donor) {
    purrr::imap_dfr(donor_list, function(x, tp) {
      tibble::tibble(
        donor     = donor,
        timepoint = tp,
        day       = as.numeric(stringr::str_extract(tp, "\\d+")),
        unique    = length(x$nucleotide),
        counts    = sum(x$count, na.rm = na_rm_counts)
      )
    })
  }) %>%
    dplyr::arrange(donor, day)
  
  u_min <- min(df$unique, na.rm = TRUE)
  u_max <- max(df$unique, na.rm = TRUE)
  c_rng <- range(df$counts, na.rm = TRUE)
  
  a <- diff(c_rng) / diff(c(u_min, u_max))
  b <- c_rng[1] - a * u_min
  
  y_min <- min(df$unique, (df$counts - b) / a, na.rm = TRUE)
  y_max <- max(df$unique, (df$counts - b) / a, na.rm = TRUE)
  y_win <- c(y_min, y_max * (1 + headroom))
  
  plot_one <- function(d) {
    d_days <- sort(unique(d$day))
    
    ggplot2::ggplot(d, ggplot2::aes(x = day)) +
      ggplot2::geom_line(
        ggplot2::aes(y = unique, colour = "Unique", linetype = "Unique"),
        linewidth = 0.7
      ) +
      ggplot2::geom_point(
        ggplot2::aes(y = unique, colour = "Unique"),
        size = 2
      ) +
      ggplot2::geom_line(
        ggplot2::aes(y = (counts - b) / a, colour = "Counts", linetype = "Counts"),
        linewidth = 0.7
      ) +
      ggplot2::geom_point(
        ggplot2::aes(y = (counts - b) / a, colour = "Counts"),
        size = 2
      ) +
      ggplot2::scale_x_continuous(
        breaks = d_days,
        expand = ggplot2::expansion(mult = c(0.02, 0.02))
      ) +
      ggplot2::scale_y_continuous(
        name = "Unique",
        labels = scales::comma,
        sec.axis = ggplot2::sec_axis(~ . * a + b, name = "Counts", labels = scales::comma)
      ) +
      ggplot2::coord_cartesian(ylim = y_win, clip = "off") +
      ggplot2::scale_colour_manual(
        values = c("Unique" = "black", "Counts" = "gray60")
      ) +
      ggplot2::scale_linetype_manual(
        values = c("Unique" = "solid", "Counts" = "dashed")
      ) +
      ggplot2::guides(
        colour = ggplot2::guide_legend(title = NULL),
        linetype = "none"
      ) +
      ggplot2::labs(title = unique(d$donor), x = "Day") +
      ggplot2::theme_bw() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5),
        axis.title.y.right = ggplot2::element_text(margin = ggplot2::margin(l = 8)),
        aspect.ratio = 0.5,
        legend.position = "bottom"
      )
  }
  
  plots <- df %>% split(.$donor) %>% purrr::map(plot_one)
  p <- patchwork::wrap_plots(plots, ncol = ncol)
  
  list(
    plot = p,
    df = df,
    transform = list(a = a, b = b, y_win = y_win, c_rng = c_rng, u_rng = c(u_min, u_max))
  )
}
