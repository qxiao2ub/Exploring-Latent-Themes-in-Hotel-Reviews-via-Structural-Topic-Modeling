
.TOPICS   <- 3L      # number of topics
.SEED     <- 123L    # RNG seed
.RUNS     <- 3L      # number of STM fits for model selection
.SAMPLE_N <- 5000L   # subsample size for speed (set to NA to skip)

# ---- packages ----
pkgs <- c("stm","ggplot2","tm","SnowballC","wordcloud","RColorBrewer")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) {
  install.packages(p, repos = "https://cloud.r-project.org", dependencies = TRUE)
}
suppressPackageStartupMessages({
  library(stm); library(ggplot2); library(tm); library(SnowballC)
  library(wordcloud); library(RColorBrewer)
})

# ---- output dir ----
.outdir <- "output"
if (!dir.exists(.outdir)) dir.create(.outdir, recursive = TRUE)

png_save <- function(file, expr, w=8, h=6, res=150) {
  grDevices::png(filename = file, width = w, height = h, units = "in", res = res)
  on.exit(grDevices::dev.off(), add = TRUE)
  force(expr)
}

# ---- data ----
resolve_csv_path <- function(csv_path) {
  if (file.exists(csv_path)) return(csv_path)
  cand <- list.files(".", pattern = "^hotel_reviews\\.csv$", recursive = TRUE, full.names = TRUE)
  if (length(cand) > 0 && file.exists(cand[1])) return(cand[1])
  stop("Could not find 'hotel_reviews.csv'.")
}
csv <- resolve_csv_path("hotel_reviews.csv")
message("Reading data: ", csv)
set.seed(.SEED)
df <- read.csv(csv, stringsAsFactors = FALSE)
stopifnot(all(c("Review","Rating") %in% names(df)))

if (is.numeric(.SAMPLE_N) && .SAMPLE_N > 0 && .SAMPLE_N < nrow(df)) {
  message("Sampling ", .SAMPLE_N, " rows from ", nrow(df), " total for a lighter run...")
  df <- df[sample(seq_len(nrow(df)), .SAMPLE_N), , drop = FALSE]
}

docs_raw <- df[["Review"]]
rating_raw <- df[["Rating"]]
rating_num <- suppressWarnings(as.numeric(rating_raw)); if (all(is.na(rating_num))) rating_num <- NULL
meta <- data.frame(Rating = as.factor(rating_raw), RatingNum = rating_num, stringsAsFactors = FALSE)
meta$ReviewOriginal <- docs_raw

message("Preprocessing text with stm::textProcessor ...")
tp <- textProcessor(docs_raw, metadata = meta, lowercase = TRUE, removestopwords = TRUE,
                    removenumbers = TRUE, removepunctuation = TRUE, stem = TRUE, verbose = FALSE)
message("Preparing documents with stm::prepDocuments ...")
prep <- prepDocuments(tp$documents, tp$vocab, tp$meta, verbose = FALSE)
if (length(prep$documents) == 0) stop("No documents left after preprocessing.")

# ---- fit multiple models ----
docs  <- prep$documents; vocab <- prep$vocab; mmeta <- prep$meta
fits <- vector("list", .RUNS); ok <- logical(.RUNS)
for (i in seq_len(.RUNS)) {
  message(sprintf("Fitting STM run %d/%d ...", i, .RUNS))
  set.seed(.SEED + i)
  fits[[i]] <- tryCatch(
    stm(documents = docs, vocab = vocab, K = .TOPICS, prevalence = ~ Rating, data = mmeta,
        init.type = "Spectral", verbose = FALSE),
    error = function(e) { message("stm() failed: ", conditionMessage(e)); NULL }
  )
  ok[i] <- !is.null(fits[[i]])
}
if (!any(ok)) stop("All STM runs failed.")
fits <- fits[ok]; ok_idx <- which(ok)

# ---- score & pick best ----
sc_avg <- ex_avg <- rep(NA_real_, length(fits))
for (i in seq_along(fits)) {
  sc <- tryCatch(semanticCoherence(fits[[i]], documents = docs), error = function(e) NA_real_)
  ex <- tryCatch(exclusivity(fits[[i]]), error = function(e) NA_real_)
  sc_avg[i] <- if (all(is.na(sc))) NA_real_ else mean(sc, na.rm = TRUE)
  ex_avg[i] <- if (all(is.na(ex))) NA_real_ else mean(ex, na.rm = TRUE)
}
scores <- data.frame(run = ok_idx, semantic_coherence = sc_avg, exclusivity = ex_avg)
keep <- !(is.na(scores$semantic_coherence) & is.na(scores$exclusivity))
scores <- scores[keep, , drop = FALSE]
if (nrow(scores) == 0) stop("Unable to compute scores for any model.")
z <- function(x) if (all(is.na(x))) rep(NA_real_, length(x)) else as.numeric(scale(x)[,1])
scores$combined <- rowMeans(cbind(z(scores$semantic_coherence), z(scores$exclusivity)), na.rm = TRUE)
try(ggsave(filename = file.path(.outdir, "coherence_exclusivity_scatter.png"),
     plot = ggplot(scores, aes(x = semantic_coherence, y = exclusivity, label = run)) +
       geom_point() + geom_text(nudge_y = 0.02, size = 3) +
       labs(title = "Model Runs: Semantic Coherence vs Exclusivity",
            x = "Semantic Coherence (avg)", y = "Exclusivity (avg)"),
     width = 8, height = 6, dpi = 150), silent = TRUE)
write.csv(scores, file.path(.outdir, "model_selection_scores.csv"), row.names = FALSE)
best_row <- which.max(scores$combined); if (length(best_row) == 0 || is.na(best_row)) best_row <- 1L
best <- fits[[match(scores$run[best_row], ok_idx)]]

# ---- analyze topics ----
K_local <- as.integer(best$settings$dim$K)
lab <- labelTopics(best, n = 7L)
write.csv(data.frame(topic = seq_len(K_local),
  words = sapply(seq_len(K_local), function(k) paste(lab$prob[k, ], collapse = ", "))), 
  file.path(.outdir, "top_words_per_topic.csv"), row.names = FALSE)

for (k in seq_len(K_local)) {
  outfile <- file.path(.outdir, sprintf("wordcloud_topic_%02d.png", k))
  tryCatch({
    png_save(outfile, cloud(best, topic = k))
  }, error = function(e) {
    dfk <- data.frame(term = lab$prob[k, ], rank = seq_along(lab$prob[k, ]))
    dfk <- dfk[1:min(15, nrow(dfk)), ]
    p <- ggplot(dfk, aes(x = reorder(term, -rank), y = rank)) + geom_col() + coord_flip() +
         labs(title = paste("Topic", k, "- top terms (fallback)"),
              x = "term", y = "rank (lower is stronger)")
    ggsave(outfile, p, width = 8, height = 6, dpi = 150)
  })
}

th <- findThoughts(best, texts = mmeta$ReviewOriginal, n = 5L)
rows <- lapply(seq_len(K_local), function(k) {
  data.frame(topic = k, rank = seq_along(th$docs[[k]]), review = th$docs[[k]], stringsAsFactors = FALSE)
})
write.csv(do.call(rbind, rows), file.path(.outdir, "top_reviews_all_topics.csv"), row.names = FALSE)
write.csv(data.frame(topic = seq_len(K_local),
          auto_label = sapply(seq_len(K_local), function(k) paste(head(lab$prob[k, ], 3), collapse = " "))),
          file.path(.outdir, "auto_labels.csv"), row.names = FALSE)

# ---- effects (safe) ----
try({
  png_save(file.path(.outdir, "topic_prevalence_summary.png"),
           plot(best, type = "summary", n = 10))
  eff <- estimateEffect(seq_len(K_local) ~ Rating, best, metadata = mmeta, uncertainty = "Global")
  saveRDS(eff, file = file.path(.outdir, "estimateEffect_Rating.rds"))
  for (k in seq_len(K_local)) {
    png_save(file.path(.outdir, sprintf("rating_effect_topic_%02d.png", k)),
             plot(eff, "Rating", method = "pointestimate",
                  topics = k, labeltype = "custom",
                  custom.labels = paste("Topic", k),
                  main = sprintf("Effect of Rating on Topic %d Prevalence", k)))
  }
}, silent = TRUE)

# ---- correlation ----
if (!is.null(mmeta$RatingNum)) {
  theta <- best$theta
  cors <- sapply(seq_len(K_local), function(k) cor(theta[, k], mmeta$RatingNum, use = "pairwise.complete.obs"))
  write.csv(data.frame(topic = seq_len(K_local), cor_with_rating = cors),
            file.path(.outdir, "topic_rating_correlations.csv"), row.names = FALSE)
}

message("DONE. Outputs in: ", normalizePath(.outdir))
