# 16_svycoxph_india.R
# Cox models for India NFHS-5 (IAKR7E), called by 16_svycoxph_india.py.
#   survival::coxph, model-based SEs
#   survival::coxph + cluster(v021): weighted robust sandwich clustered on the PSU, the same
#     estimator lifelines uses with robust=True, cluster_col (as in 05)
#   survey::svycoxph, design ids = v021, strata = v022 (and v023), weights = v005/1e6, nest = TRUE
# Args: full-sample CSV, 5,000-child validation CSV, output CSV
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; val <- args[2]; outp <- args[3]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({ library(survival); library(survey) })
options(survey.lonely.psu = "adjust")
covs <- c("urban", "wealth_q1", "wealth_q2", "wealth_q4", "wealth_q5", "educ_none", "educ_secondary_plus")
f <- as.formula(paste("Surv(duration_months, event) ~", paste(covs, collapse = " + ")))

tab <- function(m, label, d, extra = list()) {
  sm <- summary(m)$coefficients
  se_col <- if ("robust se" %in% colnames(sm)) "robust se" else "se(coef)"
  ci <- exp(cbind(sm[, "coef"] - qnorm(0.975) * sm[, se_col], sm[, "coef"] + qnorm(0.975) * sm[, se_col]))
  data.frame(model = label, covariate = rownames(sm), HR = exp(sm[, "coef"]), lower95 = ci[, 1],
             upper95 = ci[, 2], se = sm[, se_col], p = sm[, ncol(sm)], n = nrow(d),
             psu_n = length(unique(d$v021)), strata_n = if (is.null(extra$s)) NA else length(unique(d[[extra$s]])),
             row.names = NULL)
}

run_all <- function(d, tag) {
  out <- list()
  out[[1]] <- tab(coxph(f, data = d, weights = weight, ties = "efron", robust = FALSE),
                  paste0(tag, "R coxph, model-based SEs"), d)
  out[[2]] <- tab(coxph(update(f, . ~ . + cluster(v021)), data = d, weights = weight, ties = "efron", robust = TRUE),
                  paste0(tag, "R coxph, PSU-clustered sandwich (v021)"), d)
  for (s in c("v022", "v023")) {
    if (all(is.na(d[[s]]))) next
    des <- svydesign(ids = ~v021, strata = as.formula(paste0("~", s)), weights = ~weight, data = d, nest = TRUE)
    out[[length(out) + 1]] <- tab(svycoxph(f, design = des),
                                  paste0(tag, "R svycoxph (ids = v021, strata = ", s, ")"), d, list(s = s))
  }
  do.call(rbind, out)
}

res <- rbind(run_all(read.csv(inp), ""), run_all(read.csv(val), "VALIDATION 5k: "))
write.csv(res, outp, row.names = FALSE)
cat(R.version.string, "| survival", as.character(packageVersion("survival")),
    "| survey", as.character(packageVersion("survey")), "\n")
