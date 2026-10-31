# 21_svycoxph_full.R
# Design-based Cox model on the WHOLE Cox sample (called by 21_svycoxph_full.py).
#   survey::svycoxph with ids = survey-specific PSU, strata = survey-specific DHS sampling
#   stratum (v022, else v023, else the survey), weights = v005/1e6, nest = TRUE,
#   and strata(country-survey) in the model formula, so the baseline hazards are the same
#   85 country-survey strata as in the main lifelines model.
#   For comparison: survival::coxph with cluster(PSU) (the estimator of the main model).
# Args: input CSV, output CSV
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; outp <- args[2]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({ library(survival); library(survey) })
options(survey.lonely.psu = "adjust")
d <- read.csv(inp)
covs <- c("urban", "wealth_q1", "wealth_q2", "wealth_q4", "wealth_q5", "educ_none", "educ_secondary_plus")
rhs <- paste(covs, collapse = " + ")
f_svy <- as.formula(paste("Surv(duration_months, event) ~", rhs, "+ strata(cox_stratum)"))
f_cl <- as.formula(paste("Surv(duration_months, event) ~", rhs, "+ strata(cox_stratum) + cluster(psu)"))

tab <- function(m, label) {
  sm <- summary(m)$coefficients
  se_col <- if ("robust se" %in% colnames(sm)) "robust se" else "se(coef)"
  data.frame(model = label, covariate = rownames(sm), HR = exp(sm[, "coef"]),
             lower95 = exp(sm[, "coef"] - qnorm(0.975) * sm[, se_col]),
             upper95 = exp(sm[, "coef"] + qnorm(0.975) * sm[, se_col]),
             se = sm[, se_col], n = nrow(d), psus = length(unique(d$psu)),
             design_strata = length(unique(d$design_stratum)), row.names = NULL)
}
t0 <- Sys.time()
m1 <- coxph(f_cl, data = d, weights = weight, ties = "efron")
cat("coxph + cluster(PSU) done:", format(Sys.time() - t0), "\n")
des <- svydesign(ids = ~psu, strata = ~design_stratum, weights = ~weight, data = d, nest = TRUE)
m2 <- svycoxph(f_svy, design = des)
cat("svycoxph done:", format(Sys.time() - t0), "\n")
res <- rbind(tab(m1, "coxph, PSU-clustered sandwich (as main model)"),
             tab(m2, "svycoxph, PSU + DHS sampling strata"))
write.csv(res, outp, row.names = FALSE)
cat(R.version.string, "| survival", as.character(packageVersion("survival")),
    "| survey", as.character(packageVersion("survey")), "\n")
