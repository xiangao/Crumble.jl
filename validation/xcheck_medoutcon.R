# medoutcon on the same data Crumble.jl sees (validation/results/xcheck_n5000.csv).
suppressPackageStartupMessages({library(medoutcon); library(data.table)})
set.seed(1)
d <- fread("results/xcheck_n5000.csv")
w <- c("W_1", "W_2", "W_3")
for (eff in c("direct", "indirect")) {
  fit <- medoutcon(W = d[, ..w], A = d$A, Z = d$Z, M = d[, "M"], Y = d$Y,
                   effect = eff, estimator = "onestep")
  cat(sprintf("medoutcon %-8s %.4f (SE %.4f)\n", eff, fit$theta, sqrt(fit$var)))
}
