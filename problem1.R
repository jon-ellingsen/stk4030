load("wine.Rdata")

wine[, !names(wine) %in% test] <- scale(wine[, !names(wine) %in% test])

ols.a <- lm(quality ~ . -1, wine[wine$test == "FALSE", !names(wine) %in% test])

predict.a <- predict.lm(ols.a, wine[wine$test == "TRUE", !names(wine) %in% test])

mse <- mean((wine[wine$test == "TRUE", "quality"] - predict.a)^2)

