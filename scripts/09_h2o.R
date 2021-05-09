# ------------- Auto ML ---------------
# Objetivo: elaborar un modelo automl de h2o
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  library(ggrepel)       # Labels ggplot2
  library(stringi)       # Tratamiento de strings
  library(corrplot)      # Matriz de correlacion (grafico)
  library(h2o)           # AutoML
  
})

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 2
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")


h2o.init()

#-- Covertimos el dataset a un objeto h2o
surgical_dataset_h <- as.h2o(surgical_dataset)

#   Modelo 1
aml_1 <- h2o.automl(x = var_modelo1, y = target, 
                    training_frame = surgical_dataset_h, nfolds = 5, seed = 1234,
                    balance_classes = TRUE, keep_cross_validation_predictions = TRUE,
                    max_models = 20)

lb_1 <- aml_1@leaderboard
print(lb_1, n = nrow(lb_1))

leader_1 <- aml_1@leader
parameters_1 <- leader_1@parameters

aml_2 <- h2o.automl(x = var_modelo2, y = target, 
                    training_frame = surgical_dataset_h, nfolds = 5, seed = 1234, 
                    balance_classes = TRUE, keep_cross_validation_predictions = TRUE,
                    max_models = 20)

lb_2 <- aml_2@leaderboard
print(lb_2, n = nrow(lb_2))

leader_2 <- aml_2@leader
parameters_2 <- leader_2@parameters

# En ambos casos, observamos que el mejor modelo corresponde con un modelo de Gradient Boosting
# Analicemos las diferencias entre sus parametros
parameters_1[sapply(names(parameters_1), function(x) !identical(parameters_1[[x]], parameters_2[[x]]))]

parameters_1
parameters_2

#-- Parametros comunes:
#   sample_rate: 0.8
#   col_sample_rate: 0.8
#   col_sample_rate_per_tree: 0.8
#   max_depth: 15
#   min_rows: 100
#   stopping_tolerance: 0.01306994
#   score_tree_interval: 5

#-- Se diferencian en el numero de arboles
#   Modelo 1: ntrees = 82
#   Modelo 2: ntrees = 72

#   Nos guardamos los modelos
modelo1 <- h2o.getModel("GBM_5_AutoML_20210422_174850")
modelo2 <- h2o.getModel("GBM_5_AutoML_20210422_182719")

gbm1 <- h2o.gbm(
  x = var_modelo1, y = target, training_frame = surgical_dataset_h, seed = 1234,
  sample_rate = 0.8, col_sample_rate = 0.8, col_sample_rate_per_tree = 0.8, 
  max_depth = 15, min_rows = 100, stopping_tolerance = 0.01306994,
  score_tree_interval = 5, ntrees = 82, nfolds = 5, keep_cross_validation_predictions = TRUE
)

gbm2 <- h2o.gbm(
  x = var_modelo1, y = target, training_frame = surgical_dataset_h, seed = 1234,
  sample_rate = 0.8, col_sample_rate = 0.8, col_sample_rate_per_tree = 0.8, 
  max_depth = 15, min_rows = 100, stopping_tolerance = 0.01306994,
  score_tree_interval = 5, ntrees = 72, nfolds = 5, keep_cross_validation_predictions = TRUE
)

# Podemos aspirar a 0.92 de AUC, aproximadamente

h2o.shutdown(prompt = FALSE)

