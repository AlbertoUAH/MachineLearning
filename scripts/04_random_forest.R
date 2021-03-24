# ------------- Bagging ---------------
# Objetivo: elaborar el mejor modelo de Random Forest de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           sampsize, 
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(randomForest)  # Seleccion del numero de arboles
  library(readxl)        # Lectura de ficheros Excel
  
  source("./librerias/librerias_propias.R")
})

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"

#--- Variables de los modelos candidatos
#--  Modelo 1
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", 
                 "Age", "baseline_osteoart")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 8 para el modelo 1 y 5 para el modelo 2
mtry.1 <- 6
mtry.2 <- 5

#-- Importancia de las variables
#-  Recordemos el caso generico de bagging
#   Modelo 1
rf_modelo_1_general <- train_rf_model(surgical_dataset, 
                             as.formula(paste0(target, "~",paste0(var_modelo1, collapse = "+"))),
                             mtry = mtry.1, ntree = 500, grupos = 5, repe = 5, nodesize = 30,
                             sampsize=2000,seed = 1234)

show_vars_importance(rf_modelo_1_general, "Importancia variables modelo 1 (Bagging)")

rf_modelo_2_general <- train_rf_model(surgical_dataset, 
                                      as.formula(paste0(target, "~",paste0(var_modelo2, collapse = "+"))),
                                      mtry = mtry.2, ntree = 800, grupos = 5, repe = 5, nodesize = 30,
                                      sampsize=2000,seed = 1234)

show_vars_importance(rf_modelo_2_general, "Importancia variables modelo 2 (Bagging)")







