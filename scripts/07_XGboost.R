# ------------- XGboost ---------------
# Objetivo: elaborar el mejor modelo de XGboost de acuerdo
#           a los valores de prediccion obtenidos
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  
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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")

#-- Modelo 2
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Probamos con un tuneo inicial
#   Modelo 1
set.seed(1234)
xgbmgrid <- expand.grid(eta=c(0.1,0.05,0.03,0.01,0.001), nrounds=c(100,500,1000,5000),
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=1)

control<-trainControl(method = "cv",number=5,repeats = 5,
                      savePredictions = "all",classProbs=TRUE)
xgbm<- train(paste0(target,"~",paste0(var_modelo1, collapse = "+")),data=surgical_dataset,
             method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)
xgbm
plot(xgbm)

# Modelo 2
