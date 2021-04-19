# ------------- Ensamblado ---------------
# Objetivo: elaborar el mejor modelo de ensamblado en base
#           a los mejores modelos obtenidos anteriormente
# Autor: Alberto Fernandez Hernandez

# Elegimos la seleccion de variables nº 2
#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  library(ggrepel)       # Labels ggplot2
  
  source("./librerias/cruzadas ensamblado binaria fuente.R")
})

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 2
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

# Importante: para el ensamblado no vamos a unir algoritmos "malos", sino las mejores versiones de cada
# uno de ellos. Ademas, se benefician de aquellos algoritmos que no presentan una alta correlación entre si.
# Ejemplo: redes neuronales y un modelo random forest (se benefician entre si ya que son modelos con una
# naturaleza diferente que un modelo bagging y un modelo random forest, donde no podria reducirse tanto la 
# varianza).

#-- Tuneo de los modelos finales
#   Regresion logistica
logistica <- cruzadalogistica(data=surgical_dataset, vardep=target,
                              listconti=var_modelo2, listclass=c(""),
                              grupos=grupos,sinicio=1234,repe=repe)

medias_logistica    <- as.data.frame(logistica[1])
medias_logistica$modelo <-"Logistica"
pred_logistica      <- as.data.frame(logistica[2])
pred_logistica$logi <- pred_logistica$Yes

#  Avnnet
avnnet  <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                             listconti=var_modelo2, listclass=c(""),
                             grupos=grupos,sinicio=sinicio,repe=repe, 
                             size=10,decay=0.01,repeticiones=5,itera=200)

medias_avnnet    <- as.data.frame(avnnet[1])
medias_avnnet$modelo <-"Avnnet"
pred_avnnet      <- as.data.frame(avnnet[2])
pred_avnnet$logi <- pred_avnnet$Yes

#  Bagging
bagging <- cruzadarfbin(data=surgical_dataset, vardep=target,
                        listconti=var_modelo2,listclass=c(""),
                        grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                        mtry=4,ntree=900, sampsize=1000, replace = TRUE)

medias_bagging    <- as.data.frame(bagging[1])
medias_bagging$modelo <-"Bagging"
pred_bagging      <- as.data.frame(bagging[2])
pred_bagging$logi <- pred_bagging$Yes

#  Random Forest
random_forest <- cruzadarfbin(data=surgical_dataset,
                              vardep=target,listconti=var_modelo2,
                              listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                              mtry=2,ntree=2000,nodesize=20,replace=TRUE)

medias_random_forest    <- as.data.frame(random_forest[1])
medias_random_forest$modelo <-"Random_Forest"
pred_random_forest      <- as.data.frame(random_forest[2])
pred_random_forest$logi <- pred_random_forest$Yes

# Gradient Boosting
gradient_boosting <- cruzadagbmbin(data=surgical_dataset,
                                   vardep=target,listconti=var_modelo2,
                                   listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                   n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                   interaction.depth=2, bag.fraction=0.5)

medias_gradient_boosting    <- as.data.frame(gradient_boosting[1])
medias_gradient_boosting$modelo <-"Gradient_Boosting"
pred_gradient_boosting      <- as.data.frame(gradient_boosting[2])
pred_gradient_boosting$logi <- pred_gradient_boosting$Yes

# XGboost
xgboost <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
           listconti=var_modelo2,listclass=c(""),
           grupos=grupos,sinicio=sinicio,repe=repe,
           min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
           gamma=0,colsample_bytree=1,subsample=0.5)

medias_xgboost    <- as.data.frame(xgboost[1])
medias_xgboost$modelo <-"XGboost"
pred_xgboost      <- as.data.frame(xgboost[2])
pred_xgboost$logi <- pred_xgboost$Yes
















