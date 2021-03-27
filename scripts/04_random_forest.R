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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 5 para ambos modelos
mtry.1 <- 5
mtry.2 <- 5

#-- Recordemos:
#   Mejor modelo bagging (modelo 1): nodesize 20 + sampsize 1000
#   Mejor modelo bagging (modelo 2): nodesize 30 + sampsize 2000
#   En ambos modelos, el numero de arboles empleado ha sido de 800


#-- ¿Varia el numero de arboles con respecto al mtry?
#   Modelo 1
err.rates.1 <- review_ntrees(surgical_dataset, factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                             mtry = c(3,4,5), ntree = 5000, nodesize = 10, seed = 1234)

plot(err.rates.1[,1], col = 'red', type = 'l', 
     main = 'Error rate by nº trees (Modelo 1)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))
lines(err.rates.1[,2], col = 'blue')
lines(err.rates.1[,3], col = 'darkgreen')
legend("topright", legend = c("OOB: MODELO 1 - MTRY 5","OOB: MODELO 1 - MTRY 3", "OOB: MODELO 1 - MTRY 4") , 
       col = c('red', 'blue', 'darkgreen') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

#   Modelo 2
err.rates.2 <- review_ntrees(surgical_dataset, factor(target)~Age+mortality_rsi+ccsMort30Rate+bmi+ahrq_ccs,
                             mtry = c(3,4,5), ntree = 5000, nodesize = 10, seed = 1234)

plot(err.rates.2[,1], col = 'red', type = 'l', 
     main = 'Error rate by nº trees (Modelo 2)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))
lines(err.rates.2[,2], col = 'blue')
lines(err.rates.2[,3], col = 'darkgreen')
legend("topright", legend = c("OOB: MODELO 2 - MTRY 5","OOB: MODELO 2 - MTRY 3", "OOB: MODELO 2 - MTRY 4") , 
       col = c('red', 'blue', 'darkgreen') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)






