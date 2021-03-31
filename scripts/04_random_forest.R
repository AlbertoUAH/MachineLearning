# ------------- Random Forest ---------------
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
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 5 para el primer modelo y 4 para el segundo
mtry.1 <- 5
mtry.2 <- 4

#-- Recordemos:
#   Mejor modelo bagging (modelo 1): nodesize 10 + sampsize 500
#   Mejor modelo bagging (modelo 2): nodesize 10 + sampsize 500
#   En ambos modelos, el numero de arboles empleado ha sido de 900


#-- ¿Varia el numero de arboles con respecto al mtry?
#   Modelo 1
#   Con mtry = 2 se estabiliza con alrededor de 2500 arboles
err.rates.1 <- review_ntrees(surgical_dataset, factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                             mtry = c(5), ntree = 5000, nodesize = 10, seed = 1234)

plot(err.rates.1[,1], col = "red", type = 'l', 
     main = 'Error rate by nº trees (Modelo 1)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))

colors = c("darkgreen", "blue", "purple")
for(mtry in c(2,3,4)) {
  err.rates.aux <- review_ntrees(surgical_dataset, factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                                 mtry = mtry, ntree = 5000, nodesize = 10, seed = 1234)
  lines(err.rates.aux[,1], col = colors[mtry-1], type = 'l')
  rm(err.rates.aux)
}
abline(h = 0.10221, lty = 'dashed')
legend("top", legend = c("MTRY = 5","MTRY = 2", "MTRY = 3", "MTRY = 4") , 
       col = c('red', 'darkgreen', 'blue', 'purple') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)
rm(colors)

#-- Cuantas menos variables se sorteen, mayor numero de arboles se necesitan para estabilizar 
#   Por ello, vamos a realizar un primer entrenamiento con 2000 arboles, tanto para mtry = 2, 3, 4 o 5
#     mtry  Accuracy  AccuracySD
# 1    1   0.815919  0.00754739
# 2    2   0.8939171 0.008603418
# 3    3   0.8980168 0.007877379
# 4    4   0.8986662 0.008056248
# 5    5   0.8981877 0.008463885
mtry.1 <- c(1,2,3,4,5)
cruzadarfbin(data=surgical_dataset, vardep=target,listconti=var_modelo1,listclass=c(""),
             grupos=5,sinicio=1234,repe=5,nodesize=10,mtry=mtry.1,ntree=2500,replace=TRUE)

# Con un valor mtry pequeño (2, 3) se obtiene valores de precision bastante altos y muy competitivos con mtry = 5
# Observamos los nodesizes y sampsizes para cada mtry
sampsizes.1 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.1 <- list(5, 10, 20, 30, 40, 50, 100)

# Probamos con mtry = 2
bagging_modelo1_mtry2 <-  tuneo_bagging(surgical_dataset, target = target,
                                        lista.continua = var_modelo1,
                                        nodesizes = nodesizes.1,
                                        sampsizes = sampsizes.1, mtry = 2,
                                        ntree = 2500, grupos = 5, repe = 5)

bagging_modelo1_mtry2$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry2$modelo, '+', fixed = T))[1,]))
bagging_modelo1_mtry2$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry2$modelo, '+', fixed = T))[2,]))

# ¿mtry = 3?
bagging_modelo1_mtry3 <-  tuneo_bagging(surgical_dataset, target = target,
                                        lista.continua = var_modelo1,
                                        nodesizes = nodesizes.1,
                                        sampsizes = sampsizes.1, mtry = 2,
                                        ntree = 2500, grupos = 5, repe = 5)

bagging_modelo1_mtry3$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry3$modelo, '+', fixed = T))[1,]))
bagging_modelo1_mtry3$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry3$modelo, '+', fixed = T))[2,]))


#   Modelo 2
#   Aparentemente, con el modelo 2 (para los tres valores) se estabilizan con alrededor de 800 arboles
err.rates.2 <- review_ntrees(surgical_dataset, factor(target)~Age+mortality_rsi+ccsMort30Rate+bmi+ahrq_ccs,
                             mtry = c(2), ntree = 5000, nodesize = 10, seed = 1234)

plot(err.rates.2[,1], col = 'red', type = 'l', 
     main = 'Error rate by nº trees (Modelo 2)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))
lines(err.rates.1[,2], col = 'darkorange2')
lines(err.rates.2[,3], col = 'blue')
lines(err.rates.2[,4], col = 'darkgreen')
legend("topright", legend = c("OOB: MODELO 2 - MTRY 5", "OOB: MODELO 2 - MTRY 2", "OOB: MODELO 2 - MTRY 3", "OOB: MODELO 2 - MTRY 4") , 
       col = c('red', 'blue', 'darkgreen') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

#-- Modelo 1
mtry.1 <- c(2,3,4,5)
n.trees.1 <- 800
#   Hacemos prueba con mtrys
rf_modelo1_5  <- data.frame()
rf_modelo1_10 <- data.frame()
for(i in seq(1, 4)) {
  rf_modelo1_aux <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = 10,
                                   sampsizes = 1, mtry = mtry.1[i],
                                   ntree = n.trees.1, grupos = 5, repe = 5)
  rf_modelo1_aux$modelo <- rep(paste0(mtry.1[i],"-",n.trees.1), 5)
  rf_modelo1_5 <- rbind(rf_modelo1_5, rf_modelo1_aux)
  
  rf_modelo1_aux <- tuneo_bagging(surgical_dataset, target = target,
                                  lista.continua = var_modelo1,
                                  nodesizes = 10,
                                  sampsizes = 1, mtry = mtry.1[i],
                                  ntree = n.trees.1, grupos = 5, repe = 10)
  rf_modelo1_aux$modelo <- rep(paste0(mtry.1[i],"-",n.trees.1), 10)
  rf_modelo1_10 <- rbind(rf_modelo1_10, rf_modelo1_aux)
}
rm(rf_modelo1_aux)

# mtry  Accuracy    Kappa  AccuracySD  KappaSD
#   2 0.8946689 0.7008582 0.008432239 0.02507914
#   3 0.8946689 0.7008582 0.008432239 0.02507914
#   4 0.8990415 0.7126108 0.008534504 0.02536611
#   5 0.8980511 0.7099556 0.008776248 0.02583543

#-- Tasa de fallos
ggplot(rf_modelo1_5, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill = "#4271AE", line = "#1F3552", alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

#-- AUC
ggplot(rf_modelo1_5, aes(x = modelo, y = auc)) +
  geom_boxplot(fill = "#4271AE", line = "#1F3552", alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")




