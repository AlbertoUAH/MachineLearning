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
#-- Modelo 2
set.seed(1234)
err.rates.1 <- randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                            data=surgical_dataset,
                            mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]

plot(err.rates.1, col = "red", type = 'l', 
     main = 'Error rate by nº trees (Modelo 1)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))

colors = c("darkgreen", "blue", "purple")
for(mtry in c(2,3,4)) {
  set.seed(1234)
  err.rates.aux <- randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                                data=surgical_dataset,
                                mtry=mtry,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]
  
  lines(err.rates.aux, col = colors[mtry-1], type = 'l')
  rm(err.rates.aux)
}
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
# De hecho, la tasa de error con mtry = 4 y mtry = 5 es muy similar
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

rf_stats_distribution(bagging_modelo1_mtry2, title.1 = "Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 1) mtry = 2",
                      title.2 = "Distribucion del AUC por sampsizes y nodesizes (Modelo 1) mtry = 2", 
                      path.1 = "./charts/random_forest/distribuciones/04_distribucion_tasa_error_modelo1_mtry2.png",
                      path.2 = "./charts/random_forest/distribuciones/04_distribucion_auc_modelo1_mtry2.png")


# ¿mtry = 3? A simple vista, se requieren de menos arboles (alrededor de 2000)
bagging_modelo1_mtry3 <-  tuneo_bagging(surgical_dataset, target = target,
                                        lista.continua = var_modelo1,
                                        nodesizes = nodesizes.1,
                                        sampsizes = sampsizes.1, mtry = 3,
                                        ntree = 2000, grupos = 5, repe = 5)

bagging_modelo1_mtry3$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry3$modelo, '+', fixed = T))[1,]))
bagging_modelo1_mtry3$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1_mtry3$modelo, '+', fixed = T))[2,]))

rf_stats_distribution(bagging_modelo1_mtry3, title.1 = "Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 1) mtry = 3",
                      title.2 = "Distribucion del AUC por sampsizes y nodesizes (Modelo 1) mtry = 3", 
                      path.1 = "./charts/random_forest/distribuciones/04_distribucion_tasa_error_modelo1_mtry3.png",
                      path.2 = "./charts/random_forest/distribuciones/04_distribucion_auc_modelo1_mtry3.png")

# Vamos con mtry 2 (aumentando a 10 repeticiones para observar)
nodesizes.1 <- list(10)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500, 3000)
bagging_modelo1_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 2,
                                   ntree = 2500, grupos = 5, repe = 10)

# Vamos con mtry 3 (aumentando a 10 repeticiones para observar)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 3,
                                   ntree = 2000, grupos = 5, repe = 10)

# Vamos con mtry 4 (aumentando a 10 repeticiones para observar)
bagging_modelo1_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 4,
                                   ntree = 1500, grupos = 5, repe = 10)

# Analizando las salidas resultantes, podemos comprobar que emplear un valor
# mtry muy pequeño implica aumentar el numero de muestras a sortear para obtener 
# resultados similares a los del RF original (sampsize 1). Por el contrario,
# aumentando el valor mtry se necesitan menos submuestras para obtener los mismos
# resultados, practicamente, que con el conjunto original con reemplazamiento (1)
# concretamente con alrededor de 1000-1500 observaciones
# Ademas, en relacion a mtry = 3 o 4, elegimos 3, dado que ademas de obtener unos
# resultados muy similares, sorteando menos variables es posible controlar la varianza
# en mejor medida
# Nos decantamos por mtry = 3
#-- Posible candidato:   mtry 3, nodesize 10 y sampsize 500-1000-1500
union_bagging_modelo1 <- rbind(
  bagging_modelo1_3[bagging_modelo1_3$modelo == "10+1", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "10+500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "10+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "10+1500", ]
)

union_bagging_modelo1$config <- rep("5_folds-10_rep", 40)

#-- ¿Por qué modelo bagging nos decantamos? Si aumentamos los grupos a 10
#   y las repeticiones a 20
nodesizes.1 <- list(10)
sampsizes.1 <- list(1, 500, 1000, 1500)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 3,
                                   ntree = 2000, grupos = 10, repe = 20)

bagging_modelo1_3$config <- rep("10_folds-20_rep", 40)
union_bagging_modelo1 <- rbind(union_bagging_modelo1, bagging_modelo1_3)

union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                           reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_tasa_modelo1_5_10_folds.png")

union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = auc, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_auc_modelo1_5_10_folds.png")


# Empleando 10 + 1000 obtenemos una tasa de fallos por debajo de 0.1
# ¿Merece la pena aumentar el sampsize? Podemos probar con el OOB
control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE) 

set.seed(1234)
bagging_1_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                   data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                   nodesize = 10, sampsize = 1000, ntree = 2000, replace = TRUE)
1 - bagging_1_1$finalModel$err.rate[2000, 1] # 0.8973352

set.seed(1234)
bagging_1_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                     nodesize = 10, sampsize = 1500, ntree = 2000, replace = TRUE)
1 - bagging_1_2$finalModel$err.rate[2000, 1] # 0.8992142

set.seed(1234)
bagging_1_3 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                     nodesize = 10, sampsize = 500, ntree = 2000, replace = TRUE)
1 - bagging_1_3$finalModel$err.rate[2000, 1] # 0.8853775

# No hay demasiada diferencia con respecto al OOB error. No obstante, podemos
# escoger un sampsize de 1000 y "asegurarnos" el 0.89 de accuracy

#-- Modelo 2
set.seed(1234)
err.rates.2 <- randomForest(factor(target)~Age+mortality_rsi+bmi+month.8,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]

plot(err.rates.2, col = "red", type = 'l', 
     main = 'Error rate by nº trees (Modelo 2)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))

colors = c("darkgreen", "blue")
for(mtry in c(2,3)) {
  set.seed(1234)
  err.rates.aux <- randomForest(factor(target)~Age+mortality_rsi+bmi+month.8,
                              data=surgical_dataset,
                              mtry=mtry,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]
  
  lines(err.rates.aux, col = colors[mtry-1], type = 'l')
  rm(err.rates.aux)
}
legend("top", legend = c("MTRY = 4","MTRY = 2", "MTRY = 3") , 
       col = c('red', 'darkgreen', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)
rm(colors)





