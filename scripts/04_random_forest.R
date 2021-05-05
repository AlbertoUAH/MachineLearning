# ------------- Random Forest ---------------
# Objetivo: elaborar el mejor modelo de Random Forest de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           sampsize, nodesize y mtry
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
#   Mejor modelo bagging (modelo 1): nodesize 20 + sampsize 1000
#   Mejor modelo bagging (modelo 2): nodesize 20 + sampsize 1000
#   En ambos modelos, el numero de arboles empleado ha sido de 900


#-- ¿Varia el numero de arboles con respecto al mtry?
#   Modelo 1
#   Con mtry = 2 se estabiliza con alrededor de 2500 arboles
set.seed(1234)
err.rates.1 <- randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                            data=surgical_dataset,
                            mtry=mtry.1,ntree=5000,nodesize=20,replace=TRUE)$err.rate[, 1]

plot(err.rates.1, col = rgb(1,0,0, alpha = 0.5), type = 'l', 
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
#   Por ello, vamos a realizar un primer entrenamiento con 2500 arboles, tanto para mtry = 2, 3, 4 o 5
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
nodesizes.1 <- list(20)
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
#-- Posible candidato:   mtry 2-3, nodesize 20 y sampsize 500-1000-1500
union_bagging_modelo1 <- rbind(
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+1", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+500", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+1000", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+1500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1500", ]
)
union_bagging_modelo1$mtry <- c(rep("mtry2", 40), rep("mtry3", 40))

union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_tasa_modelo1_5_folds.png")

union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = auc, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_auc_modelo1_5_folds.png")

#-- ¿Por qué modelo bagging nos decantamos? Si aumentamos los grupos a 10
#   y las repeticiones a 20
nodesizes.1 <- list(20)
sampsizes.1 <- list(1, 500, 1000, 1500)
bagging_modelo1_2_bis <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 2,
                                   ntree = 2500, grupos = 10, repe = 20)

bagging_modelo1_3_bis <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = 3,
                                   ntree = 2000, grupos = 10, repe = 20)

union_bagging_modelo1_bis <- rbind(
  bagging_modelo1_2_bis[bagging_modelo1_2_bis$modelo == "20+1", ],
  bagging_modelo1_2_bis[bagging_modelo1_2_bis$modelo == "20+500", ],
  bagging_modelo1_2_bis[bagging_modelo1_2_bis$modelo == "20+1000", ],
  bagging_modelo1_2_bis[bagging_modelo1_2_bis$modelo == "20+1500", ],
  bagging_modelo1_3_bis[bagging_modelo1_3_bis$modelo == "20+1", ],
  bagging_modelo1_3_bis[bagging_modelo1_3_bis$modelo == "20+500", ],
  bagging_modelo1_3_bis[bagging_modelo1_3_bis$modelo == "20+1000", ],
  bagging_modelo1_3_bis[bagging_modelo1_3_bis$modelo == "20+1500", ]
)
union_bagging_modelo1_bis$mtry <- c(rep("mtry2", 80), rep("mtry3", 80))

union_bagging_modelo1_bis$modelo <- with(union_bagging_modelo1_bis,
                                           reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1_bis, aes(x = modelo, y = tasa, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_tasa_modelo1_10_folds.png")

union_bagging_modelo1_bis$modelo <- with(union_bagging_modelo1_bis,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1_bis, aes(x = modelo, y = auc, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_auc_modelo1_10_folds.png")


# Empleando 20 + 1000 obtenemos una tasa de fallos por debajo de 0.1
# ¿Merece la pena aumentar el sampsize? Podemos probar con el OOB
control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE) 

set.seed(1234)
bagging_1_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                   data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                   nodesize = 20, sampsize = 1000, ntree = 2000, replace = TRUE)
1 - bagging_1_1$finalModel$err.rate[2000, 1] # 0.8927229

set.seed(1234)
bagging_1_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                     nodesize = 20, sampsize = 1500, ntree = 2000, replace = TRUE)
1 - bagging_1_2$finalModel$err.rate[2000, 1] # 0.8971643

set.seed(1234)
bagging_1_3 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=3),
                     nodesize = 20, sampsize = 500, ntree = 2000, replace = TRUE)
1 - bagging_1_3$finalModel$err.rate[2000, 1] # 0.8782029

# No hay demasiada diferencia con respecto al OOB error. No obstante, podemos
# escoger un sampsize de 1500 y "asegurarnos" el 0.89 de accuracy

#-- Modelo 2
set.seed(1234)
err.rates.2 <- randomForest(factor(target)~Age+mortality_rsi+bmi+month.8,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]

plot(err.rates.2, col = rgb(1,0,0,alpha = 0.5), type = 'l', 
     main = 'Error rate by nº trees (Modelo 2)', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))
colors = c("darkgreen", "blue")
for(mtry in c(2,3)) {
  set.seed(1234)
  err.rates.aux <- randomForest(factor(target)~Age+mortality_rsi+bmi+month.8,
                              data=surgical_dataset,
                              mtry=mtry,ntree=5000,nodesize=10,replace=TRUE)$err.rate[, 1]
  
  lines(err.rates.aux, col = colors[mtry-1], type = 'l')
  abline(h = err.rates.aux[2000], lty = 'dashed')
  rm(err.rates.aux)
}
legend("top", legend = c("MTRY = 4","MTRY = 2", "MTRY = 3") , 
       col = c('red', 'darkgreen', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)
rm(colors)

# Por lo general, todos los modelos parecen estabilizarse a partir de 2000 arboles
#   mtry  Accuracy   Kappa    AccuracySD  KappaSD
#    1   0.8057384 0.3507719 0.007425563 0.02916395
#    2   0.8991789 0.7117563 0.007590889 0.02273964
#    3   0.9002382 0.7148823 0.008390515 0.02522118
#    4   0.8998619 0.7136014 0.008471527 0.02536614
# Observamos que no es necesario emplear mtry = 4 para obtener un buen accuracy
# sino que con mtry = 2 o mtry = 3 podemos obtener resultados similares
mtry.2 <- c(1,2,3,4)
cruzadarfbin(data=surgical_dataset, vardep=target,listconti=var_modelo2,listclass=c(""),
             grupos=5,sinicio=1234,repe=5,nodesize=10,mtry=mtry.2,ntree=2000,replace=TRUE)

# Posibles candidatos: mtry = 2 o mtry = 3
# Con mtry = 2
# Observamos los nodesizes y sampsizes para cada mtry
# De hecho, la tasa de error con mtry = 4 y mtry = 5 es muy similar
sampsizes.2 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.2 <- list(5, 10, 20, 30, 40, 50, 100)

# Probamos con mtry = 2
bagging_modelo2_mtry2 <-  tuneo_bagging(surgical_dataset, target = target,
                                        lista.continua = var_modelo2,
                                        nodesizes = nodesizes.2,
                                        sampsizes = sampsizes.2, mtry = 2,
                                        ntree = 2000, grupos = 5, repe = 5)

bagging_modelo2_mtry2$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2_mtry2$modelo, '+', fixed = T))[1,]))
bagging_modelo2_mtry2$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2_mtry2$modelo, '+', fixed = T))[2,]))

rf_stats_distribution(bagging_modelo2_mtry2, title.1 = "Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 2) mtry = 2",
                      title.2 = "Distribucion del AUC por sampsizes y nodesizes (Modelo 2) mtry = 2", 
                      path.1 = "./charts/random_forest/distribuciones/04_distribucion_tasa_error_modelo2_mtry2.png",
                      path.2 = "./charts/random_forest/distribuciones/04_distribucion_auc_modelo2_mtry2.png")

bagging_modelo2_mtry3 <-  tuneo_bagging(surgical_dataset, target = target,
                                        lista.continua = var_modelo2,
                                        nodesizes = nodesizes.2,
                                        sampsizes = sampsizes.2, mtry = 3,
                                        ntree = 2000, grupos = 5, repe = 5)

bagging_modelo2_mtry3$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2_mtry3$modelo, '+', fixed = T))[1,]))
bagging_modelo2_mtry3$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2_mtry3$modelo, '+', fixed = T))[2,]))

rf_stats_distribution(bagging_modelo2_mtry3, title.1 = "Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 2) mtry = 3",
                      title.2 = "Distribucion del AUC por sampsizes y nodesizes (Modelo 2) mtry = 3", 
                      path.1 = "./charts/random_forest/distribuciones/04_distribucion_tasa_error_modelo2_mtry3.png",
                      path.2 = "./charts/random_forest/distribuciones/04_distribucion_auc_modelo2_mtry3.png")

# En ambos casos, tanto con mtry = 2 como mtry = 3 los resultados son bastante similares
# Vamos con mtry 2 (aumentando a 10 repeticiones para observar)
nodesizes.2 <- list(20)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500, 3000)
bagging_modelo2_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = 2,
                                   ntree = 2000, grupos = 5, repe = 10)

# Vamos con mtry 3 (aumentando a 10 repeticiones para observar)
bagging_modelo2_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = 3,
                                   ntree = 2000, grupos = 5, repe = 10)

union_bagging_modelo2 <- rbind(
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1", ],
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+500", ],
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1000", ],
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1500", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+1", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+500", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+1000", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+1500", ]
)
union_bagging_modelo2$mtry <- c(rep("mtry2", 40), rep("mtry3", 40))

union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = tasa, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_tasa_modelo2_5_folds.png")

union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = auc, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_auc_modelo2_5_folds.png")

# ¿Y si lo llevamos a un extremo, aumentando el numero de grupos a 10 y repeticiones a 20?
nodesizes.2 <- list(20)
sampsizes.2 <- list(1, 500, 1000, 1500)
bagging_modelo2_2_bis <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = 2,
                                   ntree = 2000, grupos = 10, repe = 20)

bagging_modelo2_3_bis <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = 3,
                                   ntree = 2000, grupos = 10, repe = 20)

union_bagging_modelo2_bis <- rbind(
  bagging_modelo2_2_bis[bagging_modelo2_2_bis$modelo == "20+1", ],
  bagging_modelo2_2_bis[bagging_modelo2_2_bis$modelo == "20+500", ],
  bagging_modelo2_2_bis[bagging_modelo2_2_bis$modelo == "20+1000", ],
  bagging_modelo2_2_bis[bagging_modelo2_2_bis$modelo == "20+1500", ],
  bagging_modelo2_3_bis[bagging_modelo2_3_bis$modelo == "20+1", ],
  bagging_modelo2_3_bis[bagging_modelo2_3_bis$modelo == "20+500", ],
  bagging_modelo2_3_bis[bagging_modelo2_3_bis$modelo == "20+1000", ],
  bagging_modelo2_3_bis[bagging_modelo2_3_bis$modelo == "20+1500", ]
)
union_bagging_modelo2_bis$mtry <- c(rep("mtry2", 80), rep("mtry3", 80))

union_bagging_modelo2_bis$modelo <- with(union_bagging_modelo2_bis,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo2_bis, aes(x = modelo, y = tasa, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_tasa_modelo2_10_folds.png")

union_bagging_modelo2_bis$modelo <- with(union_bagging_modelo2_bis,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo2_bis, aes(x = modelo, y = auc, col = mtry)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/random_forest/bis_03_comparacion_final_auc_modelo2_10_folds.png")

# Parece una buena alternativa mtry = 2 y sampsize 1500, con una tasa de fallos inferior a 0.1
# ¿Merece la pena aumentar el sampsize? Podemos probar con el OOB
control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE) 
set.seed(1234)
bagging_2_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=2),
                     nodesize = 20, sampsize = 1000, ntree = 2000, replace = TRUE)
1 - bagging_2_1$finalModel$err.rate[2000, 1] # 0.8932354

set.seed(1234)
bagging_2_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=2),
                     nodesize = 20, sampsize = 1500, ntree = 2000, replace = TRUE)
1 - bagging_2_2$finalModel$err.rate[2000, 1] # 0.897506

set.seed(1234)
bagging_2_3 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = expand.grid(mtry=2),
                     nodesize = 20, sampsize = 500, ntree = 2000, replace = TRUE)
1 - bagging_2_3$finalModel$err.rate[2000, 1] # 0.8833276

#-- Modelos finales
#   Modelo 1
bagging_1_1$finalModel
#   Modelo 2
bagging_2_1$finalModel

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

matriz_conf_1 <- matriz_confusion_predicciones(bagging_1_1, NULL, surgical_test_data, 0.5)
matriz_conf_1_2 <- matriz_confusion_predicciones(bagging_1_2, NULL, surgical_test_data, 0.5)

matriz_conf_2 <- matriz_confusion_predicciones(bagging_2_1, NULL, surgical_test_data, 0.5)
matriz_conf_2_2 <- matriz_confusion_predicciones(bagging_2_2, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No   Yes
# No          6442  152       vs 6447  147
# Yes         735   1452      vs 729  1458

# Modelo 2
# Reference
# Prediction  No  Yes
# No         6457 137         vs 6480 114
# Yes        773 1414         vs 773 1414

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "random_forest"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)

#-- ¿Y si lo probamos sin reemplazamiento? Probamos con el mejor modelo en terminos de AUC (modelo 1)
bagging_modelo_sin_reemp <- tuneo_bagging(surgical_dataset, target = target,
                                          lista.continua = var_modelo1,
                                          nodesizes = 20,
                                          sampsizes = 1000, mtry = 3,
                                          ntree = 2000, grupos = 5, repe = 10, replace = FALSE)
bagging_modelo_sin_reemp$modelo <- "RF. MODELO 1 (no reemp)"
modelos_actuales <- rbind(modelos_actuales, bagging_modelo_sin_reemp)
modelos_actuales$tipo <- c(rep("LOGISTICA", 20), rep("RED NEURONAL", 20), rep("BAGGING", 20), rep("RANDOM FOREST", 20))

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
p <- ggplot(modelos_actuales, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo") +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))
p
ggsave('./charts/comparativas/03_log_avnnet_bagging_rf_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
t <- ggplot(modelos_actuales, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo") +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))
t
ggsave('./charts/comparativas/03_log_avnnet_bagging_rf_auc.jpeg')

#-- Si hacemos zoom sobre los modelos bagging y rf...
modelos_actuales_zoomed <- modelos_actuales[modelos_actuales$modelo %in% c("BAG. MODELO 1", "BAG. MODELO 2", "RF. MODELO 1",
                                                                           "RF. MODELO 2"), ]
modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,tasa, mean))

p <- ggplot(modelos_actuales_zoomed, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (solo BAGGING)") +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))

ggsave('./charts/random_forest/03_FINAL_tasa.jpeg')

modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,auc, mean))

t <- ggplot(modelos_actuales_zoomed, aes(x = modelo, y = auc, col = tipo)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("AUC por modelo (solo BAGGING)") +
            theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))

ggsave('./charts/random_forest/03_FINAL_auc.jpeg')

# No existe diferencia muy significativa entre bagging y random forest
#---- Estadisticas
# Por tasa fallos --------------- auc
#  bagging modelo 2            rf. modelo 1
#  bagging modelo 1           bagging modelo 1
#   avnnet modelo 2           bagging modelo 2
#      rf. modelo 2            rf. modelo 2
#      rf. modelo 1           avnnet modelo 1
#   avnnet modelo 1           avnnet modelo 2
#   log.   modelo 1           log.   modelo 1
#   log.   modelo 2           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/RandomForest.RData")



