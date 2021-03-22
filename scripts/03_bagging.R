# ------------- Bagging ---------------
# Objetivo: elaborar el mejor modelo de bagging de acuerdo
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

#--- Seleccion del numero de arboles
#--  Modelo 1
set.seed(1234)
rfbis.1<-randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age+baseline_osteoart,
                      data=surgical_dataset,
                      mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

#-- Modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~Age+mortality_rsi+ccsMort30Rate+bmi+ahrq_ccs,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

mostrar_err_rate(rfbis.2$err.rate[, 1], rfbis.1$err.rate[, 1]) 
                 # rfbis.2$test$err.rate[, 1], rfbis.1$test$err.rate[, 1])

#-- Ampliamos entre 0 y 2000 arboles
mostrar_err_rate(rfbis.2$err.rate[c(0:2000), 1], rfbis.1$err.rate[c(0:2000), 1]) 
                 #rfbis.2$test$err.rate[c(0:2000), 1], rfbis.1$test$err.rate[c(0:2000), 1])

#-- Ampliamos entre 0 y 1000 arboles
mostrar_err_rate(rfbis.2$err.rate[c(0:1000), 1], rfbis.1$err.rate[c(0:1000), 1])
                 #rfbis.2$test$err.rate[c(0:1000), 1], rfbis.1$test$err.rate[c(0:1000), 1])

#-- Conclusion: con 300 arboles parece ser suficiente

#--- Tuneo de modelos
n.trees.1 <- 500
n.trees.2 <- 800
# Sampsize maximo: (k-1) * n => (4/5) * 5854 = 4683.2 ~ 4600 obs.
# Conclusion: sampsize maximo: 4600 obs. (de forma aproximada)

#--  Modelo 1
sampsizes.1 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.1 <- list(5, 10, 20, 30, 40, 50, 60, 100, 150)

bagging_modelo1 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees.1, grupos = 5, repe = 5)
bagging_modelo1$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[1,]))
bagging_modelo1$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[2,]))

#-- Distribucion de la tasa de error
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=tasa,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 1)")
ggsave('./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo1.png')

#-- Distribucion del AUC
#  Parece buen candidato 20 nodesize y sampsize todos salvo 100
#-- Probaremos nodesize 20, 30 y 40
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=auc,
                 colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion del AUC por sampsizes y nodesizes (Modelo 1)")
ggsave('./charts/bagging/distribuciones/03_distribucion_auc_modelo1.png')

# Me decanto por 1500 (menor varianza en el caso de AUC, el resto de sampsizes presentan alta varianza)
nodesizes.1 <- list(20)
sampsizes.1 <- list(1500)
bagging_modelo1_2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees.1, grupos = 5, repe = 5)

# 1000-1500-2000 ofrecen buen resultado AUC (1500 y 2500 alta varianza en AUC pero bajo sesgo)
# En cambio, con un sampsize de 1000 presenta una tasa de fallos ligeramente superior
nodesizes.1 <- list(30)
sampsizes.1 <- list(1000, 1500, 2000)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 5)

# Con 1500-2000 filas se obtiene un AUC alto y con poca varianza y menor sesgo
nodesizes.1 <- list(40)
sampsizes.1 <- list(1500, 2000)
bagging_modelo1_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 5)

# ¿Merece la pena aumentar el nodesize a 50?
nodesizes.1 <- list(50)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4600)
bagging_modelo1_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 5)
# No merece demasiado la pena, dado que el AUC apenas supera el 0.910 en el mejor de los casos
# y la tasa de error el 0.10
rm(bagging_modelo1_5)

#-- Posible candidato:   nodesize 20 y sampsize 1500
#-- Posible candidato:   nodesize 30 y sampsize 1000-1500-2000
#-- Posibles candidatos: nodesize 40 y sampsize 1500-2000
#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo1 <- rbind(
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+1500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "30+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "30+1500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "30+2000", ],
  bagging_modelo1_4[bagging_modelo1_4$modelo == "40+1500", ],
  bagging_modelo1_4[bagging_modelo1_4$modelo == "40+2000", ]
)
#-- Distribucion de la tasa de error
#   Modelos candidatos: 20+2000 - 20+1500 - 40+2000 (orden descendente)
union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                   reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa)) +
        geom_boxplot(fill = "#4271AE", colour = "#1F3552",
                     alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/03_comparacion_final_tasa_modelo1_5rep.png")

#-- Distribucion del AUC
#   Mejor resultado: 20 + 1500 o 30+2000
union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = auc)) +
        geom_boxplot(fill = "#4271AE", colour = "#1F3552",
                     alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("AUC por modelo")
ggsave("./charts/bagging/03_comparacion_final_auc_modelo1_5rep.png")

# ¿Y si aumentamos el numero de repeticiones a 10?
# 20 + 1500 se sigue manteniendo en mejor AUC y menor tasa de fallos, aunque con mayor varianza (ligeramente)

#-- Conclusion: eligo nodesize 30 + sampsize 2000 (ante la duda y al existir diferencias muy pequeñas, elegimos el modelo mas sencillo
#   dado que con nodesize 30 se obtienen arboles de menor profundidad que con nodesize 20) aunque tenemos la duda si 20 + 1500

# MODELO 2
sampsizes.2 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.2 <- list(5, 10, 20, 30, 40, 50, 60, 100, 150)

bagging_modelo2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo2,
                                 nodesizes = nodesizes.2,
                                 sampsizes = sampsizes.2, mtry = mtry.2,
                                 ntree = n.trees.2, grupos = 5, repe = 5)

bagging_modelo2$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2$modelo, '+', fixed = T))[1,]))
bagging_modelo2$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2$modelo, '+', fixed = T))[2,]))

#-- Distribucion de la tasa de error
ggplot(bagging_modelo2, aes(x=factor(sampsizes), y=tasa,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion del AUC por sampsizes y nodesizes (Modelo 2)")
ggsave("./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo2.png")

#-- Distribucion del AUC
ggplot(bagging_modelo2, aes(x=factor(sampsizes), y=auc,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion del AUC por sampsizes y nodesizes (Modelo 2)")
ggsave("./charts/bagging/distribuciones/03_distribucion_auc_modelo2.png")

#   Parece buen candidato 20, 30 o 40 nodesize y sampsize todos salvo 100
#   Dada una variabilidad menor, una posibilidad seria escoger sampsize 1000-1500 (aunque 1500 presenta una mayor varianza)
nodesizes.2 <- list(20)
sampsizes.2 <- list(1000, 1500)
bagging_modelo2_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 5)

# En relacion con 30 (nodesize), parece una mejor alternativa escoger sampsize = 2000 (tanto en tasa de error como en AUC, especialmente en variabilidad)
nodesizes.2 <- list(30)
sampsizes.2 <- list(2000)
bagging_modelo2_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 5)

# Nuevamente, una posible alternativa seria emplear sampsize = 1500 o 2000 
# (aunque en el caso de sampsize 1500 presenta mayor varianza, aunque no muy significativa)
nodesizes.2 <- list(40)
sampsizes.2 <- list(1500, 2000)
bagging_modelo2_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 5)
 
# A medida que aumenta el nodesize la variabilidad tanto en sesgo como en varianza disminuye
# ¿Mereceria la pena aumentar nodesize a 50? ¿O disminuir nodesize a 10?
nodesizes.2 <- list(50)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4600)
bagging_modelo2_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 5)
# No parece merecer demasiado la pena (el mejor modelo presenta un AUC de 0.910 (aproximado) y tasa de fallos 0.10, similar con nodesize 40)
# Por otro lado, reduciendo nodesize a 10 obtenemos modelos con mayor varianza en relacion al valor AUC
rm(bagging_modelo2_5)

#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo2 <- rbind(
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1000", ],
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1500", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "30+2000", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "40+1500", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "40+2000", ]
)

#-- Distribucion de la tasa de error
#   Modelos candidatos: 30 + 2000, 20 + 1500 y 40 + 2000 (en orden descendente)
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/03_comparacion_final_tasa_modelo2_5rep.png")

#-- Distribucion del AUC
#   20 + 1000 parece buen candidato, pero el es el modelo con mayor tasa de fallo (aunque la escala
#   del grafico indique que la diferencia sea de tan solo unas milesimas)
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = auc)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/03_comparacion_final_auc_modelo2_5rep.png")

#-- ¿Y con 10 rep?
#   El modelo 30 + 2000 continua presentando menor tasa de fallos frente al modelo 20 + 1000


#-- Conclusion: nos decantamos por nodesize 30 + 2000
#-- Modelos finales
#   Bagging (modelo 1 con 6 variables): nodesize 30 + sampsize 2000 (o 20 + 1500?)
#   Bagging (modelo 2 con 5 variables): nodesize 30 + sampsize 2000

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

# Aplico caret y construyo modelos finales
control <- trainControl(method = "repeatedcv",number=5,repeats=5,
                        savePredictions = "all",classProbs=TRUE) 

#-- Modelo 1
rfgrid.1 <-expand.grid(mtry=mtry.1)
set.seed(1234)
bagging_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                  data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.1,
                  nodesize = 30, sampsize = 2000, ntree = n.trees.1, replace = TRUE)

matriz_conf_1 <- matriz_confusion_predicciones(bagging_1, NULL, surgical_test_data, 0.5)

#-- Modelo 2
rfgrid.2 <-expand.grid(mtry=mtry.2)
set.seed(1234)
bagging_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                   data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.2,
                   nodesize = 30, sampsize = 2000, ntree = n.trees.2, replace = TRUE)

matriz_conf_2 <- matriz_confusion_predicciones(bagging_2, NULL, surgical_test_data, 0.5)
rm(rfgrid.1)
rm(rfgrid.2)

#--- Predicciones
#     Modelo 1 (20 + 1500)
#     Reference
#     Prediction   No  Yes
#     No         6496  98
#     Yes        738  1449

#     Modelo 1 (30 + 2000)
#     Reference
#     Prediction   No  Yes
#     No         6509  85
#     Yes        748  1439

# No hay demasiada diferencia entre ambos, por lo que nos podriamos decantar por un modelo "mas sencillo"

#     Modelo 2
#     Reference
#     Prediction   No  Yes
#     No         6512  82
#     Yes        752  1435
#

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "bagging"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)

#-- ¿Y si lo probamos sin reemplazamiento?
nodesizes.final <- list(30)
sampsizes.final <- list(2000)
bagging_modelo_sin_reemp <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.final,
                                   sampsizes = sampsizes.final, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 5, replace = FALSE)
bagging_modelo_sin_reemp$modelo <- "BAGGING MODELO 1 (no rep)"
modelos_actuales <- rbind(modelos_actuales, bagging_modelo_sin_reemp)
                          
modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
ggplot(modelos_actuales, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/comparativas/03_log_avnnet_bagging_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
ggplot(modelos_actuales, aes(x = modelo, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/comparativas/03_log_avnnet_bagging_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#  bagging modelo 2          bagging modelo 2
#  bagging modelo 1          bagging modelo 2
#   avnnet modelo 1           avnnet modelo 2
#   avnnet modelo 2           avnnet modelo 1
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/Bagging.RData")


