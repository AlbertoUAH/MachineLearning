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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 8 para el modelo 1 y 5 para el modelo 2
mtry.1 <- 5
mtry.2 <- 5

#--- Seleccion del numero de arboles
#--  Modelo 1
set.seed(1234)
rfbis.1<-randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                      data=surgical_dataset,
                      mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

#-- Modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~Age+mortality_rsi+ccsMort30Rate+bmi+ahrq_ccs,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

#-- Aparentemente, con menos de 1000 arboles el error se estabiliza en ambos modelos...
mostrar_err_rate(rfbis.2$err.rate[, 1], rfbis.1$err.rate[, 1]) 

#-- ...Ampliamos entre 0 y 2000 arboles ¿Puede que se estabilice con 800-900 arboles?
mostrar_err_rate(rfbis.2$err.rate[c(0:3000), 1], rfbis.1$err.rate[c(0:3000), 1]) 

#-- Ampliamos entre 0 y 1000 arboles. Teniendo en cuenta el valor del eje Y, 
#   con 800 arboles el error se estabiliza
mostrar_err_rate(rfbis.2$err.rate[c(0:1000), 1], rfbis.1$err.rate[c(0:1000), 1])
                 
#-- El error en ambos modelos parece estabilizarse a partir de 800 arboles

#--- Tuneo de modelos
n.trees.1 <- 800
n.trees.2 <- 800
# Sampsize maximo: (k-1) * n => (4/5) * 5854 = 4683.2 ~ 4600 obs.
# Conclusion: sampsize maximo: 4600 obs. (de forma aproximada)

#--  Modelo 1
sampsizes.1 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.1 <- list(5, 10, 20, 30, 40, 50, 100)

bagging_modelo1 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees.1, grupos = 5, repe = 5)
bagging_modelo1$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[1,]))
bagging_modelo1$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[2,]))

#-- Primera observacion con el parametro nodesize
#   5 - 0.8819609 ; 10 - 0.884011 ; 20 - 0.8840113 ; 30 - 0.8826471 ; 40 - 0.8754697 ; 50 - 0.8788897 ; 60 - 0.8826444
#   Como primera impresion empleando un unico arbol, el hecho de aumentar el nodesize no parece mejorar el modelo
#   Aunque no podemos fiarnos de ello, dado que solo estamos elaborando un solo arbol
#   A simple vista, parece recomendar nodesize de 10-20
best_minbucket_dt(surgical_dataset, as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))), 
                  minbuckets = c(5, 10, 20, 30, 40, 50, 60), grupos = 5, repe = 10)


#-- Distribucion de la tasa de error
#   Con sampsize 500-2000, empleando 5-10-20-30 nodesize se obtiene practicamente la misma tasa de error
#   que con el conjunto total de observaciones (parece ser suficiente)
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=tasa,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion de la tasa de error por sampsizes y nodesizes (Modelo 1)")
ggsave('./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo1.png')

#-- Distribucion del AUC
#   Nuevamente, con sampsize 500-2000, a tasa AUC empleando 5-10-20-30 nodesize es similar a la obtenida con
#   el conjunto total de surgical_dataset (parece ser suficiente)
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=auc,
                 colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion del AUC por sampsizes y nodesizes (Modelo 1)")
ggsave('./charts/bagging/distribuciones/03_distribucion_auc_modelo1.png')

# ¿Y en relacion al nodesize? Podemos probar inicialmente con 10, 20 y 30

# Nodesize 10: 1000-1500-2000 parecen ser una buena opcion
nodesizes.1 <- list(10)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees.1, grupos = 5, repe = 10)

# Nodesize 20: 1000-1500-2000 parecen ser una buena opcion
# Sin embargo, el hecho de aumentar el numero de nodos no mejora la precision
# e incluso en determinados modelos los iguala
nodesizes.1 <- list(20)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10)

# Nodesize 30: No merece la pena aumentar el tamaño del arbol
nodesizes.1 <- list(50)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10)

# ¿Merece la pena aumentar el nodesize a 50? ¿O disminuirlo a 5?
nodesizes.1 <- list(5)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10)
rm(bagging_modelo1_5)
# En cualquiera de los casos, no merece la pena ni aumentar ni reducir el numero de
# nodesize, especialmente en el caso de nodesize = 5, dado que obtenemos arboles
# mas complejos que con nodesize = 10

#-- Posible candidato:   nodesize 10 y sampsize 500-1000-2000
#-- Posible candidato:   nodesize 20 y sampsize 500-1000-2000
#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo1 <- rbind(
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+1", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+500", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+1000", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+2000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+2000", ]
)

union_bagging_modelo1$config <- rep("5_folds-10_rep", 80)

#-- Distribucion de la tasa de error
#   Modelos candidatos: con 1500-2000 submuestras, el modelo parece tener una menor
#   tasa de error, especialmente en el caso de nodesize 20
union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                   reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa, col = config)) +
        geom_boxplot(alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo1_5rep.png")

#-- Distribucion del AUC
#   Mejor resultado: 
union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = auc, col = config)) +
        geom_boxplot(alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("AUC por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo1_5rep.png")

#-- Para salir de dudas ¿Aumentamos el numero de grupos a 10?
# Nodesize 10
nodesizes.1 <- list(10)
sampsizes.1_aux <- list(1, 500, 1000, 2000)
bagging_modelo1_2_10folds <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1_aux, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 10, repe = 20)
rm(sampsizes.1_aux)

# Nodesize 20
nodesizes.1 <- list(20)
sampsizes.2_aux <- list(1, 500, 1000, 2000)
bagging_modelo1_3_10folds <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.2_aux, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 10, repe = 20)
rm(sampsizes.2_aux)

union_bagging_modelo1_10folds <- rbind(
  bagging_modelo1_2_10folds[bagging_modelo1_2_10folds$modelo == "10+1", ],
  bagging_modelo1_2_10folds[bagging_modelo1_2_10folds$modelo == "10+500", ],
  bagging_modelo1_2_10folds[bagging_modelo1_2_10folds$modelo == "10+1000", ],
  bagging_modelo1_2_10folds[bagging_modelo1_2_10folds$modelo == "10+2000", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+1", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+500", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+1000", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+2000", ]
)

union_bagging_modelo1_10folds$config <- rep("10_folds-20_rep", 80)
union_bagging_modelo1_final          <- rbind(union_bagging_modelo1,
                                              union_bagging_modelo1_10folds)

#-- Realizamos la comparacion final del modelo 1 con k = 5 y k = 10 grupos
union_bagging_modelo1_final$modelo <- with(union_bagging_modelo1_final,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1_final, aes(x = modelo, y = tasa, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo1_5_10_folds.png")

#-- Distribucion del AUC
#   Mejor resultado: 
union_bagging_modelo1_final$modelo <- with(union_bagging_modelo1_final,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo1_final, aes(x = modelo, y = auc, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo1_5_10_folds.png")

#-- Teniendo en cuenta resultados empiricos
#   -> Observamos que el tamaño optimo de muestra se situa entre 500 y 2000 (sampsize)
#   -> Diria que 10 + 2000 o 10 + 500 

#-- Curiosidad ¿Y si reducimos el sampsize por debajo de 500?
sampsizes.2_aux <- list(200, 300, 400, 500)
bagging_modelo1_3_10folds_v2 <- tuneo_bagging(surgical_dataset, target = target,
                                           lista.continua = var_modelo1,
                                           nodesizes = nodesizes.1,
                                           sampsizes = sampsizes.2_aux, mtry = mtry.1,
                                           ntree = n.trees.1, grupos = 10, repe = 20)
rm(sampsizes.2_aux)


#---------------------------------- Modelo 2 ----------------------------------
sampsizes.2 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.2 <- list(5, 10, 20, 30, 40, 50, 100)

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
  ggtitle("Distribucion de la tasa por sampsizes y nodesizes (Modelo 2)")
ggsave("./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo2.png")

#-- Distribucion del AUC
ggplot(bagging_modelo2, aes(x=factor(sampsizes), y=auc,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2, shape = 18) +
  ggtitle("Distribucion del AUC por sampsizes y nodesizes (Modelo 2)")
ggsave("./charts/bagging/distribuciones/03_distribucion_auc_modelo2.png")

# Del mismo modo que sucedia con el modelo 1, con sampsize 500-2000 se obtienen unos modelos cuyos valores
# de error y AUC son muy similares a los obtenidos con el conjunto total de observaciones. Ademas, podemos
# estudiar los modelos a partir de valores de nodesize como 10, 20 o 30, principalmente. Si necesitamos reducir
# el valor de nodesize o aumentarlo, lo comprobaremos.

# Nodesize 10: 
nodesizes.2 <- list(10)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10)

# Nodesize 20: 
nodesizes.2 <- list(20)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10)

# Nodesize 30: 
nodesizes.2 <- list(30)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10)
 
# A medida que aumenta el nodesize la variabilidad tanto en sesgo como en varianza disminuye
# ¿Mereceria la pena aumentar nodesize a 50? ¿O disminuir nodesize a 10?
nodesizes.2 <- list(10)
sampsizes.2 <- list(1, 1000, 1500, 2000, 2500, 3000)
bagging_modelo2_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10)

# Con nodesize 50 la varianza en el AUC aumenta significativamente con respecto a nodesizes anteriores
# Por el contario, un nodesize de 10 obtiene valores AUC "ligeramente" superiores, aunque las tasas de error
# son practicamente identicas a otros modelos como nodesize 20 ¿Merecera la pena establecer nodesize a 10?
# Escogemos 1000-1500 como sampsize

#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo2_bis <- rbind(
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1000", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "30+1500", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "30+2000", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "40+1000", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "40+1500", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "40+2000", ],
  bagging_modelo2_5[bagging_modelo2_5$modelo == "10+1000", ],
  bagging_modelo2_5[bagging_modelo2_5$modelo == "10+1500", ]
)

union_bagging_modelo2 <- rbind(union_bagging_modelo2, union_bagging_modelo2_bis)
union_bagging_modelo2$rep <- c(rep("5", 40), rep("10", 80))

#-- Distribucion de la tasa de error
#   Modelos candidatos: 10 + 1500, 30 + 2000, 30 + 1500 (en orden descendente)
#   En relacion a la varianza: 10 + 1500,  30 + 2000, 10 + 1000 o 20 + 1000 presentan menor varianza
#   aunque estos dos ultimos presenten una tasa de error "ligeramente" superior
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo2_10rep.png")

#-- Distribucion del AUC
#   En relacion con el valor AUC, muchos de los modelos presentan alta varianza
#   a excepcion de 20+1000, 30+2000, 40+2000 o 40+1000. De todos los modelos
#   anteriores, 30+2000 presenta la menor tasa de fallos. Por tasa de fallos, seria un modelo adecuado
#   Por elevado AUC: 20 + 1000, aunque su tasa de fallos sea ligeramente superior
#   Posibles candidatos (30 + 2000), (20 + 1000, aunque su tasa de error sea ligeramente mayor), (40+2000)
#   40 + 1000 tiene el problema de que presenta una alta varianza en la tasa de fallos
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo2_10rep.png")

# ¿Se mantienen dichas distribuciones con 10 repeticiones?
# Por tasa de fallos, se siguen manteniendo con poca varianza modelos como 10+1500, 30+2000, 10+1000
# En relacion con AUC, modelos como 10+1000 o 10+1500 presenta una mayor varianza en comparacion con modelos como
# 30+2000. De hecho, la mejoria que supone el modelo 20+1000 en relacion al AUC es de tan solo 25 milesimas en el
# valor maximo, teniendo en cuenta de que se trata de un modelo bagging con arboles de mayor profundidad (nodesize menor)
# pero que emplea menos submuestras para los arboles. Por ende, igual es mas interesante elaborar varios arboles de menor
# profundidad, pero que utilicen mas submuestras. Ademas, aunque la diferencia no sea excesiva es preferible el modelo 30+2000
# en relacion a la tasa de fallos, dado que es un modelo mas "predecible" que el modelo 20+1000

#-- Conclusion: nos decantamos por nodesize 30 + 2000
#-- Modelos finales
#   Bagging (modelo 1 con 5 variables): nodesize 20 + sampsize 1000
#   Bagging (modelo 2 con 5 variables): nodesize 30 + sampsize 2000

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

# Aplico caret y construyo modelos finales
control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE) 

#-- Modelo 1
rfgrid.1 <-expand.grid(mtry=mtry.1)
set.seed(1234)
bagging_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                  data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.1,
                  nodesize = 20, sampsize = 1000, ntree = n.trees.1, replace = TRUE)

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
#     Modelo 1 (20 + 1000)
#     Reference
#     Prediction   No  Yes
#     No         6470  124
#     Yes        743  1444

#     Modelo 2 (30 + 2000)
#     Reference
#     Prediction   No  Yes
#     No         6511  83
#     Yes        750  1437

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
                                   ntree = n.trees.2, grupos = 5, repe = 10, replace = FALSE)
bagging_modelo_sin_reemp$modelo <- "BAG. MODELO 2 (no reemp)"
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

#-- Si hacemos zoom sobre los modelos avnnet...
modelos_actuales_zoomed <- modelos_actuales[modelos_actuales$modelo %in% c("BAG. MODELO 1", "BAG. MODELO 2", "BAG. MODELO 2 (no reemp)"), ]
modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,tasa, mean))
ggplot(modelos_actuales_zoomed, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (solo BAGGING)")

ggsave('./charts/bagging/03_FINAL_tasa.jpeg')

modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,auc, mean))
ggplot(modelos_actuales_zoomed, aes(x = modelo, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (solo BAGGING)")

ggsave('./charts/bagging/03_FINAL_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#  bagging modelo 2          bagging modelo 1
#  bagging modelo 1          bagging modelo 2
#   avnnet modelo 1           avnnet modelo 2
#   avnnet modelo 2           avnnet modelo 1
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/Bagging.RData")


