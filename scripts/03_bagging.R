# ------------- Bagging ---------------
# Objetivo: elaborar el mejor modelo de bagging de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           sampsize, 
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)      # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)        # Paralelizacion de funciones (I)
  library(doParallel)      # Paralelizacion de funciones (II)
  library(caret)           # Recursive Feature Elimination
  library(randomForest)    # Seleccion del numero de arboles
  library(readxl)          # Lectura de ficheros Excel
  library(randomForestSRC) # Tuneado random forest
  
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
# En nuestro caso, 8 para el modelo 1 y 5 para el modelo 2
mtry.1 <- 5
mtry.2 <- 4

#--- Seleccion del numero de arboles
#--  Modelo 1
set.seed(1234)
rfbis.1<-randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,
                      data=surgical_dataset,
                      mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

#-- Modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~Age+mortality_rsi+bmi+month.8,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

rfbis1 <- as.data.frame(rfbis.1$err.rate); rfbis1$index <- c(1:5000)
rfbis2 <- as.data.frame(rfbis.2$err.rate); rfbis2$index <- c(1:5000)
colors <- c("Modelo 1" = "red", "Modelo 2" = "blue")
ggplot(NULL, aes(x = index, y = OOB)) + geom_line(data=rfbis2[c(1:2000), ], aes(col = "Modelo 1")) + geom_line(data=rfbis1[c(1:2000), ], aes(col = "Modelo 2")) + 
  ggtitle("OOB Error (Bagging) (up to 2000 trees)")  + scale_color_manual(values = colors) +
  theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/OOB_2.png")

#-- Aparentemente, con menos de 1000 arboles el error se estabiliza en ambos modelos...
mostrar_err_rate(rfbis.2$err.rate[, 1], rfbis.1$err.rate[, 1]) 

#-- ...Ampliamos entre 0 y 2000 arboles ¿Puede que se estabilice con 800-900 arboles?
mostrar_err_rate(rfbis.2$err.rate[c(1:3000), 1], rfbis.1$err.rate[c(1:3000), 1]) 

#-- Ampliamos entre 0 y 1000 arboles. Teniendo en cuenta el valor del eje Y, 
#   con 900 arboles el error se estabiliza
mostrar_err_rate(rfbis.2$err.rate[c(1:1000), 1], rfbis.1$err.rate[c(1:1000), 1])
                 
#-- El error en ambos modelos parece estabilizarse a partir de 900 arboles
#-- Con respecto al modelo 2, tenemos la duda de si se estabilizaria a partir de 900 o 3000 arboles

#--- Tuneo de modelos
n.trees.1 <- 900
n.trees.2 <- 900
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
bagging_modelo1_temp <- bagging_modelo1 %>% group_by(modelo, sampsizes, nodesizes) %>% summarise(tasa_fallos_media = mean(tasa))
p <- ggplot(bagging_modelo1_temp, aes(x=factor(sampsizes), y=tasa_fallos_media,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2.5, shape = 20) +
  scale_colour_manual(values=rainbow(7))+
  ggtitle("Tasa de fallos (Modelo 1)") +
  theme(text = element_text(size=13, face = "bold"))
ggsave('./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo1.png')

#-- Distribucion del AUC
#   Nuevamente, con sampsize 500-2000, a tasa AUC empleando 5-10-20-30 nodesize es similar a la obtenida con
#   el conjunto total de surgical_dataset (parece ser suficiente)
bagging_modelo1_temp <- bagging_modelo1 %>% group_by(modelo, sampsizes, nodesizes) %>% summarise(auc_media = mean(auc))
t <- ggplot(bagging_modelo1_temp, aes(x=factor(sampsizes), y=auc_media,
                 colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2.5, shape = 20) + 
  scale_colour_manual(values=rainbow(7))+
  ggtitle("AUC (Modelo 1)") +
  theme(text = element_text(size=13, face = "bold"))
ggsave('./charts/bagging/distribuciones/03_distribucion_auc_modelo1.png')
ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave('./charts/03_distribucion_auc_tasa_fallos_modelo1.png')


# ¿Y en relacion al nodesize? Podemos probar inicialmente con 10, 20 y 30

# Nodesize 10: con 500-1000-1500 o 2000 parecen ser una buena opcion
# AUC: en torno a 0.913-0.915, tasa de fallos: por debajo de 0.108
nodesizes.1 <- list(10)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees.1, grupos = 5, repe = 10, 
                                 show_nrnodes = "yes")
# 10 + 1: 597 ; 10 + 500: 123 ; 10 + 1000: 201 ; 10 + 1500: 253 ; 10 + 2000: 309 ; 10 + 2500: 355

# Nodesize 20: 500-1000-1500-2000 parecen ser una buena opcion
# Sin embargo, el hecho de aumentar el numero de nodos no mejora la precision
# AUC: maximo 0.916, tasa de fallos: menos de 0.109
nodesizes.1 <- list(20)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")
# 20 + 1: 457 ; 20 + 500: 85 ; 20 + 1000: 145 ; 20 + 1500: 187 ; 20 + 2000: 229 ; 20 + 2500: 275

# Nodesize 30: No merece la pena aumentar el tamaño del arbol
nodesizes.1 <- list(30)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")

# 30 + 1: 353 ; 30 + 500: 67 ; 30 + 1000: 119 ; 30 + 1500: 153 ; 30 + 2000: 205 ; 30 + 2500: 227

# ¿Merece la pena aumentar el nodesize a 50? ¿O disminuirlo a 5?
nodesizes.1 <- list(50)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo1_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")
rm(bagging_modelo1_5)
# 50 + 1: 269 ; 50 + 500: 47 ; 50 + 1000: 83 ; 50 + 1500: 113 ; 50 + 2000: 139 ; 50 + 2500: 161
# 5 + 1: 767 ; 5 + 500: 153 ; 5 + 1000: 253 ; 5 + 1500: 329 ; 5 + 2000: 391 ; 5 + 2500: 461
# 1 + 1: 1053 ; 1 + 500: 191 ; 1 + 1000: 317 ; 1 + 1500: 443 ; 1 + 2000: 523 ; 1 + 2500: 617

bagging_comp1 <- rbind(bagging_modelo1_3, bagging_modelo1_5); bagging_comp1$nodesize <- c(rep("20", 60), rep("50", 60)) 
bagging_comp1$modelo <- with(bagging_comp1, reorder(modelo,tasa, mean))
p <- ggplot(bagging_comp1, aes(x = factor(modelo), y = tasa, colour = nodesize, group = modelo)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
bagging_comp1$modelo <- with(bagging_comp1, reorder(modelo,auc, mean))
t <- ggplot(bagging_comp1, aes(x = factor(modelo), y = auc, colour = nodesize, group = modelo)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("AUC por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave('./charts/03_distribucion_auc_tasa_fallos_modelo1_comp_50rep.png')
randomForestSRC::tune.nodesize(formula = as.formula(paste0("target~", paste0(var_modelo1, collapse = "+"))), surgical_dataset,
                                                    nodesizeTry = c(5, 10, 20, 30, 50), mtryStart = 5, ntreeTry = 900,
                                                    sampsize = 500, seed = 1234, stepFactor=0)
# $nsize.opt
# [1] 5
# 
# $err
# nodesize       err
# 1        5 0.1627947
# 2       10 0.1807311
# 3       20 0.2020840
# 4       30 0.2104544
# 5       50 0.2210454

randomForestSRC::tune.nodesize(formula = as.formula(paste0("target~", paste0(var_modelo1, collapse = "+"))), surgical_dataset,
                               nodesizeTry = c(5, 10, 20, 30, 50), mtryStart = 5, ntreeTry = 900, sampsize = 1000, seed = 1234, stepFactor=0)
# $nsize.opt
# [1] 5
# 
# $err
# nodesize       err
# 1        5 0.1458832
# 2       10 0.1576700
# 3       20 0.1723608
# 4       30 0.1879057
# 5       50 0.2046464

# En cualquiera de los casos, no merece la pena ni aumentar ni reducir el numero de
# nodesize, especialmente en el caso de nodesize = 5, dado que obtenemos arboles
# mas complejos que con nodesize = 10

#-- Posible candidato:   nodesize 10 y sampsize 500-1000-1500
#-- Posible candidato:   nodesize 20 y sampsize 500-1000-1500
#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo1 <- rbind(
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+1", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+500", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+1000", ],
  bagging_modelo1_2[bagging_modelo1_2$modelo == "10+1500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+500", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "20+1500", ]
)

union_bagging_modelo1$config <- rep("5_folds-10_rep", 80)

#-- Distribucion de la tasa de error
#   Modelos candidatos: con 1000-1500 submuestras, el modelo parece tener una menor
#   tasa de error, especialmente en el caso de nodesize 10
#   No obstante, modelos como 10 + 500 o 20 + 500 no presentan una tasa de fallos tan elevada
union_bagging_modelo1$modelo <- with(union_bagging_modelo1,
                   reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa, col = config)) +
        geom_boxplot(alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo1_5rep.png")

#-- Distribucion del AUC
#   Mejor resultado: nuevamente, modelos como 10 + 1500, 20 + 1500 o 10 + 1000 presenta un valor
#   AUC elevado, aunque modelos como 10 + 500 tambien presentan un buen valor AUC
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
sampsizes.1_aux <- list(1, 500, 1000, 1500)
bagging_modelo1_2_10folds <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1_aux, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 10, repe = 20)
rm(sampsizes.1_aux)

# Nodesize 20
nodesizes.1 <- list(20)
sampsizes.2_aux <- list(1, 500, 1000, 1500)
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
  bagging_modelo1_2_10folds[bagging_modelo1_2_10folds$modelo == "10+1500", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+1", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+500", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+1000", ],
  bagging_modelo1_3_10folds[bagging_modelo1_3_10folds$modelo == "20+1500", ]
)

union_bagging_modelo1_10folds$config <- rep("10_folds-20_rep", 80)
union_bagging_modelo1_final          <- rbind(union_bagging_modelo1,
                                              union_bagging_modelo1_10folds)

#-- Realizamos la comparacion final del modelo 1 con k = 5 y k = 10 grupos
#-- Distribucion de la tasa de fallos
union_bagging_modelo1_final$modelo <- with(union_bagging_modelo1_final,
                                     reorder(modelo,tasa, mean))
p <- ggplot(union_bagging_modelo1_final[!grepl("10+", union_bagging_modelo1_final$modelo,fixed = TRUE), ], aes(x = modelo, y = tasa, col = config)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo1_5_10_folds.png")

#-- Distribucion del AUC
union_bagging_modelo1_final$modelo <- with(union_bagging_modelo1_final,
                                     reorder(modelo,auc, mean))
t <- ggplot(union_bagging_modelo1_final[!grepl("10+", union_bagging_modelo1_final$modelo,fixed = TRUE), ], aes(x = modelo, y = auc, col = config)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("AUC por modelo") + theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo1_5_10_folds.png")

ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave("./charts/bagging/bis_03_comparacion_final_modelo1_5_10_folds.png")

#-- Teniendo en cuenta resultados empiricos
#   -> Observamos que el tamaño optimo de muestra se situa entre 500 y 1500 (sampsize)
#   -> Diria que 20 + 1000

#-- Curiosidad ¿Y si reducimos el sampsize por debajo de 500?
sampsizes.2_aux <- list(200, 300, 400, 500)
bagging_modelo1_3_10folds_v2 <- tuneo_bagging(surgical_dataset, target = target,
                                           lista.continua = var_modelo1,
                                           nodesizes = 20,
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

bagging_modelo2_temp <- bagging_modelo2 %>% group_by(modelo, sampsizes, nodesizes) %>% summarise(tasa_fallos_media = mean(tasa))
#-- Distribucion de la tasa de error
p <- ggplot(bagging_modelo2_temp, aes(x=factor(sampsizes), y=tasa_fallos_media,
                                      colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2.5, shape = 20) +
  scale_colour_manual(values=rainbow(7))+
  ggtitle("Tasa de fallos (Modelo 2)") +
  theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/distribuciones/03_distribucion_tasa_error_modelo2.png")

#-- Distribucion del AUC
bagging_modelo2_temp <- bagging_modelo2 %>% group_by(modelo, sampsizes, nodesizes) %>% summarise(auc_medio = mean(auc))
#-- Distribucion de la tasa de error
t <- ggplot(bagging_modelo2_temp, aes(x=factor(sampsizes), y=auc_medio,
                                      colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=2.5, shape = 20) +
  scale_colour_manual(values=rainbow(7))+
  ggtitle("AUC (Modelo 2)") +
  theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/distribuciones/03_distribucion_auc_modelo2.png")

ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave('./charts/03_distribucion_auc_tasa_fallos_modelo2.png')

# Del mismo modo que sucedia con el modelo 1, con sampsize 500-2000 se obtienen unos modelos cuyos valores
# de error y AUC son muy similares a los obtenidos con el conjunto total de observaciones. Ademas, podemos
# estudiar los modelos a partir de valores de nodesize como 10, 20 o 30, principalmente. Si necesitamos reducir
# el valor de nodesize o aumentarlo, lo comprobaremos.

# Nodesize 10: nuevamente, en torno a 500-1000-1500-2000 se obtienen resultados muy similares
# a un modelo con el conjunto total de muestras con reemplazamiento
# Mejor AUC: 0.9075, tasa de fallos: entre 0.1075 y 0.095
nodesizes.2 <- list(10)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10, 
                                   show_nrnodes = "yes")
# 10 + 1: 625; 10 + 500: 115; 10 + 1000: 189; 10 + 1500: 253; 10 + 2000: 311; 10 + 2500: 359

# Nodesize 20: nuevamente, nos encontramos con modelos con sampsize 500-1000-1500
# con valores AUC y tasas de fallos muy similares
# Sin embargo, el modelo no parece mejorar conforme aumenta el valor de nodesize
nodesizes.2 <- list(20)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")
# 20 + 1: 453; 20 + 500: 85; 20 + 1000: 145; 20 + 1500: 195; 20 + 2000: 237; 20 + 2500: 269

# Nodesize 30: no merece demasiado la pena aumentar el nodesize a 30
nodesizes.2 <- list(30)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")
# 30 + 1: 373; 30 + 500: 67; 30 + 1000: 121; 30 + 1500: 151; 30 + 2000: 191; 30 + 2500: 221


# ¿Mereceria la pena aumentar nodesize a 50? ¿O disminuir nodesize a 5?
nodesizes.2 <- list(50)
sampsizes.2 <- list(1, 500, 1000, 1500, 2000, 2500)
bagging_modelo2_5 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees.2, grupos = 5, repe = 10,
                                   show_nrnodes = "yes")
rm(bagging_modelo2_5)
# 50 + 1: 277; 50 + 500: 53; 50 + 1000: 87; 50 + 1500: 123; 50 + 2000: 143; 50 + 2500: 167
# 5 + 1: 783;  5 + 500: 147; 5 + 1000: 239; 5 + 1500: 329;  5 + 2000: 403;  5 + 2500: 453
# 1 + 1: 1111; 1 + 500: 195; 1 + 1000: 315; 1 + 1500: 429;  1 + 2000: 537;  1 + 2500: 641
bagging_comp2 <- rbind(bagging_modelo2_3, bagging_modelo2_5); bagging_comp2$nodesize <- c(rep("20", 60), rep("5", 60)) 
bagging_comp2$modelo <- with(bagging_comp2, reorder(modelo,tasa, mean))
p <- ggplot(bagging_comp2, aes(x = factor(modelo), y = tasa, colour = nodesize, group = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
bagging_comp2$modelo <- with(bagging_comp2, reorder(modelo,auc, mean))
t <- ggplot(bagging_comp2, aes(x = factor(modelo), y = auc, colour = nodesize, group = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave('./charts/03_distribucion_auc_tasa_fallos_modelo2_comp_50rep.png')


# Aumentando o disminuyendo el nodesize a 5 0 50, respectivamente, el modelo no mejora practicamente
# ni en tasa de fallos ni en AUC

#-- Posible candidato:   nodesize 10 y sampsize 500-1000-1500
#-- Posible candidato:   nodesize 20 y sampsize 500-1000-1500
#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo2 <- rbind(
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+1", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+500", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "20+1000", ]
)

union_bagging_modelo2$config <- rep("5_folds-10_rep", 15)

#-- Distribucion de la tasa de error
#   Modelos candidatos: 
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,tasa, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = tasa, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo2_5rep.png")

#-- Distribucion del AUC
#   Mejor resultado: 
union_bagging_modelo2$modelo <- with(union_bagging_modelo2,
                                     reorder(modelo,auc, mean))
ggplot(union_bagging_modelo2, aes(x = modelo, y = auc, col = config)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo2_5rep.png")

#-- Para salir de dudas ¿Aumentamos el numero de grupos a 10?
# Nodesize 10
nodesizes.1 <- list(10)
sampsizes.2_aux <- list(1, 500, 1000, 1500)
bagging_modelo2_2_10folds <- tuneo_bagging(surgical_dataset, target = target,
                                           lista.continua = var_modelo2,
                                           nodesizes = nodesizes.1,
                                           sampsizes = sampsizes.2_aux, mtry = mtry.2,
                                           ntree = n.trees.2, grupos = 10, repe = 20)
rm(sampsizes.2_aux)
rm(nodesizes.1)

# Nodesize 20
nodesizes.1 <- list(20)
sampsizes.2_aux <- list(1, 500, 1000, 1500)
bagging_modelo2_3_10folds <- tuneo_bagging(surgical_dataset, target = target,
                                           lista.continua = var_modelo2,
                                           nodesizes = nodesizes.1,
                                           sampsizes = sampsizes.2_aux, mtry = mtry.2,
                                           ntree = n.trees.2, grupos = 10, repe = 20)
rm(sampsizes.2_aux)
rm(nodesizes.1)

#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo2_10folds <- rbind(
  bagging_modelo2_3_10folds[bagging_modelo2_3_10folds$modelo == "20+1", ],
  bagging_modelo2_3_10folds[bagging_modelo2_3_10folds$modelo == "20+500", ],
  bagging_modelo2_3_10folds[bagging_modelo2_3_10folds$modelo == "20+1000", ]
)

union_bagging_modelo2_10folds$config <- rep("10_folds-20_rep", 30)
union_bagging_modelo2_final          <- rbind(union_bagging_modelo2,
                                              union_bagging_modelo2_10folds)

#-- Distribucion de la tasa de error
#   Modelos candidatos: 20+1000
union_bagging_modelo2_final$modelo <- with(union_bagging_modelo2_final,
                                     reorder(modelo,tasa, mean))
p <- ggplot(union_bagging_modelo2_final, aes(x = modelo, y = tasa, col = config)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/bis_03_comparacion_final_tasa_modelo2_10rep.png")

#-- Distribucion del AUC
#   
union_bagging_modelo2_final$modelo <- with(union_bagging_modelo2_final,
                                     reorder(modelo,auc, mean))
t <- ggplot(union_bagging_modelo2_final, aes(x = modelo, y = auc, col = config)) +
            geom_boxplot(alpha = 0.7) +
            scale_x_discrete(name = "Modelo") +
            ggtitle("AUC por modelo") + theme(text = element_text(size=13, face = "bold"))
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo2_10rep.png")

ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave("./charts/bagging/bis_03_comparacion_final_modelo2_5_10_folds.png")

# ¿Se mantienen dichas distribuciones con 10 repeticiones?
# Por lo general, si. Por ello, elegimos como modelos candidatos 20 + 1000

# Previamente, observamos que con 3000 arboles el modelo se estabiliza
bagging_modelo2_3_10folds_v2_3000 <- tuneo_bagging(surgical_dataset, target = target,
                                                    lista.continua = var_modelo2,
                                                    nodesizes = 20,
                                                    sampsizes = 1000, mtry = mtry.2,
                                                    ntree = 3000, grupos = 10, repe = 20)

#-- Distribucion del AUC
# mtry  Accuracy   Kappa    AccuracySD  KappaSD
#    4 0.8982833 0.7023332 0.01059103 0.03384279
# 20 + 1000 -> FINISHED
bagging_modelo2_3_10folds_v2_3000$modelo <- with(bagging_modelo2_3_10folds_v2_3000,
                                           reorder(modelo,auc, mean))
ggplot(bagging_modelo2_3_10folds_v2_3000, aes(x = modelo, y = auc)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/bis_03_comparacion_final_auc_modelo2_10_3000rep.png")

#-- Curiosidad ¿Y si reducimos el sampsize por debajo de 500?
sampsizes.2_aux <- list(200, 300, 400, 500)
bagging_modelo2_3_10folds_v2 <- tuneo_bagging(surgical_dataset, target = target,
                                              lista.continua = var_modelo2,
                                              nodesizes = 20,
                                              sampsizes = sampsizes.2_aux, mtry = mtry.2,
                                              ntree = n.trees.2, grupos = 10, repe = 20)
rm(sampsizes.2_aux)

#-- Modelos candidatos
#   Bagging (modelo 1 con 5 variables): nodesize 10 + sampsize 500 - 1000
#   Bagging (modelo 2 con 4 variables): nodesize 10 + sampsize 500 - 1000

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
                  nodesize = 20, sampsize = 500, ntree = n.trees.1, replace = TRUE)

matriz_conf_1 <- matriz_confusion_predicciones(bagging_1, NULL, surgical_test_data, 0.5)

set.seed(1234)
bagging_1_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.1,
                     nodesize = 20, sampsize = 1000, ntree = n.trees.1, replace = TRUE)

matriz_conf_1_1 <- matriz_confusion_predicciones(bagging_1_1, NULL, surgical_test_data, 0.5)

#-- Modelo 2
rfgrid.2 <-expand.grid(mtry=mtry.2)
set.seed(1234)
bagging_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                   data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.2,
                   nodesize = 20, sampsize = 500, ntree = n.trees.2, replace = TRUE)

matriz_conf_2 <- matriz_confusion_predicciones(bagging_2, NULL, surgical_test_data, 0.5)

set.seed(1234)
bagging_2_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                     data=surgical_dataset, method="rf", trControl = control,tuneGrid = rfgrid.2,
                     nodesize = 20, sampsize = 1000, ntree = n.trees.2, replace = TRUE)

matriz_conf_2_2 <- matriz_confusion_predicciones(bagging_2_2, NULL, surgical_test_data, 0.5)

rm(rfgrid.1)
rm(rfgrid.2)

#--- Predicciones
#     Modelo 1 (20 + 500)
#     Reference
#     Prediction   No  Yes
#     No         6442  172
#     Yes        740  1447

#     Modelo 1 (20 + 1000)
#     Reference
#     Prediction   No  Yes
#     No         6471  123
#     Yes         746  1441

#     Modelo 2 (20 + 500)
#     Reference
#     Prediction   No  Yes
#     No         6452  142
#     Yes         750  1437

#     Modelo 2 (20 + 1000)
#     Reference
#     Prediction   No  Yes
#     No         6515  79
#     Yes         782  1405

#-- Los resultados son muy similares en ambas situaciones
#   Elegiria un modelo con 20 nodesize y sampsize 1000

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "bagging"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)

#-- ¿Y si lo probamos sin reemplazamiento? Probamos con el mejor modelo en terminos de AUC (modelo 1)
bagging_modelo_sin_reemp <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = 20,
                                   sampsizes = 1000, mtry = mtry.1,
                                   ntree = n.trees.1, grupos = 5, repe = 10, replace = FALSE)
bagging_modelo_sin_reemp$modelo <- "BAG. MODELO 1 (no reemp)"
modelos_actuales <- rbind(modelos_actuales, bagging_modelo_sin_reemp)
modelos_actuales$tipo <- c(rep("LOGISTICA", 20), rep("RED NEURONAL", 20), rep("BAGGING", 20))
                          
modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
ggplot(modelos_actuales, aes(x = modelo, y = tasa, col = tipo)) +
       geom_boxplot(alpha = 0.7) +
       scale_x_discrete(name = "Modelo") +
       ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))
ggsave('./charts/comparativas/03_log_avnnet_bagging_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
ggplot(modelos_actuales, aes(x = modelo, y = auc, col = tipo)) +
       geom_boxplot(alpha = 0.7) +
       scale_x_discrete(name = "Modelo") +
       ggtitle("AUC por modelo") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))
ggsave('./charts/comparativas/03_log_avnnet_bagging_auc.jpeg')

#-- Si hacemos zoom sobre los modelos bagging...
modelos_actuales_zoomed <- modelos_actuales[modelos_actuales$modelo %in% c("BAG. MODELO 1", "BAG. MODELO 2", "BAG. MODELO 1 (no reemp)"), ]
modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,tasa, mean))
p <- ggplot(modelos_actuales_zoomed, aes(x = modelo, y = tasa)) +
            geom_boxplot() +
            scale_x_discrete(name = "Modelo") +
            ggtitle("Tasa de fallos (solo BAGGING)") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
ggsave('./charts/bagging/03_FINAL_tasa.jpeg')

modelos_actuales_zoomed$modelo <- with(modelos_actuales_zoomed,
                                       reorder(modelo,auc, mean))
t <- ggplot(modelos_actuales_zoomed, aes(x = modelo, y = auc)) +
            geom_boxplot() +
            scale_x_discrete(name = "Modelo") +
            ggtitle("AUC (solo BAGGING)") + theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45))
ggsave('./charts/bagging/03_FINAL_auc.jpeg')

ggpubr::ggarrange(p, t, common.legend = TRUE)
ggsave('./charts/bagging/03_comparacion_final_bagging.png')

#---- Estadisticas
# Por tasa fallos --------------- auc
#  bagging modelo 1          bagging modelo 1
#  bagging modelo 2          bagging modelo 2
#   avnnet modelo 2          avnnet  modelo 1
#   avnnet modelo 1           avnnet modelo 2
#   log.   modelo 1           log.   modelo 1
#   log.   modelo 2           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/Bagging.RData")


