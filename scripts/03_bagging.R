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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0", 
                 "Age", "moonphase.0", "baseline_osteoart")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 8 para el modelo 1 y 5 para el modelo 2
mtry.1 <- 8
mtry.2 <- 5

#--- Seleccion del numero de arboles
#--  Modelo 1
set.seed(1234)
rfbis.1<-randomForest(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+dow.0+Age+moonphase.0+baseline_osteoart,
                      data=surgical_dataset,
                      mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

#-- Modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~Age+mortality_rsi+ccsMort30Rate+bmi+ahrq_ccs,
                      data=surgical_dataset,
                      mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

rfbis.1.df <- as.data.frame(rfbis.1$err.rate)
rfbis.1.df$n_trees <- seq(1, nrow(rfbis.1.df))
rfbis.2.df <- as.data.frame(rfbis.2$err.rate)
rfbis.2.df$n_trees <- seq(1, nrow(rfbis.2.df))

plot(rfbis.2$err.rate[,1], col = 'red', type = 'l')
lines(rfbis.1$err.rate[,1], col = 'blue')
legend("topright", legend = c("5 variables","8 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

#-- Ampliamos entre 0 y 2000 arboles
plot(rfbis.2$err.rate[c(0:2000),1], type = 'l', col = 'red')
lines(rfbis.1$err.rate[c(0:2000),1], col = 'blue')
legend("topright", legend = c("5 variables","8 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

#-- Ampliamos entre 0 y 1000 arboles
plot(rfbis.2$err.rate[c(0:1000),1], type = 'l', col = 'red')
lines(rfbis.1$err.rate[c(0:1000),1], col = 'blue')
legend("topright", legend = c("5 variables","8 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

#--- Tuneo de modelos
n.trees <- 300
# Sampsize maximo: (k-1) * n => (4/5) * 5854 = 4683.2 ~ 4600 obs.
# Conclusion: sampsize maximo: 4600 obs. (de forma aproximada)

#--  Modelo 1
sampsizes.1 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.1 <- list(5, 10, 20, 30, 40, 50, 100, 150)

bagging_modelo1 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees, grupos = 5, repe = 5)
bagging_modelo1$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[1,]))
bagging_modelo1$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo1$modelo, '+', fixed = T))[2,]))

#-- Distribucion de la tasa de error
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=tasa,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=3, shape = 18)

#-- Distribucion del AUC
#  Parece buen candidato 20 nodeszie y sampsize todos salvo 100
ggplot(bagging_modelo1, aes(x=factor(sampsizes), y=auc,
                 colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=3, shape = 18)

nodesizes.1 <- list(20)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4600)
bagging_modelo1_2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo1,
                                 nodesizes = nodesizes.1,
                                 sampsizes = sampsizes.1, mtry = mtry.1,
                                 ntree = n.trees, grupos = 5, repe = 5)

# A partir de 2500 submuestras el error se estabiliza
nodesizes.1 <- list(50)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4600)
bagging_modelo1_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees, grupos = 5, repe = 5)

nodesizes.1 <- list(30)
sampsizes.1 <- list(1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4600)
bagging_modelo1_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo1,
                                   nodesizes = nodesizes.1,
                                   sampsizes = sampsizes.1, mtry = mtry.1,
                                   ntree = n.trees, grupos = 5, repe = 5)

#-- Posible candidato:   nodesize 20 y sampsize 1000 (ligera tasa de error superior a con reemplazamiento)
#-- Posible candidato:   nodesize 50 y sampsize 2500
#-- Posibles candidatos: nodesize 30 y sampsize 2000
#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo1 <- rbind(
  bagging_modelo1_2[bagging_modelo1_2$modelo == "20+1000", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "50+1", ],
  bagging_modelo1_3[bagging_modelo1_3$modelo == "50+2500", ],
  bagging_modelo1_4[bagging_modelo1_4$modelo == "30+2000", ]
)
#-- Distribucion de la tasa de error
ggplot(union_bagging_modelo1, aes(x = modelo, y = tasa)) +
        geom_boxplot(fill = "#4271AE", colour = "#1F3552",
                     alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/03_comparacion_final_tasa_modelo1_5rep.png")

#-- Distribucion del AUC
ggplot(union_bagging_modelo1, aes(x = modelo, y = auc)) +
        geom_boxplot(fill = "#4271AE", colour = "#1F3552",
                     alpha = 0.7) +
        scale_x_discrete(name = "Modelo") +
        ggtitle("AUC por modelo")
ggsave("./charts/bagging/03_comparacion_final_auc_modelo1_10rep.png")

# MODELO 2
sampsizes.2 <- list(1, 100, 500, 1000, 2000, 3000, 4600)
nodesizes.2 <- list(5, 10, 20, 30, 40, 50, 100, 150)

bagging_modelo2 <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = var_modelo2,
                                 nodesizes = nodesizes.2,
                                 sampsizes = sampsizes.2, mtry = mtry.2,
                                 ntree = n.trees, grupos = 5, repe = 5)

bagging_modelo2$nodesizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2$modelo, '+', fixed = T))[1,]))
bagging_modelo2$sampsizes <- as.numeric(c(data.frame(strsplit(bagging_modelo2$modelo, '+', fixed = T))[2,]))

#-- Distribucion de la tasa de error
ggplot(bagging_modelo2, aes(x=factor(sampsizes), y=tasa,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=3, shape = 18)

#-- Distribucion del AUC
# #  Parece buen candidato 20, 30 o 50 nodesize y sampsize todos salvo 100
ggplot(bagging_modelo2, aes(x=factor(sampsizes), y=auc,
                            colour=factor(nodesizes))) +
  geom_point(position=position_dodge(width=0.3),size=3, shape = 18)

#- Mejor opcion: 20 + 1000
nodesizes.2 <- list(20)
sampsizes.2 <- list(1000)
bagging_modelo2_2 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees, grupos = 5, repe = 10)

#- Mejor opcion: 30 + 2000
nodesizes.2 <- list(30)
sampsizes.2 <- list(2000)
bagging_modelo2_3 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees, grupos = 5, repe = 10)

#- Mejora opcion: 50 + 2000-2500-3000
nodesizes.2 <- list(50)
sampsizes.2 <- list(1, 2000, 2500, 3000)
bagging_modelo2_4 <- tuneo_bagging(surgical_dataset, target = target,
                                   lista.continua = var_modelo2,
                                   nodesizes = nodesizes.2,
                                   sampsizes = sampsizes.2, mtry = mtry.2,
                                   ntree = n.trees, grupos = 5, repe = 10)

#-- ¿Por qué modelo bagging nos decantamos?
union_bagging_modelo2 <- rbind(
  bagging_modelo2_2[bagging_modelo2_2$modelo == "20+1000", ],
  bagging_modelo2_3[bagging_modelo2_3$modelo == "30+2000", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "50+1", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "50+2000", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "50+2500", ],
  bagging_modelo2_4[bagging_modelo2_4$modelo == "50+3000", ]
)

#-- Distribucion de la tasa de error
ggplot(union_bagging_modelo2, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/bagging/03_comparacion_final_tasa_modelo2_10rep.png")

#-- Distribucion del AUC
ggplot(union_bagging_modelo2, aes(x = modelo, y = auc)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/bagging/03_comparacion_final_auc_modelo2_10rep.png")

#-- Modelo 2: nos decantamos por nodesize 30 + sampsize 2000
