# ------------- Gradient Boosting ---------------
# Objetivo: elaborar el mejor modelo de Gradient Boosting de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           shrinkage, n.minobsnode, n.trees e interaction.depth
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(randomForest)  # Seleccion del numero de arboles
  library(readxl)        # Lectura de ficheros Excel
  library(ggrepel)       # Evita solapamientos en las etiquetas de un grafico
  library(RColorBrewer)  # Paleta de colores
  
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

#-- Prueba inicial Modelo 1
set.seed(1234)
gbmgrid<-expand.grid(shrinkage=c(0.4,0.3,0.2,0.1,0.05,0.03,0.01,0.001),
                     n.minobsinnode=c(5,10,20),
                     n.trees=c(100,500,1000,5000),
                     interaction.depth=c(2))

control<-trainControl(method="repeatedcv",number=5,savePredictions = "all", 
                      repeats=5,classProbs=TRUE)
gbm_modelo1 <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                     method="gbm",trControl=control,tuneGrid=gbmgrid,
                     distribution="bernoulli", bag.fraction=1,verbose=FALSE)
gbm_modelo1
plot(gbm_modelo1, main = "Gradient Boosting Hyperparameters Tunning (Modelo 1)")
# Recomendacion caret: n.trees = 500, shrinkage = 0.2 y n.minobsinnode = 20
# Accuracy: 0.9055354
#-- Observaciones: por lo general, podemos comprobar que existe un patron que se repite practicamente en cualquier
#   valor de minobsinnode: bien un alto numero de iteraciones (alrededor de 1000-5000) y un shrinkage bajo (entre 0.001 y 0.05)
#   o bien un numero moderado-bajo de iteraciones (100-500-1000) y un shrinkage alto (entre 0.10 y 0.40)
#   Curiosamente, el valor de accuracy se mantiene en torno a 0.90 conforme aumenta el parámetro shrinkage con 100 iteraciones
#   ¿Y si aumentamos aun mas el valor shrinkage?
set.seed(1234)
gbmgrid_aux<-expand.grid(shrinkage=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                     n.minobsinnode=c(10,20),
                     n.trees=c(100),
                     interaction.depth=c(2))

gbm_modelo1_aux <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                         method="gbm",trControl=control,tuneGrid=gbmgrid_aux,
                         distribution="bernoulli", bag.fraction=1,verbose=FALSE)
gbm_modelo1_aux
plot(gbm_modelo1_aux, main = "Gradient Boosting Hyperparameters Tunning (Modelo 1) aumentando shrinkage")

# En cualquiera de los casos, no parece benecifiarse de un aumento considerable, aunque en 0.3 parece alcanzar su valor maximo (0.9037235)
# La diferencia con respecto al modelo recomendado por caret es muy poco significativa: 0.9055354

#-- Prueba inicial Modelo 2
gbm_modelo2 <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                     method="gbm",trControl=control,tuneGrid=gbmgrid,
                     distribution="bernoulli", bag.fraction=1,verbose=FALSE)
gbm_modelo2
plot(gbm_modelo2, main = "Gradient Boosting Hyperparameters Tunning (Modelo 2)")
# Recomendacion caret: n.trees = 1000, shrinkage = 0.1 y n.minobsinnode = 20
# Accuracy: 0.9041672

#-- Observaciones: del mismo modo, con el segundo modelo nos encontramos con dos grandes contrastes: o bien emplear
#   un elevado numero de iteraciones (alrededor de 1000-5000) y un shrinkage bajo (entre 0.001 y 0.05); o bien emplear
#   un numero moderado-bajo de iteraciones (100-500-1000) y un shrinkage alto (entre 0.10 y 0.40)
#   Del mismo modo que sucede en el modelo anterior, el modelo parece "beneficiarse" de un aumento en el numero del valor
#   shrinkage, por lo que podriamos probar a aumentar el valor de dicho parametro para estudiar su evolucion
gbm_modelo2_aux <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                         method="gbm",trControl=control,tuneGrid=gbmgrid_aux,
                         distribution="bernoulli", bag.fraction=1,verbose=FALSE)
gbm_modelo2_aux
plot(gbm_modelo2_aux, main = "Gradient Boosting Hyperparameters Tunning (Modelo 2) aumentando shrinkage")
# Por lo general, no parece beneficiarse de un aumento en el parametro shrinkage, incluso todo lo contrario, dado que
# disminuye conforme aumenta su valor

# Diria entre 0.2, 0.3, 0.4 y 100 arboles
#-- Estudio del Early Stopping
#   Modelo 1
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                      interaction.depth=c(2))
set.seed(1234)
gbm_modelo1_early_stopping <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                    method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                    distribution="bernoulli", bag.fraction=1,verbose=FALSE)
set.seed(1234)
gbm_modelo1_early_stopping_2 <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                      method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                                      distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo1_es_final <- rbind(gbm_modelo1_early_stopping$results, gbm_modelo1_early_stopping_2$results)
gbm_modelo1_es_final$shrinkage <- c(rep("0.2", 10), rep("0.3", 10))

gbm_modelo1_es_final %>% ggplot(aes(x = n.trees, y = Accuracy, label = n.trees,
                                    group = shrinkage, col = shrinkage)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelo 1 (Early Stopping)")
#-- Llama la atencion 100 + 0.2 ; 100 + 0.3

#  Modelo 2
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                      interaction.depth=c(2))
set.seed(1234)
gbm_modelo2_early_stopping <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                                    method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                    distribution="bernoulli", bag.fraction=1,verbose=FALSE)
set.seed(1234)
gbm_modelo2_early_stopping_2 <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                                      method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                                      distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo2_es_final <- rbind(gbm_modelo2_early_stopping$results, gbm_modelo2_early_stopping_2$results)
gbm_modelo2_es_final$shrinkage <- c(rep("0.2", 10), rep("0.3", 10))

gbm_modelo2_es_final %>% ggplot(aes(x = n.trees, y = Accuracy, label = n.trees,
                                    group = shrinkage, col = shrinkage)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelo 2 (Early Stopping)")

# Nuevamente, nos encontramos que con 100 iteraciones parece una buena alternativa

#--------------------- bag. fraction ----------------------

#-- Tuneo bag.fraction (fraccion de observaciones del conjunto de entrenamiento seleccionadas aleatoriamente a proponer en la construccion
#   del siguiente arbol) modelo 1
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(100),
                                    interaction.depth=c(2))

gbm_modelo1_early_stopping_bag_fraction <- data.frame()
for(bag_fraction in c(.05, .1, .2, .3, .5, .8, 1)) {
  set.seed(1234)
  temp <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                  method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                  distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  gbm_modelo1_early_stopping_bag_fraction <- rbind(gbm_modelo1_early_stopping_bag_fraction, temp$results)
}
rm(temp)
gbm_modelo1_early_stopping_bag_fraction$bag.fraction <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1)


#-- Llama la atencion 100 + 0.3 ; 200 + 0.3 ; 100 + 0.2 ; 500 + 0.2
#-- Tuneo bag.fraction modelo 2
gbm_modelo2_early_stopping_bag_fraction <- data.frame()
for(bag_fraction in c(.05, .1, .2, .3, .5, .8, 1)) {
  set.seed(1234)
  temp <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  gbm_modelo2_early_stopping_bag_fraction <- rbind(gbm_modelo2_early_stopping_bag_fraction, temp$results)
}
gbm_modelo2_early_stopping_bag_fraction$bag.fraction <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1)
rm(temp); rm(bag_fraction)

#-- Analicemos el modelo 1
gbm_modelo1_early_stopping_bag_fraction$bag.fraction <- as.factor(gbm_modelo1_early_stopping_bag_fraction$bag.fraction)

graph <- ggplot(gbm_modelo1_early_stopping_bag_fraction, aes(x = n.trees, y = Accuracy, colour = bag.fraction)) +
  geom_point() +
  facet_grid( . ~ shrinkage )+
  facet_wrap(. ~ shrinkage, ncol = 2, scales = "free_x")
labs(colour = 'bag.fraction') +
  theme_grey() +
  theme(
    legend.position = 'right'
  )
graph
# Como podemos observar, a diferencia de los modelos Bagging y Random Forest, no basta con "sortear" menos de un 10 % de la observaciones de entrenamiento
# sino que sorteando un 80-50 % de las observaciones es suficiente
# ¿Y el modelo 2?
gbm_modelo2_early_stopping_bag_fraction$bag.fraction <- as.factor(gbm_modelo2_early_stopping_bag_fraction$bag.fraction)

graph_2 <- ggplot(gbm_modelo2_early_stopping_bag_fraction, aes(x = n.trees, y = Accuracy, color = bag.fraction)) +
  geom_point() +
  facet_grid( . ~ shrinkage )+
  facet_wrap(. ~ shrinkage, ncol = 2, scales = "free_x")
labs(colour = 'bag.fraction') +
  theme_grey() +
  theme(
    legend.position = 'right'
  )
graph_2
#-- Desde un punto de vista general, en ambos modelos llama la atencion que con un 80-50 % de la muestra
#   sorteada se obtienen resultados muy similares que con el dataset completo
# Como consecuencia, con 100 arboles, minobsinnode 20, shrinkage 0.3 y bag.fraction 0.5-1

#-- Comparacion modelos candidatos (5 y 10 rep)
#   Modelo 1
tuneo_gradient_boosting_modelo1 <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 1), shrinkage = c(0.2),
  interaction.depth = 2, grupos = 5, repe = 5, path.1 = "",
  path.2 = ""
)

tuneo_gradient_boosting_modelo1_10rep <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 1), shrinkage = c(0.2),
  interaction.depth = 2, grupos = 5, repe = 10, path.1 = "",
  path.2 = ""
)

tuneo_gradient_boosting_modelo2 <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo2, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 1), shrinkage = c(0.2),
  interaction.depth = 2, grupos = 5, repe = 5, path.1 = "",
  path.2 = ""
)

tuneo_gradient_boosting_modelo2_10_rep <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo2, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 1), shrinkage = c(0.2),
  interaction.depth = 2, grupos = 5, repe = 10, path.1 = "",
  path.2 = ""
)

tuneo_modelo1 <- rbind(tuneo_gradient_boosting_modelo1, tuneo_gradient_boosting_modelo1_10rep)
tuneo_modelo1$rep <- c(rep("5", 10), rep("10", 20))

tuneo_modelo2 <- rbind(tuneo_gradient_boosting_modelo2, tuneo_gradient_boosting_modelo2_10_rep)
tuneo_modelo2$rep <- c(rep("5", 10), rep("10", 20))

tuneo_modelo1$modelo <- with(tuneo_modelo1, reorder(modelo,tasa, mean))
ggplot(tuneo_modelo1, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/gradient_boosting/modelo1/05_tasa_fallos_modelo1_10rep.png")

tuneo_modelo1$modelo <- with(tuneo_modelo1, reorder(modelo,auc, mean))
ggplot(tuneo_modelo1, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/gradient_boosting/modelo1/05_auc_modelo1_10rep.png")

tuneo_modelo2$modelo <- with(tuneo_modelo2, reorder(modelo,tasa, mean))
ggplot(tuneo_modelo2, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/gradient_boosting/modelo2/05_tasa_fallos_modelo2_10rep.png")

tuneo_modelo2$modelo <- with(tuneo_modelo2, reorder(modelo,auc, mean))
ggplot(tuneo_modelo2, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")
ggsave("./charts/gradient_boosting/modelo2/05_auc_modelo2_10rep.png")

# Podemos comprobar como la diferencia entre bag.fraction 0.5 y 1 es practicamente nula, por lo que nos decantamos por bag.fraction = 0.5

#-- Conclusion: nos decantamos por n.minobsinnode = 20, shrinkage = 0.3, interaction.depth = 2, bag.fraction = 0.5
#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE) 
set.seed(1234)
gbm_modelo1_final <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                     data=surgical_dataset, method="gbm", trControl = control,tuneGrid = expand.grid(shrinkage=c(0.2), 
                                                                                                    n.minobsinnode=c(20),
                                                                                                    n.trees=c(100),
                                                                                                    interaction.depth=c(2)),
                     distribution="bernoulli", bag.fraction=0.5,verbose=FALSE)

set.seed(1234)
gbm_modelo2_final <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                           data=surgical_dataset, method="gbm", trControl = control,tuneGrid = expand.grid(shrinkage=c(0.2), 
                                                                                                           n.minobsinnode=c(20),
                                                                                                           n.trees=c(100),
                                                                                                           interaction.depth=c(2)),
                           distribution="bernoulli", bag.fraction=0.5,verbose=FALSE)

matriz_conf_1 <- matriz_confusion_predicciones(gbm_modelo1_final, NULL, surgical_test_data, 0.5)

matriz_conf_2 <- matriz_confusion_predicciones(gbm_modelo2_final, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No   Yes
# No          6457  137
# Yes         715   1472

# Modelo 2
# Reference
# Prediction  No  Yes
# No         6477 117
# Yes         715 1472

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "gradient_boosting"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)
modelos_actuales$tipo <- c(rep("LOGISTICA", 20), rep("RED NEURONAL", 20), rep("BAGGING", 20), rep("RANDOM FOREST", 20),
                           rep("GBM", 20))

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
ggplot(modelos_actuales, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/comparativas/05_log_avnnet_bagging_rf_gbm_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
ggplot(modelos_actuales, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/comparativas/05_log_avnnet_bagging_rf_gbm_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#      gbm modelo 1               rf. modelo 1
#      gbm modelo 2           bagging modelo 1
#  bagging modelo 2           bagging modelo 2
#  bagging modelo 1               gbm modelo 1
#   avnnet modelo 2               rf. modelo 2
#      rf. modelo 2            avnnet modelo 1
#      rf. modelo 1               gbm modelo 2
#   avnnet modelo 1            avnnet modelo 2
#   log.   modelo 1            log.   modelo 1
#   log.   modelo 2            log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/GradientBoosting.RData")


