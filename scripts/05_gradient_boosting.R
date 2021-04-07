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

control<-trainControl(method = "cv",number=5,savePredictions = "all",
                      classProbs=TRUE)
gbm_modelo1 <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                     method="gbm",trControl=control,tuneGrid=gbmgrid,
                     distribution="bernoulli", bag.fraction=1,verbose=FALSE)
gbm_modelo1
plot(gbm_modelo1, main = "Gradient Boosting Hyperparameters Tunning (Modelo 1)")
# Recomendacion caret: n.trees = 500, shrinkage = 0.2 y n.minobsinnode = 20
# Accuracy: 0.9046799
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

# En cualquiera de los casos, no parece benecifiarse de un aumento considerable


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
set.seed(1234)
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(50, 100,400, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                      interaction.depth=c(2))

gbm_modelo1_early_stopping <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                    method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                    distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo1_early_stopping_2 <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                      method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                                      distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo1_es_final <- rbind(gbm_modelo1_early_stopping$results, gbm_modelo1_early_stopping_2$results)
gbm_modelo1_es_final$shrinkage <- c(rep("0.2", 10), rep("0.3", 10))

gbm_modelo1_es_final %>% ggplot(aes(x = n.trees, y = Accuracy, label = n.trees,
                                    group = shrinkage, col = shrinkage)) + 
  geom_point() +
  geom_line() +
  geom_text_repel(data=gbm_modelo1_es_final %>% sample_frac(0.6)) +
  ggtitle("Evolucion Accuracy Modelo 1 (Early Stopping)")
#-- Llama la atencion 100 + 0.3 ; 100 + 0.2 ; 500 + 0.2

#-- Tuneo bag.fraction (fraccion de observaciones del conjunto de entrenamiento seleccionadas aleatoriamente a proponer en la construccion
#   del siguiente arbol)
set.seed(1234)
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(100),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(100, 500),
                                      interaction.depth=c(2))

gbm_modelo1_early_stopping_bag_fraction <- data.frame()
for(bag_fraction in c(.05, .1, .2, .3, .5, .8, 1)) {
  temp <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                  method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                  distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  
  temp_2 <- train(factor(target)~mortality_rsi+ccsMort30Rate+bmi+month.8+Age,data=surgical_dataset,
                                  method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                                  distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  gbm_modelo1_early_stopping_bag_fraction <- rbind(gbm_modelo1_early_stopping_bag_fraction, temp$results, temp_2$results)
}
rm(temp); rm(temp_2)
gbm_modelo1_early_stopping_bag_fraction$bag.fraction <- c(rep(0.05,4), rep(.1,4), rep(.2,4), rep(.3,4), rep(.5,4), rep(.8,4), rep(1,4))


#  Modelo 2
gbm_modelo2_early_stopping <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                                    method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                                    distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo2_early_stopping_2 <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                                      method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                                      distribution="bernoulli", bag.fraction=1,verbose=FALSE)

gbm_modelo2_es_final <- rbind(gbm_modelo1_early_stopping$results, gbm_modelo2_early_stopping_2$results)
gbm_modelo2_es_final$shrinkage <- c(rep("0.2", 10), rep("0.3", 10))

gbm_modelo2_es_final %>% ggplot(aes(x = n.trees, y = Accuracy, label = n.trees,
                                    group = shrinkage, col = shrinkage)) + 
  geom_point() +
  geom_line() +
  geom_text_repel(data=gbm_modelo2_es_final %>% sample_frac(0.6)) +
  ggtitle("Evolucion Accuracy Modelo 2 (Early Stopping)")
#-- Llama la atencion 100 + 0.3 ; 200 + 0.3 ; 100 + 0.2 ; 500 + 0.2
#-- Tuneo bag.fraction
set.seed(1234)
gbmgrid_early_stopping<-expand.grid(shrinkage=c(0.2),
                                    n.minobsinnode=c(20),
                                    n.trees=c(100, 500),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(100, 200),
                                      interaction.depth=c(2))

gbm_modelo2_early_stopping_bag_fraction <- data.frame()
for(bag_fraction in c(.05, .1, .2, .3, .5, .8, 1)) {
  temp <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping,
                distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  
  temp_2 <- train(factor(target)~mortality_rsi+bmi+month.8+Age,data=surgical_dataset,
                  method="gbm",trControl=control,tuneGrid=gbmgrid_early_stopping_2,
                  distribution="bernoulli",bag.fraction=bag_fraction,verbose=FALSE)
  
  gbm_modelo2_early_stopping_bag_fraction <- rbind(gbm_modelo2_early_stopping_bag_fraction, temp$results, temp_2$results)
}
gbm_modelo2_early_stopping_bag_fraction$bag.fraction <- c(rep(0.05, 4), rep(.1, 4), rep(.2, 4), rep(.3, 4), rep(.5, 4), rep(.8, 4), rep(1, 4))
rm(temp); rm(temp_2); rm(bag_fraction)

#-- Analicemos el modelo 1
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
graph_2 <- ggplot(gbm_modelo2_early_stopping_bag_fraction, aes(x = n.trees, y = Accuracy, colour = bag.fraction)) +
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
# Como consecuencia, con 100 arboles, minobsinnode 20, shrinkage 0.2,0.3 y bag.fraction 0.5-0.8-1

#-- Comparacion modelos candidatos
#   Modelo 1
tuneo_gradient_boosting_modelo1 <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 0.8, 1), shrinkage = c(0.2, 0.3),
  interaction.depth = 2, grupos = 5, repe = 5, path.1 = "./charts/gradient_boosting/modelo1/05_tasa_fallos_modelo1_5rep.png",
  path.2 = "./charts/gradient_boosting/modelo1/05_auc_modelo1_5rep.png"
)

# ¿Y si elevamos a 10 el numero de repeticiones?
tuneo_gradient_boosting_modelo1_10rep <- tuneo_gradient_boosting(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  n.trees = 100, n.minobsinnode = 20, bag.fraction = c(0.5, 0.8, 1), shrinkage = c(0.2, 0.3),
  interaction.depth = 2, grupos = 5, repe = 10)


tuneo_modelo1 <- rbind(tuneo_gradient_boosting_modelo1, tuneo_gradient_boosting_modelo1_10rep)
tuneo_modelo1$rep <- c(rep("5", 30), rep("10", 60))

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
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/gradient_boosting/modelo1/05_auc_modelo1_10rep.png")




