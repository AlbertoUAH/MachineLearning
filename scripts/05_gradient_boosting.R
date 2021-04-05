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
                                    n.trees=c(50, 100, 300, 500, 800, 1000, 1500, 2000, 2500, 5000),
                                    interaction.depth=c(2))

gbmgrid_early_stopping_2<-expand.grid(shrinkage=c(0.3),
                                      n.minobsinnode=c(20),
                                      n.trees=c(50, 100, 300, 500, 800, 1000, 1500, 2000, 2500, 5000),
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
#-- Con un shrinkage de 0.3 + 100 arboles obtiene un buen valor

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

#-- Nuevamente, parece una buena opcion emplear 0.3 de shrinkage y 100 arboles


