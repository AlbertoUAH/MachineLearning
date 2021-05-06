# ------------- XGboost ---------------
# Objetivo: elaborar el mejor modelo de XGboost de acuerdo
#           a los valores de prediccion obtenidos
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  library(ggrepel)       # Labels ggplot2
  
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

#-- Probamos con un tuneo inicial
#   Modelo 1

xgbmgrid <- expand.grid(min_child_weight=c(5,10,20),eta=c(0.1,0.05,0.03,0.01,0.001), nrounds=c(100,500,1000,5000),
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=1)
set.seed(1234)
control<-trainControl(method = "repeatedcv",number=5,repeats = 5,
                      savePredictions = "all",classProbs=TRUE)
xgbm<- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),data=surgical_dataset,
             method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)
xgbm
plot(xgbm, main = "Tuneo parametros XGBoost Modelo 1")

# Modelo 2
set.seed(1234)
xgbm_2 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)
xgbm_2



plot(xgbm_2, main = "Tuneo parametros XGBoost Modelo 2")


#-- En ambos modelos, practicamente llegamos a la misma conclusion: debe tomarse un shrinkage bajo y un numero bajo de iteraciones
#   (entre 100 y 500), o bien otro extremo y aplicar un elevado numero de irteraciones (en torno a 5000) y un shrinkage bajo, 
#   aunque la diferencia entre ambos es prácticamente nula. Ademas, y dado que las diferencias entre los tres modelos son pequeñas,
#   es recomendable emplear un valor min_child_weight alto (20).
#   Por tanto, probamos a fijar parametros como min_child_weight y eta

#-- Podriamos probar a aumentar el valor eta
xgbmgrid <- expand.grid(min_child_weight=c(20),eta=c(0.1, 0.2, 0.3, 0.4), nrounds=c(50,100,500,1000,5000),
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=1)
set.seed(1234)
xgbm_3 <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

# Modelo 2
set.seed(1234)
xgbm_4 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

#-- Estudio del Early Stopping
#   Modelo 1 y Modelo 2
gbm_modelos_early_stopping <- rbind(xgbm_3$results, xgbm_4$results)
gbm_modelos_early_stopping$eta <- as.factor(gbm_modelos_early_stopping$eta)
gbm_modelos_early_stopping$Modelo <- c(rep("Modelo 1", 20), rep("Modelo 2", 20))

#   Modelo 1
gbm_modelos_early_stopping %>% filter(Modelo == "Modelo 1") %>% 
  ggplot(aes(x = nrounds, y = Accuracy, label = nrounds,
                          group = eta, col = eta)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelo 1 (Early Stopping)")

#   Modelo 2
gbm_modelos_early_stopping[gbm_modelos_early_stopping$eta == 0.1 & gbm_modelos_early_stopping$nrounds >= 100, ] %>% filter(Modelo == "Modelo 2") %>% 
  ggplot(aes(x = factor(nrounds), y = Accuracy)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelo 2 (Early Stopping)")

gbm_modelos_early_stopping[gbm_modelos_early_stopping$eta == 0.1 & gbm_modelos_early_stopping$nrounds >= 100, ] %>% filter(Modelo == "Modelo 2") %>% ggplot(aes(x = factor(nrounds), y = Accuracy, label = factor(nrounds),
                                                                           group = eta, col = eta)) + 
  geom_point() +
  geom_line() + geom_label(data = gbm_modelos_early_stopping[gbm_modelos_early_stopping$eta == 0.1 & gbm_modelos_early_stopping$nrounds == 100, ]  %>% filter(Modelo == "Modelo 2"), aes(label = round(Accuracy, 3)), show.legend = FALSE) +
  ggtitle("Evolucion XGboost (Early Stopping)") +
  theme(
    text = element_text(size=14, face = "bold")
  )

#  En general, en ambos modelos se obtiene un buen accuracy con 100 iteraciones y un valor eta = 0.1

#-- Tuneo max_depth
# Modelo 1
xgbmgrid <- expand.grid(min_child_weight=c(20),eta=c(0.1), nrounds=c(100),
                        max_depth=c(1, 3, 6, 10, 15, 20),gamma=0,colsample_bytree=1,subsample=1)
set.seed(1234)
xgbm_5 <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

# Modelo 2
set.seed(1234)
xgbm_6 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

gbm_modelos_max_depth     <- rbind(xgbm_5$results, xgbm_6$results)
gbm_modelos_max_depth$max_depth <- as.factor(gbm_modelos_max_depth$eta)
gbm_modelos_max_depth$Modelo    <- c(rep("Modelo 1", 6), rep("Modelo 2", 6))

gbm_modelos_max_depth[gbm_modelos_max_depth$Modelo == "Modelo 1", ] %>% 
  ggplot(aes(x = max_depth, y = Accuracy, label = nrounds)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelos 1 y 2 (max_depth)")

# Por lo general, max_depth = 6 parece ser un buen tamaño

#-- Por ultimo ¿Y si tuneamos el tamaño de la muestra?
#   Porbamos con los mismos parametros que en gbm
# Modelo 1
xgbmgrid <- expand.grid(min_child_weight=c(20),eta=c(0.1), nrounds=c(100),
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1))
set.seed(1234)
xgbm_7 <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

# Modelo 2
set.seed(1234)
xgbm_8 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),data=surgical_dataset,
                method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

gbm_modelos_subsample     <- rbind(xgbm_7$results, xgbm_8$results)
gbm_modelos_subsample$subsample <- as.factor(gbm_modelos_subsample$subsample)
gbm_modelos_subsample$Modelo    <- c(rep("Modelo 1", 7), rep("Modelo 2", 7))

gbm_modelos_subsample %>% 
  ggplot(aes(x = subsample, y = Accuracy, label = nrounds,
             group = Modelo, col = Modelo)) + 
  geom_point() +
  geom_line() +
  ggtitle("Evolucion Accuracy Modelos 1 y 2 (subsample)")

# Con subsample de 0.5 o 1 parecen ser buenas alternativas

#-- Aumentamos a 10 repeticiones
xgbmgrid <- expand.grid(min_child_weight=20,eta=0.1, nrounds=100,
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=c(0.5,1))

tuneo_xgboost_modelo1 <- tuneo_xgboost(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  nrounds = 100, eta = 0.1, min_child_weight = 20, grupos = 5, repe = 5, subsample = c(0.5, 1), path.1 = "",
  path.2 = ""
)

tuneo_xgboost_modelo1_10rep <- tuneo_xgboost(
  dataset = surgical_dataset, lista.continua = var_modelo1, target = target,
  nrounds = 100, eta = 0.1, min_child_weight = 20, grupos = 5, repe = 10, subsample = c(0.5, 1), path.1 = "",
  path.2 = ""
)

tuneo_xgboost_modelo2 <- tuneo_xgboost(
  dataset = surgical_dataset, lista.continua = var_modelo2, target = target,
  nrounds = 100, eta = 0.1, min_child_weight = 20, grupos = 5, repe = 5, subsample = c(0.5, 1), path.1 = "",
  path.2 = ""
)

tuneo_xgboost_modelo2_10rep <- tuneo_xgboost(
  dataset = surgical_dataset, lista.continua = var_modelo2, target = target,
  nrounds = 100, eta = 0.1, min_child_weight = 20, grupos = 5, repe = 10, subsample = c(0.5, 1), path.1 = "",
  path.2 = ""
)

tuneo_modelo1 <- rbind(tuneo_xgboost_modelo1, tuneo_xgboost_modelo1_10rep)
tuneo_modelo1$rep <- c(rep("5", 10), rep("10", 20))

tuneo_modelo2 <- rbind(tuneo_xgboost_modelo2, tuneo_xgboost_modelo2_10rep)
tuneo_modelo2$rep <- c(rep("5", 10), rep("10", 20))

tuneo_modelo1$modelo <- with(tuneo_modelo1, reorder(modelo,tasa, mean))
ggplot(tuneo_modelo1, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")
ggsave("./charts/xgboost/tasa_fallos_modelo1_05_10_rep.png")

tuneo_modelo2$modelo <- with(tuneo_modelo2, reorder(modelo,auc, mean))
p <- ggplot(tuneo_modelo2, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por subsample") +
  theme(
    text = element_text(size=14, face = "bold")
  )

tuneo_modelo2$modelo <- with(tuneo_modelo2, reorder(modelo,tasa, mean))
q <- ggplot(tuneo_modelo2, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por subsample") +
  theme(
    text = element_text(size=14, face = "bold")
  )

ggsave("./charts/xgboost/auc_modelo2_05_10_rep.png")

# Entre 0.5 y 1 no existe una diferencia muy significativa, por lo que nos decantamos por
# 0.5 subsample

#-- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                        savePredictions = "all",classProbs=TRUE)

xgbmgrid <- expand.grid(min_child_weight=20,eta=0.1, nrounds=100,
                        max_depth=6,gamma=0,colsample_bytree=1,subsample=1)

# Modelo 1
set.seed(1234)
xgbm_modelo1 <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),data=surgical_dataset,
                      method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

# Modelo 2
set.seed(1234)
xgbm_modelo2 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),data=surgical_dataset,
                      method="xgbTree",trControl=control,tuneGrid=xgbmgrid,verbose=FALSE)

matriz_conf_1 <- matriz_confusion_predicciones(xgbm_modelo1, NULL, surgical_test_data, 0.5)

matriz_conf_2 <- matriz_confusion_predicciones(xgbm_modelo2, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No  Yes
# No         6445  149
# Yes         722 1465

# Modelo 2
# Prediction   No  Yes
#        No  6464  130
#        Yes  752 1435

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "xgboost"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)
modelos_actuales$tipo <- c(rep("LOGISTICA", 10), rep("RED NEURONAL", 10), rep("BAGGING", 10), rep("RANDOM FOREST", 10),
                           rep("GBM", 10), rep("SVM", 30), rep("XGBOOST", 10))

modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ]$modelo <- with(modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ],
                                reorder(modelo,tasa, mean))
p <- ggplot(modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ], aes(x = modelo, y = tasa)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo") + theme(axis.text.x = element_text(angle = 45, vjust = 0.5), text = element_text(size=14, face = "bold"))

ggsave('./charts/comparativas/07_log_avnnet_bagging_rf_gbm_svm_xgboost_tasa.jpeg')


modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ]$modelo <- with(modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ],
                                                                                                              reorder(modelo,auc, mean))
q <- ggplot(modelos_actuales[modelos_actuales$tipo %in% c("XGBOOST", "GBM", "RANDOM FOREST", "BAGGING"), ], aes(x = modelo, y = auc)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo") + theme(axis.text.x = element_text(angle = 45, vjust = 0.5), text = element_text(size=14, face = "bold"))

ggsave('./charts/comparativas/07_log_avnnet_bagging_rf_gbm_svm_xgboost_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#          xgboost
#          svm
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
save.image(file = "./rdata/XGboost.RData")

aux <- data.frame()
for (max_depth in c(1,3,6,10,15,20)) {
  salida <- cruzadaxgbmbin(data=surgical_dataset, vardep=target,
                           listconti=var_modelo2,
                           listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,nrounds=100,eta=0.01,
                           min_child_weight=20,gamma=0,colsample_bytree=1,subsample=1,max_depth=max_depth)
  salida$max_depth <- rep(max_depth, 5)
  aux <- rbind(aux, salida)
}

aux$modelo <- with(aux, reorder(max_depth,tasa, mean))
p <- ggplot(aux, aes(x = factor(max_depth), y = tasa)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por max_depth") +
  theme(
    text = element_text(size=14, face = "bold")
  )


aux$modelo <- with(aux, reorder(max_depth,auc, mean))
q <- ggplot(aux, aes(x = factor(max_depth), y = auc)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por max_depth") +
  theme(
    text = element_text(size=14, face = "bold")
  )





