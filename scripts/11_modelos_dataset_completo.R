# ------------- Prueba conjunto total de observaciones ---------------
# Objetivo: estudiar el comportamiento de los mejores modelos con el
#           dataset original
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
  library(stringi)       # Tratamiento de strings
  library(ggplotgui)     # Interfaz ggplot2
  
  source("./librerias/librerias_propias.R")
  source("./librerias/cruzada SVM binaria polinomial.R")
  source("./librerias/cruzada SVM binaria lineal.R")
})

# Funcion para calcular la tasa de fallos
tasafallos<-function(x,y) {
  confu<-confusionMatrix(x,y)
  tasa<-confu[[3]][1]
  return(tasa)
}

# Funcion para calcular el AUC
auc<-function(x,y) {
  curvaroc<-roc(response=x,predictor=y)
  auc<-curvaroc$auc
  return(auc)
}

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)
surgical_dataset_completo <- rbind(surgical_dataset, surgical_test_data)
rm(surgical_test_data)

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 2
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

datasets <- list(surgical_dataset, surgical_dataset_completo)
names(datasets) <- c("Subconjunto dataset", "Dataset original")

#-- Tuneo de los modelos finales
#   Regresion logistica
modelos <- data.frame()

for (i in 1:2) {
  logistica <- cruzadalogistica(data=datasets[[i]], vardep=target,
                                listconti=var_modelo2, listclass=c(""),
                                grupos=grupos,sinicio=sinicio,repe=repe)[[1]]
  
  logistica$tipo <- "Logistica"
  logistica$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, logistica)
}

#  Avnnet
for (i in 1:2) {
  avnnet  <- cruzadaavnnetbin(data=datasets[[i]],vardep=target,
                              listconti=var_modelo2, listclass=c(""),
                              grupos=grupos,sinicio=sinicio,repe=repe, 
                              size=10,decay=0.01,repeticiones=5,itera=200)[[1]]
  
  avnnet$tipo <- "Avnnet"
  avnnet$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, avnnet)
}

#  Bagging
for (i in 1:2) {
  bagging <- cruzadarfbin(data=datasets[[i]], vardep=target,
                          listconti=var_modelo2,listclass=c(""),
                          grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                          mtry=4,ntree=900, sampsize=1000, replace = TRUE)
  
  bagging$tipo <- "Bagging"
  bagging$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, bagging)
}

#  Random Forest
for (i  in 1:2) {
  random_forest <- cruzadarfbin(data=datasets[[i]],
                                vardep=target,listconti=var_modelo2,
                                listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                mtry=2,sampsize=1000, ntree=2000,nodesize=20,replace=TRUE)
  
  random_forest$tipo <- "Random_Forest"
  random_forest$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, random_forest)
}

# Gradient Boosting
for (i  in 1:2) {
  gradient_boosting <- cruzadagbmbin(data=datasets[[i]],
                                     vardep=target,listconti=var_modelo2,
                                     listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                     n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                     interaction.depth=2, bag.fraction=0.5)
  
  gradient_boosting$tipo   <-"gbm"
  gradient_boosting$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, gradient_boosting)
}

# SVM Lineal
for (i in 1:2) {
  svm_lin   <- cruzadaSVMbin(data=datasets[[i]], vardep=target, listconti=var_modelo1,
                             listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                             C=0.01, replace=TRUE)
  
  svm_lin$tipo   <-"SVM_Lineal"
  svm_lin$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, svm_lin)
}

# SVM Polinomial
for (i in 1:2) {
  svm_poly   <- cruzadaSVMbinPoly(data=datasets[[i]], vardep=target, listconti=var_modelo2,
                                  listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                                  C=0.01, degree=2, scale=0.1)
  
  svm_poly$tipo   <-"SVM_Poly"
  svm_poly$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, svm_poly)
}


# SVM RBF
for (i in 1:2) {
  svm_rbf   <- cruzadaSVMbinRBF(data=datasets[[i]], vardep=target, listconti=var_modelo2,
                                listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                                C=0.5, sigma=5)
  
  svm_rbf$tipo   <-"SVM_RBF"
  svm_rbf$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, svm_rbf)
}

# XGboost
for (i  in 1:2) {
  xgboost <- cruzadaxgbmbin(data=datasets[[i]],vardep=target,
                            listconti=var_modelo2,listclass=c(""),
                            grupos=grupos,sinicio=sinicio,repe=repe,
                            min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                            gamma=0,colsample_bytree=1,subsample=1)
  
  xgboost$tipo   <-"XGboost"
  xgboost$modelo <- names(datasets)[i]
  
  modelos <- rbind(modelos, xgboost)
}
rm(i)

#-- Comparamos las salidas
#   Tasa de fallos
modelos$tipo <- with(modelos,
                       reorder(tipo, tasa, mean))
ggplot(modelos, aes(x = tipo, y = tasa, colour = modelo)) +
  geom_boxplot(notch = FALSE) +
  ggtitle("Comparacion tasa de fallos dataset (subconjunto) y dataset original") +
  theme_bw() + theme(text = element_text(face = "bold", size = 13))
ggsave('./charts/comparacion_datasets_tasa_fallos.png')

#  AUC
modelos$tipo <- with(modelos,
                     reorder(tipo, auc, mean))
ggplot(modelos, aes(x = tipo, y = auc, colour = modelo)) +
  geom_boxplot(notch = FALSE) +
  ggtitle("Comparacion AUC dataset (subconjunto) y dataset original") +
  theme_bw() + theme(text = element_text(face = "bold", size = 13))
ggsave('./charts/comparacion_datasets_auc.png')
