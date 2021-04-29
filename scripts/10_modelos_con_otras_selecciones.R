# ------------- Prueba seleccion de variables descartadas ---------------
# Objetivo: estudiar el comportamiento de los mejores modelos con las selecciones
#           de variables descartadas durante la practica
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

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 2
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

#-- Recordemos los mejores modelos obtenidos hasta el momento
#   Bagging, XGboost, Random Forest, GBM, Avnnet y SVM RBF

#-- Seleccion de variables empleadas
modelo_1 <- list("modelo1" = c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age"))
modelo_2 <- list("modelo2" = c("mortality_rsi", "bmi", "month.8", "Age"))

#-- Recordemos las selecciones que desctartamos al comienzo
candidato_aic_2 <- list("candidato_aic_2" = c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", 
                                            "dow.0", "Age", "moonphase.0", "month.0", "baseline_osteoart", 
                                            "baseline_charlson", "ahrq_ccs"))

candidato_aic_3 <- list("candidato_aic_3" = c("mortality_rsi", "ahrq_ccs", "bmi", "month.8", "Age"))

candidato_bic_2 <- list("candidato_bic_2" = c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0",           
                                              "Age", "moonphase.0", "baseline_osteoart"))

candidato_bic_4 <- list("candidato_bic_4" = c("mortality_rsi", "baseline_osteoart", "bmi", "month.8", "Age"))

candidato_rfe_lr_top3 <- list("candidato_rfe_lr_top3" = c("ccsMort30Rate", "mortality_rsi", "bmi"))

candidato_rfe_rf      <- list("candidato_rfe_rf" = c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs"))

sel_variables <- list(modelo_1, modelo_2, candidato_aic_2, candidato_aic_3, candidato_bic_2, candidato_bic_4,
                   candidato_rfe_lr_top3, candidato_rfe_rf)


#-- Tuneo de los modelos finales
#   Regresion logistica
modelos <- data.frame()

for (vars in sel_variables) {
  logistica <- cruzadalogistica(data=surgical_dataset, vardep=target,
                                listconti=vars[[1]], listclass=c(""),
                                grupos=grupos,sinicio=1234,repe=repe)[[1]]
  
  logistica$tipo <- "Logistica"
  logistica$modelo <- names(vars)
  
  modelos <- rbind(modelos, logistica)
}

#  Avnnet
for (vars in sel_variables) {
  avnnet  <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                              listconti=vars[[1]], listclass=c(""),
                              grupos=grupos,sinicio=sinicio,repe=repe, 
                              size=10,decay=0.01,repeticiones=5,itera=200)[[1]]
  
  avnnet$tipo <- "Avnnet"
  avnnet$modelo <- names(vars)
  
  modelos <- rbind(modelos, avnnet)
}

#  Bagging
for (vars in sel_variables) {
  bagging <- cruzadarfbin(data=surgical_dataset, vardep=target,
                          listconti=vars[[1]],listclass=c(""),
                          grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                          mtry=length(vars[[1]]),ntree=900, sampsize=1000, replace = TRUE)
  
  bagging$tipo <- "Bagging"
  bagging$modelo <- names(vars)
  
  modelos <- rbind(modelos, bagging)
}

#  Random Forest
for (vars  in sel_variables) {
  random_forest <- cruzadarfbin(data=surgical_dataset,
                                vardep=target,listconti=vars[[1]],
                                listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                mtry=2,sampsize=1000, ntree=2000,nodesize=20,replace=TRUE)
  
  random_forest$tipo <- "Random_Forest"
  random_forest$modelo <- names(vars)
  
  modelos <- rbind(modelos, random_forest)
}

# Gradient Boosting
for (vars  in sel_variables) {
  gradient_boosting <- cruzadagbmbin(data=surgical_dataset,
                                     vardep=target,listconti=vars[[1]],
                                     listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                     n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                     interaction.depth=2, bag.fraction=0.5)

  gradient_boosting$tipo   <-"Gradient_Boosting"
  gradient_boosting$modelo <- names(vars)
  
  modelos <- rbind(modelos, gradient_boosting)
}

# XGboost
for (vars  in sel_variables) {
  xgboost <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
                            listconti=vars[[1]],listclass=c(""),
                            grupos=grupos,sinicio=sinicio,repe=repe,
                            min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                            gamma=0,colsample_bytree=1,subsample=0.5)
  
  xgboost$tipo   <-"XGboost"
  xgboost$modelo <- names(vars)
  
  modelos <- rbind(modelos, xgboost)
}

# SVM RBF
for (vars in sel_variables) {
  svm_rbf   <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target, listconti=vars[[1]],
                                listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                                C=1, sigma=5)
  
  svm_rbf$tipo   <-"SVM_RBF"
  svm_rbf$modelo <- names(vars)
  
  modelos <- rbind(modelos, svm_rbf)
}

source("./librerias/cruzadas ensamblado binaria fuente.R")
# Ensamblado Bagging + XGboost
for (vars in sel_variables) {
  bagging_ensemb <- cruzadarfbin(data=surgical_dataset, vardep=target,
                                 listconti=vars[[1]],listclass=c(""),
                                 grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                                 mtry=4,ntree=900, sampsize=1000,replace = TRUE)
  
  medias_bagging    <- as.data.frame(bagging_ensemb[1])
  medias_bagging$modelo <-"Bagging"
  pred_bagging      <- as.data.frame(bagging_ensemb[2])
  pred_bagging$bagging <- pred_bagging$Yes
  
  xgboost_ensemb <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
                                   listconti=vars[[1]],listclass=c(""),
                                   grupos=grupos,sinicio=sinicio,repe=repe,
                                   min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                                   gamma=0,colsample_bytree=1,subsample=0.5)
  
  medias_xgboost    <- as.data.frame(xgboost_ensemb[1])
  medias_xgboost$modelo <-"XGboost"
  pred_xgboost      <- as.data.frame(xgboost_ensemb[2])
  pred_xgboost$xgboost <- pred_xgboost$Yes
  
  unipredi <- cbind(pred_bagging, pred_xgboost)
  unipredi <- unipredi[, !duplicated(colnames(unipredi))]
  
  unipredi[, "bagging-xgboost"] <- (unipredi[, "bagging"] + unipredi[, "xgboost"]) / 2
  
  listado <- c("bagging-xgboost")
  
  # Cambio a Yes, No, todas las predicciones
  repeticiones<-nlevels(factor(unipredi$Rep))
  unipredi$Rep<-as.factor(unipredi$Rep)
  unipredi$Rep<-as.numeric(unipredi$Rep)
  
  
  medias0<-data.frame(c())
  for (prediccion in listado)
  {
    unipredi$proba<-unipredi[,prediccion]
    unipredi[,prediccion]<-ifelse(unipredi[,prediccion]>0.5,"Yes","No")
    for (repe in 1:repeticiones)
    {
      paso <- unipredi[(unipredi$Rep==repe),]
      pre<-factor(paso[,prediccion])
      archi<-paso[,c("proba","obs")]
      archi<-archi[order(archi$proba),]
      obs<-paso[,c("obs")]
      tasa=1-tasafallos(pre,obs)
      t<-as.data.frame(tasa)
      t$modelo<-prediccion
      auc<-suppressMessages(auc(archi$obs,archi$proba))
      t$auc<-auc
      medias0<-rbind(medias0,t)
    }
  }
  
  medias0$tipo   <-"Ensamblado"
  medias0$modelo <- names(vars)
  
  modelos <- rbind(modelos, medias0[c("tasa", "auc", "tipo", "modelo")])
  print(names(vars))
  print("FINISHED!!!")
}
rm(unipredi); rm(bagging_ensemb); rm(xgboost_ensemb); rm(medias_bagging); rm(medias_xgboost)
rm(pred_bagging); rm(pred_xgboost); rm(listado); rm(prediccion); rm(paso); rm(pre); rm(archi)
rm(obs); rm(tasa); rm(t); rm(auc) ;rm(medias0)


#-- Comenzamos con los cuatro mejores modelos: Bagging, Random Forest, XGboost + Ensamblado
#   Tasa de fallos
modelos$modelo <- with(modelos,
                                reorder(modelo,tasa, mean))
ggplot(modelos[modelos$tipo %in% c("Bagging", "Random_Forest", "XGboost", "Ensamblado"), ], aes(x = modelo, y = tasa, colour = tipo)) +
  geom_boxplot(adjust = 1.1) +
  facet_grid( . ~ tipo, scales = "free", space = "free") +
  ggtitle("Tasa de fallos por modelo") + 
  theme(axis.text.x = element_text(angle = 45), legend.position = "none")


#   AUC
modelos$modelo <- with(modelos, reorder(modelo,auc, mean))
ggplot(modelos[modelos$tipo %in% c("Bagging", "Random_Forest", "XGboost", "Ensamblado"), ], aes(x = modelo, y = auc, colour = tipo)) +
  geom_boxplot(adjust = 1.1) +
  facet_grid( . ~ tipo, scales = "free", space = "free") +
  ggtitle("AUC por modelo") + 
  theme(axis.text.x = element_text(angle = 45), legend.position = "none")

#-- Si nos vamos con gbm, avnnet y SVM RBF
#   Tasa de fallos
modelos$modelo <- with(modelos, reorder(modelo,tasa, mean))
ggplot(modelos[modelos$tipo %in% c("Gradient_Boosting", "Avnnet", "SVM_RBF"), ], aes(x = modelo, y = tasa, colour = tipo)) +
  geom_boxplot(adjust = 1.1) +
  facet_grid( . ~ tipo, scales = "free", space = "free") +
  ggtitle("Tasa de fallos por modelo") + 
  theme(axis.text.x = element_text(angle = 45), legend.position = "none")


#   AUC
modelos$modelo <- with(modelos, reorder(modelo,auc, mean))
ggplot(modelos[modelos$tipo %in% c("Gradient_Boosting", "Avnnet", "SVM_RBF"), ], aes(x = modelo, y = auc, colour = tipo)) +
  geom_boxplot(adjust = 1.1) +
  facet_grid( . ~ tipo, scales = "free", space = "free") +
  ggtitle("AUC por modelo") + 
  theme(axis.text.x = element_text(angle = 45), legend.position = "none")

source("./librerias/librerias_propias.R")
source("./librerias/cruzada SVM binaria lineal.R")
# ¿Y si tuneamos el mtry en el modelo Random Forest?
rf_modelo_aic_df <- data.frame()
for(mtry in 3:10) {
  rf_modelo_aic <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = candidato_aic$candidato_aic,
                                 nodesizes = 20, sampsizes = 1000, mtry = mtry,
                                 ntree = 2000, grupos = 5, repe = 10, replace = FALSE)
  rf_modelo_aic$mtry <- rep(mtry, 10)
  rf_modelo_aic_df <- rbind(rf_modelo_aic_df, rf_modelo_aic)
  print(rf_modelo_aic_df)
}

rf_modelo_bic_df <- data.frame()
for(mtry in 3:9) {
  rf_modelo_bic <- tuneo_bagging(surgical_dataset, target = target,
                                 lista.continua = candidato_bic$candidato_bic,
                                 nodesizes = 20,
                                 sampsizes = 1000, mtry = mtry,
                                 ntree = 2000, grupos = 5, repe = 10, replace = FALSE)
  rf_modelo_bic$mtry <- rep(mtry, 10)
  rf_modelo_bic_df <- rbind(rf_modelo_bic_df, rf_modelo_bic)
  print(rf_modelo_bic_df)
}
rm(rf_modelo_aic); rm(rf_modelo_bic)

#-- Analicemos el AUC
ggplot(rf_modelo_aic_df, aes(x = factor(mtry), y = auc)) +
  geom_boxplot(adjust = 1.1) +
  ggtitle("AUC por modelo")

#-- Analicemos el AUC
ggplot(rf_modelo_bic_df, aes(x = factor(mtry), y = auc)) +
  geom_boxplot(adjust = 1.1) +
  ggtitle("AUC por modelo")

#-- Con ambos candidatos, el maximo AUC es de 0.915, muy similar al modelo 2








