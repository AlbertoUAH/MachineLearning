# ------------- Aumento numero de grupos y repeticioned ---------------
# Objetivo: estudiar la evolucion de los mejores modelos aumentando a 10 grupos y 20 repeticiones
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
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

#-- Bagging
bagging_1 <- cruzadarfbin(data=surgical_dataset, vardep=target,
                        listconti=var_modelo2,listclass=c(""),
                        grupos=10,sinicio=sinicio,repe=20,nodesize=20,
                        mtry=4,ntree=900, sampsize=1000, replace = TRUE)

bagging_2 <- cruzadarfbin(data=surgical_dataset, vardep=target,
                        listconti=var_modelo2,listclass=c(""),
                        grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                        mtry=4,ntree=900, sampsize=1000, replace = TRUE)


#-- Random Forest
random_forest_1 <- cruzadarfbin(data=surgical_dataset,
                                vardep=target,listconti=var_modelo2,
                                listclass=c(""),grupos=10,sinicio=sinicio,repe=20,
                                mtry=2,sampsize=1000, ntree=2000,nodesize=20,replace=TRUE)

random_forest_2 <- cruzadarfbin(data=surgical_dataset,
                              vardep=target,listconti=var_modelo2,
                              listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                              mtry=2,sampsize=1000, ntree=2000,nodesize=20,replace=TRUE)

#-- Gradient Boosting
gradient_boosting_1 <- cruzadagbmbin(data=surgical_dataset,
                                     vardep=target,listconti=var_modelo2,
                                     listclass=c(""),grupos=10,sinicio=sinicio,repe=20,
                                     n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                     interaction.depth=2, bag.fraction=0.5)

gradient_boosting_2 <- cruzadagbmbin(data=surgical_dataset,
                                     vardep=target,listconti=var_modelo2,
                                     listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                     n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                     interaction.depth=2, bag.fraction=0.5)

#-- XGboost
xgboost_1 <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
                            listconti=var_modelo2,listclass=c(""),
                            grupos=10,sinicio=sinicio,repe=20,
                            min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                            gamma=0,colsample_bytree=1,subsample=0.5)

xgboost_2 <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
                            listconti=var_modelo2,listclass=c(""),
                            grupos=grupos,sinicio=sinicio,repe=repe,
                            min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                            gamma=0,colsample_bytree=1,subsample=0.5)

modelos <- rbind(bagging_1, bagging_2, random_forest_1, random_forest_2, gradient_boosting_1, gradient_boosting_2,
                 xgboost_1, xgboost_2)

source("./librerias/cruzadas ensamblado binaria fuente.R")
# Ensamblado Bagging + XGboost
for (cv in list(c(5, 10), c(10, 20))) {
  bagging_ensemb <- cruzadarfbin(data=surgical_dataset, vardep=target,
                                 listconti=var_modelo2,listclass=c(""),
                                 grupos=cv[1],sinicio=sinicio,repe=cv[2],nodesize=20,
                                 mtry=4,ntree=900, sampsize=1000,replace = TRUE)
  
  medias_bagging    <- as.data.frame(bagging_ensemb[1])
  medias_bagging$modelo <-"Bagging"
  pred_bagging      <- as.data.frame(bagging_ensemb[2])
  pred_bagging$bagging <- pred_bagging$Yes
  
  xgboost_ensemb <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
                                   listconti=var_modelo2,listclass=c(""),
                                   grupos=cv[1],sinicio=sinicio,repe=cv[2],
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
  modelos <- rbind(modelos, medias0[c("tasa", "auc")])
}

modelos$modelo <- c(rep("Bagging", 30), rep("Random_Forest", 30), rep("Gradient_Boosting", 30), rep("XGboost", 30),
                    rep("Ensamblado", 30))
modelos$cv     <- rep(c(rep("10_folds_20_rep", 20), rep("5_folds_10_rep", 10)), 5)

modelos$modelo <- with(modelos,
                     reorder(modelo, tasa, mean))
ggplot(modelos, aes(x = modelo, y = tasa, colour = cv)) +
  geom_boxplot(notch = FALSE) +
  ggtitle("Tasa de fallos CV 5 folds vs CV 10 folds") +
  theme_bw()

modelos$modelo <- with(modelos,
                       reorder(modelo, auc, mean))
ggplot(modelos, aes(x = modelo, y = auc, colour = cv)) +
  geom_boxplot(notch = FALSE) +
  ggtitle("AUC CV 5 folds vs CV 10 folds") +
  theme_bw()


