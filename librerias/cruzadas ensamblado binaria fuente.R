# *********************************
# CRUZADAS PARA ENSAMBLADO DEPENDIENTE BINARIA
# *********************************
# VALIDACIÓN CRUZADA REPETIDA Y BOXPLOT para
# 
# LOGISTICA
# AVNNET
# RF
# GBM
# XGBM
# SVM

# ES NECESARIO QUE LA VARIABLE DEPENDIENTE ESTÉ CATEGORIZADA
# COMO Yes,No
 
# *********************************


library(dummies)
library(MASS)
library(reshape)
library(caret)

library(pROC)


# *********************************
# CRUZADA LOGISTICA
# ********************************* 

cruzadalogistica <- function(data=data,vardep=NULL,
 listconti=NULL,listclass=NULL,grupos=4,sinicio=1234,repe=5)
{

  if (any(listclass==c(""))==FALSE)
  {
   for (i in 1:dim(array(listclass))) {
    numindi<-which(names(data)==listclass[[i]])
    data[,numindi]<-as.character(data[,numindi])
    data[,numindi]<-as.factor(data[,numindi])
   }
  }   
  
  data[,vardep]<-as.factor(data[,vardep])
  
  # Creo la formula para la logistica
  
if (any(listclass==c(""))==FALSE)
  {
   koko<-c(listconti,listclass)
  }  else   {
   koko<-c(listconti)
  }
 
  modelo<-paste(koko,sep="",collapse="+")
  formu<-formula(paste(vardep,"~",modelo,sep=""))
  
  formu 
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  regresion <- train(formu,data=data,
   trControl=control,method="glm",family = binomial(link="logit"))                  
  preditest<-regresion$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
 return(list(medias,preditest))
  
 }




# *********************************
# CRUZADA avNNet
# **************

cruzadaavnnetbin<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",grupos=4,sinicio=1234,repe=5,
  size=c(5),decay=c(0.01),repeticiones=5,itera=100,trace=FALSE)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste(vardep,"~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  avnnetgrid <-  expand.grid(size=size,decay=decay,bag=FALSE)
  
  avnnet<- train(formu,data=databis,
   method="avNNet",linout = FALSE,maxit=itera,repeats=repeticiones,
   trControl=control,tuneGrid=avnnetgrid,trace=trace)
  
  print(avnnet$results)
  
  preditest<-avnnet$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }


# *********************************
# CRUZADA Random Forest
# ******************************


cruzadarfbin<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,nodesize=20,
  mtry=2,ntree=50,replace=TRUE,sampsize=1)
 { 
  
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  rfgrid <-expand.grid(mtry=mtry)
  
      if  (sampsize==1)
  {
    rf<- train(formu,data=databis,
   method="rf",trControl=control,
   tuneGrid=rfgrid,nodesize=nodesize,replace=replace,ntree=ntree)
  }
  
else  if  (sampsize!=1)
  {
    rf<- train(formu,data=databis,
   method="rf",trControl=control,
   tuneGrid=rfgrid,nodesize=nodesize,replace=replace,sampsize=sampsize,
   ntree=ntree)
 }
  
  print(rf$results)
  
  preditest<-rf$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }

# ***************************************************************
# gbm : parámetros

# 
#     Number of Boosting Iterations (n.trees, numeric)
#     Max Tree Depth (max.depth, numeric)
#     Shrinkage (shrinkage, numeric)
#     Min. Terminal Node Size (n.minobsinnode, numeric)
#    
# ***************************************************************



cruzadagbmbin<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,bag.fraction=1,
  n.minobsinnode=20,shrinkage=0.1,n.trees=100,interaction.depth=2)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  
  
   gbmgrid <-expand.grid(n.minobsinnode=n.minobsinnode,
    shrinkage=shrinkage,n.trees=n.trees,
    interaction.depth=interaction.depth)
  
  gbm<- train(formu,data=databis,
   method="gbm",trControl=control,bag.fraction=bag.fraction,
   tuneGrid=gbmgrid,distribution="bernoulli",verbose=FALSE)
  
  print(gbm$results)
  
  preditest<-gbm$pred
  
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }
 

# ***************************************************************
# xgboost: parámetros

   # nrounds (# Boosting Iterations)
    # max_depth (Max Tree Depth)
    # eta (Shrinkage)
    # gamma (Minimum Loss Reduction)
    # colsample_bytree (Subsample Ratio of Columns)
    # min_child_weight (Minimum Sum of Instance Weight)
    # subsample (Subsample Percentage)
#    
# PONER linout = FALSE
# ***************************************************************

cruzadaxgbmbin<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,
  min_child_weight=20,eta=0.1,nrounds=100,max_depth=2,
  gamma=0,colsample_bytree=1,subsample=1,alpha=0,lambda=0,lambda_bias=0)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  xgbmgrid <-expand.grid( min_child_weight=min_child_weight,
  eta=eta,nrounds=nrounds,max_depth=max_depth,
  gamma=gamma,colsample_bytree=colsample_bytree,subsample=subsample)
  
  xgbm<- train(formu,data=databis,
   method="xgbTree",trControl=control,
   tuneGrid=xgbmgrid,verbose=FALSE,
   alpha=alpha,lambda=lambda,lambda_bias=lambda_bias)
  
  print(xgbm$results)
  
  preditest<-xgbm$pred
  
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }
 


# ***************************************************************
# svmLinear: parámetros

# Cost (C, numeric)     

# PONER linout = FALSE
# ***************************************************************


cruzadaSVMbin<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,
  C=1,replace=TRUE)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  SVMgrid <-expand.grid(C=C)
  
  SVM<- train(formu,data=databis,
   method="svmLinear",trControl=control,
   tuneGrid=SVMgrid,replace=replace)
  
  print(SVM$results)
  
  preditest<-SVM$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }





cruzadaSVMbinPoly<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,
  C=1,degree=2,scale=1)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  SVMgrid <-expand.grid(C=C,degree=degree,scale=scale)
  
  SVM<- train(formu,data=databis,
   method="svmPoly",trControl=control,
   tuneGrid=SVMgrid,replace=replace)
  
  print(SVM$results)
  
  preditest<-SVM$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }



# ***************************************************************
# svmRadial: parámetros

    # Sigma (sigma, numeric)
    # Cost (C, numeric)

# PONER linout = FALSE
# ***************************************************************



cruzadaSVMbinRBF<-
 function(data=data,vardep="vardep",
  listconti="listconti",listclass="listclass",
  grupos=4,sinicio=1234,repe=5,
  C=1,sigma=1)
 { 
  
  # Preparación del archivo
  
  # b)pasar las categóricas a dummies
  
if (any(listclass==c(""))==FALSE)
  {
   databis<-data[,c(vardep,listconti,listclass)]
   databis<- dummy.data.frame(databis, listclass, sep = ".")
  }  else   {
   databis<-data[,c(vardep,listconti)]
  }
  
  # c)estandarizar las variables continuas
  
  # Calculo medias y dtipica de datos y estandarizo (solo las continuas)
  
  means <-apply(databis[,listconti],2,mean)
  sds<-sapply(databis[,listconti],sd)
  
  # Estandarizo solo las continuas y uno con las categoricas
  
  datacon<-scale(databis[,listconti], center = means, scale = sds)
  numerocont<-which(colnames(databis)%in%listconti)
  databis<-cbind(datacon,databis[,-numerocont,drop=FALSE ])
  
  databis[,vardep]<-as.factor(databis[,vardep])
  
  formu<-formula(paste("factor(",vardep,")~.",sep=""))
  
  # Preparo caret   
  
  set.seed(sinicio)
  control<-trainControl(method = "repeatedcv",number=grupos,repeats=repe,
   savePredictions = "all",classProbs=TRUE) 
  
  # Aplico caret y construyo modelo
  
  SVMgrid <-expand.grid(C=C,sigma=sigma)
  
  SVM<- train(formu,data=databis,
   method="svmRadial",trControl=control,
   tuneGrid=SVMgrid,replace=replace)
  
  print(SVM$results)
  
  preditest<-SVM$pred
  
  preditest$prueba<-strsplit(preditest$Resample,"[.]")
  preditest$Fold <- sapply(preditest$prueba, "[", 1)
  preditest$Rep <- sapply(preditest$prueba, "[", 2)
  preditest$prueba<-NULL
  
  tasafallos<-function(x,y) {
   confu<-confusionMatrix(x,y)
   tasa<-confu[[3]][1]
   return(tasa)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   tabla<-table(preditest$Rep)
listarep<-c(names(tabla))
medias<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
tasa=1-tasafallos(paso1$pred,paso1$obs)  
medias<-rbind(medias,tasa)
}
names(medias)<-"tasa"

  
  # CalculamoS AUC  por cada Repetición de cv 
  # Definimnos función
  
  auc<-function(x,y) {
   curvaroc<-roc(response=x,predictor=y)
   auc<-curvaroc$auc
   return(auc)
  }
  
  # Aplicamos función sobre cada Repetición
  
  
   
   mediasbis<-data.frame()
for (repi in listarep) {
paso1<-preditest[which(preditest$Rep==repi),]
auc=suppressMessages(auc(paso1$obs,paso1$Yes))
mediasbis<-rbind(mediasbis,auc)
}
names(mediasbis)<-"auc"

  
  # Unimos la info de auc y de tasafallos
  
  medias$auc<-mediasbis$auc
  
   return(list(medias,preditest))
  
 }

