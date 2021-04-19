


    # nrounds (# Boosting Iterations)
    # max_depth (Max Tree Depth)
    # eta (Shrinkage)
    # gamma (Minimum Loss Reduction)
    # colsample_bytree (Subsample Ratio of Columns)
    # min_child_weight (Minimum Sum of Instance Weight)
    # subsample (Subsample Percentage)


library(dummies)
library(MASS)
library(reshape)
library(caret)

library(pROC)


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
  
  return(medias)
  
 }
 

# load ("saheartbis.Rda")
# source ("cruzadas avnnet y log binaria.R")
# source ("cruzada arbolbin.R")
# source ("cruzada rf binaria.R")
# source ("cruzada gbm binaria.R")
# source ("cruzada xgboost binaria.R")
# 
# medias1<-cruzadalogistica(data=saheartbis,
#  vardep="chd",listconti=c("sbp", "tobacco", "ldl",
#   "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""), grupos=4,sinicio=1234,repe=5)
# 
#  medias1$modelo="Logística"
# 
# 
# medias2<-cruzadaavnnetbin(data=saheartbis,
#  vardep="chd",listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),grupos=4,sinicio=1234,repe=5,
#   size=c(5),decay=c(0.1),repeticiones=5,itera=200)
# 
#   medias2$modelo="avnnet"
# 
# 
#   medias3<-cruzadaarbolbin(data=saheartbis,
#  vardep="chd",listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),grupos=4,sinicio=1234,repe=5,
#   cp=c(0),minbucket =5)
# 
#   medias3$modelo="arbol"
# 
# 
#   medias4<-cruzadarfbin(data=saheartbis, vardep="chd",
#    listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,nodesize=10,
#   mtry=6,ntree=200,replace=TRUE)
# 
#   medias4$modelo="bagging"
# 
#     medias5<-cruzadarfbin(data=saheartbis, vardep="chd",
#    listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,nodesize=10,
#   mtry=3,ntree=200,replace=TRUE)
# 
#   medias5$modelo="rf"
# 
# 
# medias6<-cruzadagbmbin(data=saheartbis, vardep="chd",
#    listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,
# n.minobsinnode=10,shrinkage=0.008,n.trees=1000,interaction.depth=2)
# 
# medias6$modelo="gbm"
# 
# 
# medias7<-cruzadaxgbmbin(data=saheartbis, vardep="chd",
#    listconti=c("sbp", "tobacco",
#   "ldl", "adiposity",  "obesity", "famhist.Absent"),
#  listclass=c(""),
#   grupos=4,sinicio=1234,repe=5,
#    min_child_weight=10,eta=0.008,nrounds=1000,max_depth=2,
#   gamma=0,colsample_bytree=1,subsample=1)
# 
# medias7$modelo="xgbm"
# 
# union1<-rbind(medias1,medias2,medias3,medias4,medias5,medias6,medias7)
# 
# par(cex.axis=0.5)
# boxplot(data=union1,tasa~modelo,main="TASA FALLOS")
# boxplot(data=union1,auc~modelo,main="AUC")
# 


