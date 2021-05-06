
resultadosxgboost<-function(dataf=dataf,vardep=vardep,listconti,
                        min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
                        gamma=0,colsample_bytree=1,subsample=1,alpha=0,lambda=0,lambda_bias=0,
                        corte=0.5)
  
{
  library(caret)
  library(pROC)
  
  tasafallos<-function(x,y) {
    confu<-confusionMatrix(x,y)
    tasa<-confu[[3]][1]
    return(tasa)
  }
  auc<-function(x,y) {
    curvaroc<-roc(response=x,predictor=y)
    auc<-curvaroc$auca
    return(auc)
  }
  
  set.seed(1234)
  control<-trainControl(method = "cv",number=5,savePredictions = "all",classProbs=TRUE)
  
  # PREPARACIÓN DATOS
  
  
  tabla1<-as.data.frame(table(dataf[,vardep]))
  tabla1<-tabla1[order(tabla1$Freq),]
  minoritaria<-as.character(tabla1[1,c("Var1")])
  tabla1<-tabla1[order(-tabla1$Freq),]
  mayoritaria<-as.character(tabla1[1,c("Var1")])
  if (minoritaria==mayoritaria)
  {
    tabla1<-tabla1[order(tabla1$Freq),]
    mayoritaria<-as.character(tabla1[2,c("Var1")])
  }
  cosa<-as.data.frame(prop.table(table(dataf[[vardep]])))
  fremin<-100*round(min(cosa$Freq),2)
  totalobs=nrow(dataf)
  
  cosa<-as.data.frame(table(dataf[[vardep]]))
  totalmin<-round(min(cosa$Freq),2)
  
  dataf[vardep]<-ifelse(dataf[vardep]==minoritaria,"Yes","No")
  
  formu1<-as.formula(paste("factor(",target,")~",paste0(listconti, collapse = "+")))
  
  xgbmgrid <-expand.grid( min_child_weight=min_child_weight,
                          eta=eta,nrounds=nrounds,max_depth=max_depth,
                          gamma=gamma,colsample_bytree=colsample_bytree,subsample=subsample)

  gbm <- train(formu1,data=dataf,
               method="xgbTree",trControl=control,
               tuneGrid=xgbmgrid,verbose=FALSE,
               alpha=alpha,lambda=lambda,lambda_bias=lambda_bias)
  print("HOLA")
  preditest<-gbm$pred
  
  preditest$pred<-ifelse(preditest$Yes>corte,"Yes","No")
  preditest$pred<-as.factor(preditest$pred)
  
  tasa=1-tasafallos(preditest$pred,preditest$obs)
  auc=auc(preditest$obs,preditest$Yes)
  a<-as.data.frame(table(preditest$pred))
  nYes<-a[2,2]
  
  if (is.na(nYes)==T)
  {nYes=0}
  confu<-confusionMatrix(preditest$pred,preditest$obs)
  FP<-confu[[2]][2]
  FN<-confu[[2]][3]
  VP<-confu[[2]][4]
  VN<-confu[[2]][1]
  
  print("HOLA")
  
  sensitivity <- as.numeric(sensitivity(preditest$pred,preditest$obs,"Yes"))
  especificity <- as.numeric(specificity(preditest$pred,preditest$obs,"No"))
  
  return(list("FP" = FP,"FN" = FN,"VP" = VP,
              "VN" = VN,"AUC" = auc,"tasa" = tasa,
              "sensitividad" = as.numeric(sensitivity), 
              "especificidad" = as.numeric(especificity)))
}

# EJEMPLO

# v<-toydata(n=400,0.4,1,1,5,-5,0.0000)
# v[[1]]+theme(legend.position = "none")+xlab("")+ylab("")
# dataf<-v[[2]]
# vardep<-"clase"
# res<-resultadosgbm(dataf=dataf,vardep=vardep)
