library(ggplot2)

repito<-function(nodos, conjunto.1, vardep)
 {
  cstrength<-medic.data.final[,vardep]
 # Aquí se cambia la semilla de la partición
set.seed(12347)
sample <- sample.int(n = nrow(medic.data.final),
 size = floor(0.75*nrow(medic.data.final)), replace = F)

train <- medic.data.final[sample, ]
test  <- medic.data.final[-sample, ]

# Estandarizo train y test
# Los datos de test se estandarizan con las medias y d.típica de train

means <-apply(train[,conjunto.1],2,mean,na.rm=TRUE)
sds<-sapply(train[,conjunto.1],sd,na.rm=TRUE)

train2<-scale(train[,conjunto.1], center = means, scale = sds)
numerocont<-which(colnames(train)%in%conjunto.1)
train2<-cbind(train2,train[,-numerocont,drop=FALSE ])

test2<-scale(test[,conjunto.1], center = means, scale = sds)
numerocont<-which(colnames(test)%in%conjunto.1)
test2<-cbind(test2,test[,-numerocont,drop=FALSE ])


library(nnet)
set.seed(22342)

# Se construye la red con los datos train

red1<-nnet(data=train2,
           as.formula(paste0(vardep, "~", paste0(conjunto.1, collapse = "+"))),linout = FALSE,size=nodos,maxit=100)

summary(red1)

# Se calculan las predicciones sobre datos test

predi<-predict(red1,newdata=test2,type="class")

# Se calcula el error de predicción sobre datos test
MSEtestred <- sum((test2$target - predi)^2)/nrow(test2)

MSEtestred
 
# Error cometido por el modelo de regresión

reg1<-glm(data=train2,as.formula(paste0(vardep, "~", paste0(conjunto.1, collapse = "+"))), family = binomial())

summary(reg1)
 
# Para obtener el MSE y el RMSE

predi<-predict(reg1,newdata=test2,type="prob")

# Se calcula el error de predicción sobre datos test
MSEtestreg <- sum((test2$target - predi)^2)/nrow(test2)

MSEtestreg
 return(list(nodos,MSEtestred,MSEtestreg))
}

# Aquí creo un data.frame para ir guardando los datos para un gráfico

resul<-data.frame(c())
resulfin<-data.frame(c())

for (nodos in 3:35)
{
 repe<-repito(nodos, conjunto.1, vardep)
 cat(repe[[1]],"\n")
 print(repe[[2]])
 print(repe[[3]])

 nodos<-repe[[1]]
  
 resul<-as.data.frame(nodos)
 resul$red<-repe[[2]]
 resul$reg<-repe[[3]]
 resulfin<-rbind(resulfin,resul)
  }

library(reshape2)

meltdf <- melt(resulfin,id="nodos")

ggplot(meltdf,aes(x=nodos,
 y=value,colour=variable,group=variable)) + geom_line()+
 scale_x_continuous(breaks =seq(3,35, by=1))+
 theme(axis.text.x=element_text(size=7))
 
 
    

