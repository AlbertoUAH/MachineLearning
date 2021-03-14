# BAGGING
# 2 candidatos
rm(list = setdiff(ls(), c("logistico.1", "logistico.2", "cvnnet.candidato.final.1",
                          "cvnnet.candidato.final.2", "medic.data.final",
                          "conjunto.1", "conjunto.2", "vardep")))

source ("./librerias/cruzada rf binaria.R")
library(randomForest)
library(parallel)
library(doParallel)
library(caret)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

# En primer lugar, tuneamos el numero de variables independientes del modelo
# En nuestro caso, 7 para el modelo 1 y 5 para el modelo 2
mtry.1 <- 7
mtry.2 <- 5

# Paso 1. Determinar el numero optimo de arboles para el modelo 1
# Parece que a partir de 3000 arboles se estabiliza la varianzq
set.seed(1234)
rfbis.1<-randomForest(factor(target)~complication_rsi+ccsComplicationRate..0.0.1.+ccsComplicationRate..0.2.more.+month.8+Age+bmi+moonphase+mortality_rsi+ccsMort30Rate..0.001.0.002.,
                    data=medic.data.final,
                    mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

# Y el modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~Age+complication_rsi+ccsComplicationRate..0.0.1.+mortality_rsi+bmi,
                    data=medic.data.final,
                    mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

plot(rfbis.2$err.rate[,1], col = 'red')
points(rfbis.1$err.rate[,1], col = 'blue')
legend("topright", legend = c("5 variables","7 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

# Hacemos zoom entre 0 y 2000
plot(rfbis.2$err.rate[c(0:2000),1], type = 'l', col = 'red')
lines(rfbis.1$err.rate[c(0:2000),1], col = 'blue')
legend("topright", legend = c("5 variables","7 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

# Aparentemente, el error se estabiliza con 300 arboles
plot(rfbis.2$err.rate[c(0:1000),1], type = 'l', col = 'red')
lines(rfbis.1$err.rate[c(0:1000),1], col = 'blue')
legend("topright", legend = c("5 variables","7 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

# Tunning modelos
n.tree <- 300
nodesizes.1 <- list(5, 10, 20, 30, 40, 50, 100, 150, 200)
# Sampsize maximo: (k-1) * n => 6835 / 5 = 1367 y se utilizan (4/5) * 6835 = 5468 obs.
# Conclusion: sampsize maximo: 5468 obs. (de forma aproximada)
sampsizes <- list(1, 500, 1000, 2000, 3000, 4000, 5000, 5400)


# MODELO 1
lista.rf.1 <- list()
for(x in apply(data.frame(expand.grid(nodesizes.1, sampsizes)),1,as.list)) {
  salida <- cruzadarfbin(data=medic.data.final, vardep=vardep,
                         listconti=conjunto.1,
                         listclass=c(""),
                         grupos=5,sinicio=1234,repe=5,nodesize=x$Var1,
                         mtry=mtry.1,ntree=n.tree, sampsize=x$Var2)
  cat(x$Var1, "+",  x$Var2 , "-> FINISHED\n")
  salida$modelo <- paste0(x$Var1, "+",  x$Var2)
  lista.rf.1 <- c(lista.rf.1, list(salida))
}

# MODELO 2
nodesizes.2 <- list(5, 10, 20, 30, 40, 50, 100, 150, 200)
lista.rf.2 <- list()
for(x in apply(data.frame(expand.grid(nodesizes.2, sampsizes)),1,as.list)) {
  salida <- cruzadarfbin(data=medic.data.final, vardep=vardep,
                         listconti=conjunto.2,
                         listclass=c(""),
                         grupos=5,sinicio=1234,repe=5,nodesize=x$Var1,
                         mtry=mtry.2,ntree=n.tree, sampsize=x$Var2)
  cat(x$Var1, "+",  x$Var2 , "-> FINISHED\n")
  salida$modelo <- paste0(x$Var1, "+",  x$Var2)
  lista.rf.2 <- c(lista.rf.2, list(salida))
}


# Al finalizar
stopCluster(cluster) 
registerDoSEQ()
