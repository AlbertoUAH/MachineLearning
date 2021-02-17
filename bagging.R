# BAGGING
source ("./librerias/cruzada rf binaria.R")
library(randomForest)
library(parallel)
library(doParallel)
library(caret)
medias6.df$modelo <- "MODELO LOGISTICO 1"
medias.base.df$modelo <- "MODELO LOGISTICO 2"

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 


variables.candidato.1 <- c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                           "StreamingMovies.No", "PaperlessBilling", "Contract.One.year")

variables.candidato.2 <- c("OnlineSecurity.No", "TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", "TechSupport.No")

vardep <- "target"

telco.data.final <- read.csv("telco_data_final.csv", row.names = "X")

mtry.1 <- 7
mtry.2 <- 5

# Paso 1. Determinar el numero optimo de arboles para el modelo 1
# Parece que a partir de 3000 arboles se estabiliza la varianzq
set.seed(1234)
rfbis.1<-randomForest(factor(target)~Contract.Month.to.month+InternetService.Fiber.optic+TotalCharges+InternetService.DSL+StreamingMovies.No+PaperlessBilling+Contract.One.year,
                    data=telco.data.final,
                    mtry=mtry.1,ntree=5000,nodesize=10,replace=TRUE)

# Y el modelo 2
set.seed(1234)
rfbis.2<-randomForest(factor(target)~OnlineSecurity.No+TotalCharges+MonthlyCharges+InternetService.Fiber.optic+TechSupport.No,
                    data=telco.data.final,
                    mtry=mtry.2,ntree=5000,nodesize=10,replace=TRUE)

plot(rfbis.2$err.rate[,1], type = 'l', col = 'red')
lines(rfbis.1$err.rate[,1], col = 'blue')
legend("topright", legend = c("5 variables","7 variables") , 
       col = c('red', 'blue') , bty = "n", horiz = FALSE, 
       lty=1, cex = 0.75)

# Tunning modelos
n.tree <- 2000
nodesizes.1 <- list(5, 10, 20, 30, 40, 50)
# Sampsize maximo: (k-1) * n => 7043 / 5 = 1408 y se utilizan (4/5) * 7043 = 5634 obs.
# Conclusion: sampsize maximo: 5633 obs. (de forma aproximada)
sampsizes <- list(TRUE, 500, 1000, 2000, 3000, 4000, 5000)

lista.rf.1 <- list()
apply(as.data.frame(expand.grid(nodesizes.1, sampsizes)), 1, function(tunning) {
  param1 <- tunning$Var1; param2 <- tunning$Var2
  salida <- cruzadarfbin(data=telco.data.final, vardep=vardep,
                 listconti=variables.candidato.1,
                 listclass=c(""),
                 grupos=5,sinicio=1234,repe=5,nodesize=param1,
                 mtry=mtry.1,ntree=n.tree,replace=param2)
  cat("NODESIZE-", tunning$Var1, "-SAMPSIZE:", tunning$Var2, "-> FINISHED\n")
  salida$modelo <- paste0("NODESIZE-", tunning$Var1, "-SAMPSIZE:", tunning$Var2)
  lista.rf.1 <- c(lista.rf.1, list(salida))
})



# Al finalizar
stopCluster(cluster) 
registerDoSEQ()
