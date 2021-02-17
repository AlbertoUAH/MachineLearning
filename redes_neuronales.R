# REDES NEURONALES
# 2 candidatos
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)
medias6.df$modelo <- "MODELO LOGISTICO 1"
medias.base.df$modelo <- "MODELO LOGISTICO 2"

telco.data.final <- read.csv("telco_data_final.csv", row.names = "X")

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

variables.candidato.1 <- c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                           "StreamingMovies.No", "PaperlessBilling", "Contract.One.year")

variables.candidato.2 <- c("OnlineSecurity.No", "TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", "TechSupport.No")

vardep <- "target"

# Si quisieramos 30 observaciones por parametro
# h * (k + 1) + h + 1 = 7043 / 30 ~ 234.8
# Si k = 7, entonces 9 * h + 1 = 234.8, es decir, 26 nodos
# Si k = 5, entonces 7 * h + 1 = 234.8, es decir, 33 nodos
size.candidato.1 <- c(5, 10, 15, 20, 25, 30, 40)
decay.candidato.1 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                          grupos=5,sinicio=1234,repe=5, size=size.candidato.1,decay=decay.candidato.1,repeticiones=5,itera=200)

# 1
cvnnet.candidato.1.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=200)

medias1.1.df <- data.frame(cvnnet.candidato.1.1[1])
medias1.1.df$modelo="NODOS: 5 - DECAY: 0.1"

# 2
cvnnet.candidato.1.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=5,decay=0.001,repeticiones=5,itera=200)

medias1.2.df <- data.frame(cvnnet.candidato.1.2[1])
medias1.2.df$modelo="NODOS: 5 - DECAY: 0.001"

# 3
cvnnet.candidato.1.3 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=20,decay=0.1,repeticiones=5,itera=200)

medias1.3.df <- data.frame(cvnnet.candidato.1.3[1])
medias1.3.df$modelo="NODOS: 20 - DECAY: 0.1"

# 4
cvnnet.candidato.1.4 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=25,decay=0.1,repeticiones=5,itera=200)


medias1.4.df <- data.frame(cvnnet.candidato.1.4[1])
medias1.4.df$modelo="NODOS: 25 - DECAY: 0.1"

union1<-rbind(medias6.df,medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df)

save.image("redes_neuronales.RData")

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Si quisieramos 20 observaciones por parametro
# h * (k + 1) + h + 1 = 7043 / 20 ~ 352.2
# Si k = 7, entonces 9 * h + 1 = 352.2, es decir, 40 nodos
# Si k = 5, entonces 7 * h + 1 = 352.2, es decir, 50 nodos
size.candidato.2 <- c(5, 10, 15, 20, 25, 30, 40, 50)
decay.candidato.2 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.2,decay=decay.candidato.2,repeticiones=5,itera=200)

# 1
cvnnet.candidato.2.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=25,decay=0.1,repeticiones=5,itera=200)

medias2.1.df <- data.frame(cvnnet.candidato.2.1[1])
medias2.1.df$modelo="NODOS: 25 - DECAY: 0.1"

# 2
cvnnet.candidato.2.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=10,decay=0.1,repeticiones=5,itera=200)

medias2.2.df <- data.frame(cvnnet.candidato.2.2[1])
medias2.2.df$modelo="NODOS: 10 - DECAY: 0.1"

# 3
cvnnet.candidato.2.3 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.001,repeticiones=5,itera=200)

medias2.3.df <- data.frame(cvnnet.candidato.2.3[1])
medias2.3.df$modelo="NODOS: 5 - DECAY: 0.001"

# 4
cvnnet.candidato.2.4 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=200)

medias2.4.df <- data.frame(cvnnet.candidato.2.4[1])
medias2.4.df$modelo="NODOS: 5 - DECAY: 0.1"

# 5
cvnnet.candidato.2.5 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.01,repeticiones=5,itera=200)

medias2.5.df <- data.frame(cvnnet.candidato.2.5[1])
medias2.5.df$modelo="NODOS: 5 - DECAY: 0.01"

union2<-rbind(medias.base.df,medias2.1.df,medias2.2.df,medias2.3.df,medias2.4.df,medias2.5.df)

save.image("redes_neuronales.RData")

par(cex.axis=0.5) 
boxplot(data=union2,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union2,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# MODELOS DE RED CANDIDATOS
# Con 7 variables: 5 nodos y 0.1 decay
# Con 5 variables: 5 nodos y 0.1 decay
# Ahora, podemos modificar el maxit
lista.1 <- list(); lista.2 <- list()
for(maxit in c(50, 100, 150, 200, 300, 400, 500)) {
  cvnnet.candidato.final.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                           grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=maxit)
  
  medias.temp.df <- data.frame(cvnnet.candidato.final.1[1])
  medias.temp.df$modelo=paste("MAXIT: ", as.character(maxit))
  lista.1[[as.character(maxit)]] <- medias.temp.df
  
  cvnnet.candidato.final.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                               grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=maxit)
  
  medias.temp.df <- data.frame(cvnnet.candidato.final.2[1])
  medias.temp.df$modelo=paste("MAXIT: ", as.character(maxit))
  lista.2[[as.character(maxit)]] <- medias.temp.df
}
rm(medias.temp.df)
# Modelo 1
boxplot(data = rbind(plyr::ldply(lista.1, data.frame)[, -1], medias6.df), tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)
boxplot(data = rbind(plyr::ldply(lista.1, data.frame)[, -1], medias6.df), auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Modelo 2
boxplot(data = rbind(plyr::ldply(lista.2, data.frame)[, -1], medias.base.df), tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)
boxplot(data = rbind(plyr::ldply(lista.2, data.frame)[, -1], medias.base.df), auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Podemos bajar el numero maximo de iteraciones a 150
cvnnet.candidato.final.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.01,repeticiones=5,itera=100)
cvnnet.candidato.final.1.df <- data.frame(cvnnet.candidato.final.1[1])
cvnnet.candidato.final.1.df$modelo="AVNNET 1. NODOS: 5 - DECAY: 0.1 - MAXIT: 100"

cvnnet.candidato.final.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                             grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=200)
cvnnet.candidato.final.2.df <- data.frame(cvnnet.candidato.final.2[1])
cvnnet.candidato.final.2.df$modelo="AVNNET 2. NODOS: 5 - DECAY: 0.1 - MAXIT: 200"

medias6.df$modelo <- "MODELO LOGISTICO 1"
medias.base.df$modelo <- "MODELO LOGISTICO 2"
union.final <- rbind(medias6.df, medias.base.df, cvnnet.candidato.final.1.df, cvnnet.candidato.final.2.df)

union.final$formula <- c(rep("1", 5), rep("2", 5), rep("1", 5), rep("2", 5))
union.final$formula <- as.factor(union.final$formula)

colores <- ifelse(levels(union.final$formula)=="1" , rgb(0.1,0.1,0.7,0.5) , 
                  rgb(0.8,0.1,0.3,0.6))

par(cex.axis=0.5) 
boxplot(data=union.final,tasa~modelo,main="TASA FALLOS", lwd = 1, col = colores)
legend("bottomright", legend = c("FORMULA 1","FORMULA 2") , 
       col = c(rgb(0.1,0.1,0.7,0.5) , rgb(0.8,0.1,0.3,0.6)) , bty = "n",
       pch=20 , horiz = FALSE, inset = c(0.03, 0.1), cex = 0.75)

par(cex.axis=0.5) 
boxplot(data=union.final,auc~modelo,main="AUC", lwd = 1, col = colores)
legend("bottomright", legend = c("FORMULA 1","FORMULA 2") , 
       col = c(rgb(0.1,0.1,0.7,0.5) , rgb(0.8,0.1,0.3,0.6)) , bty = "n",
       pch=20 , horiz = FALSE, inset = c(0.03, 0.1), cex = 0.75)


for(modelo in c(cvnnet.candidato.final.1, cvnnet.candidato.final.2)) {
  print(modelo$table)
}

# Al finalizar
stopCluster(cluster) 
registerDoSEQ()
