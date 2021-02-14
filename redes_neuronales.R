# REDES NEURONALES
# 2 candidatos
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)

telco.data.final <- read.csv("telco_data_final.csv", row.names = "X")

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

variables.candidato.1 <- c("TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", 
                           "OnlineSecurity.No", "tenure.cont", "TechSupport.No", "Contract.Month.to.month")

variables.candidato.2 <- c("tenure.cont", "TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", 
                           "TechSupport.No")

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
                                       grupos=5,sinicio=1234,repe=5, size=5,decay=0.01,repeticiones=5,itera=200)

medias1.1.df <- data.frame(cvnnet.candidato.1.1[1])
medias1.1.df$modelo="NODOS: 5 - DECAY: 0.01"

# 2
cvnnet.candidato.1.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=5,decay=0.1,repeticiones=5,itera=200)

medias1.2.df <- data.frame(cvnnet.candidato.1.2[1])
medias1.2.df$modelo="NODOS: 5 - DECAY: 0.1"

# 3
cvnnet.candidato.1.3 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=5,decay=0.001,repeticiones=5,itera=200)

medias1.3.df <- data.frame(cvnnet.candidato.1.3[1])
medias1.3.df$modelo="NODOS: 5 - DECAY: 0.001"

# 4
cvnnet.candidato.1.4 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=10,decay=0.1,repeticiones=5,itera=200)


medias1.4.df <- data.frame(cvnnet.candidato.1.4[1])
medias1.4.df$modelo="NODOS: 10 - DECAY: 0.1"

union1<-rbind(medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

save.image("redes_neuronales.RData")

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
                                         grupos=5,sinicio=1234,repe=5, size=50,decay=0.1,repeticiones=5,itera=200)

medias2.1.df <- data.frame(cvnnet.candidato.2.1[1])
medias2.1.df$modelo="NODOS: 50 - DECAY: 0.1"

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

union2<-rbind(medias2.1.df,medias2.2.df,medias2.3.df,medias2.4.df,medias2.5.df)

save.image("redes_neuronales.RData")

par(cex.axis=0.5) 
boxplot(data=union2,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union2,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Al finalizar
stopCluster(cluster) 
registerDoSEQ()
