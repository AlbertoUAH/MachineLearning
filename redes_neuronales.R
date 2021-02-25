# REDES NEURONALES
# 2 candidatos
rm(list = setdiff(ls(), c("logistico.1", "logistico.2", "medic.data.final",
                          "conjunto.1", "conjunto.2", "vardep")))

setwd("/Users/alberto/UCM/Machine Learning/Practica ML/MachineLearning/")
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

# Primer modelo. Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 6835 / 30 ~ 228 parametros
# Si k = 9, entonces 11 * h + 1 = 228, es decir, 20-21 nodos

# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 6835 / 20 ~ 342 parametros
# Si k = 9, entonces 11 * h + 1 = 342, es decir, 30-31 nodos
size.candidato.1 <- c(5, 10, 15, 20, 25, 30, 35, 40, 45)
decay.candidato.1 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.1 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                          grupos=5,sinicio=1234,repe=5, size=size.candidato.1,decay=decay.candidato.1,repeticiones=5,itera=200)

# 1
cvnnet.candidato.1.1 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=30,decay=0.1,repeticiones=5,itera=200)

medias1.1.df <- data.frame(cvnnet.candidato.1.1[1])
medias1.1.df$modelo="NODOS: 30 - DECAY: 0.1"

# 2
cvnnet.candidato.1.2 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=35,decay=0.1,repeticiones=5,itera=200)

medias1.2.df <- data.frame(cvnnet.candidato.1.2[1])
medias1.2.df$modelo="NODOS: 35 - DECAY: 0.1"

# 3
cvnnet.candidato.1.3 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=40,decay=0.1,repeticiones=5,itera=200)

medias1.3.df <- data.frame(cvnnet.candidato.1.3[1])
medias1.3.df$modelo="NODOS: 40 - DECAY: 0.1"

# 4
cvnnet.candidato.1.4 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=45,decay=0.1,repeticiones=5,itera=200)


medias1.4.df <- data.frame(cvnnet.candidato.1.4[1])
medias1.4.df$modelo="NODOS: 45 - DECAY: 0.1"

# 5
cvnnet.candidato.1.5 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=25,decay=0.1,repeticiones=5,itera=200)


medias1.5.df <- data.frame(cvnnet.candidato.1.5[1])
medias1.5.df$modelo="NODOS: 25 - DECAY: 0.1"

# 6
cvnnet.candidato.1.6 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=20,decay=0.1,repeticiones=5,itera=200)


medias1.6.df <- data.frame(cvnnet.candidato.1.6[1])
medias1.6.df$modelo="NODOS: 20 - DECAY: 0.1"

# 7
cvnnet.candidato.1.7 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=15,decay=0.1,repeticiones=5,itera=200)


medias1.7.df <- data.frame(cvnnet.candidato.1.7[1])
medias1.7.df$modelo="NODOS: 15 - DECAY: 0.1"

# 8
cvnnet.candidato.1.8 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=10,decay=0.1,repeticiones=5,itera=200)


medias1.8.df <- data.frame(cvnnet.candidato.1.8[1])
medias1.8.df$modelo="NODOS: 10 - DECAY: 0.1"

# 9
cvnnet.candidato.1.9 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.001,repeticiones=5,itera=200)


medias1.9.df <- data.frame(cvnnet.candidato.1.9[1])
medias1.9.df$modelo="NODOS: 5 - DECAY: 0.001"

union1<-rbind(logistico.1,medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df,medias1.5.df,medias1.6.df,medias1.7.df,medias1.8.df,medias1.9.df)

save.image("redes_neuronales.RData")

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

rm(medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df,medias1.5.df,medias1.6.df,medias1.7.df,medias1.8.df,medias1.9.df)
rm(cvnnet.candidato.1.1, cvnnet.candidato.1.2, cvnnet.candidato.1.3, cvnnet.candidato.1.4, cvnnet.candidato.1.5, cvnnet.candidato.1.6, cvnnet.candidato.1.7,
   cvnnet.candidato.1.8, cvnnet.candidato.1.9)

# Candidatos: 20-25 nodos, lr = 0.1
# ¿Y si aumentamos el lr?
cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.1, listclass=c(""),
                 grupos=5,sinicio=1234,repe=5, size=c(20,25),decay=c(0.1, 0.2, 0.5),repeticiones=5,itera=200)

# No mejora...
#   size decay   bag  Accuracy     Kappa  AccuracySD    KappaSD
# 1   20   0.1 FALSE 0.8399415 0.6401366 0.008740163 0.01952005
# 2   20   0.2 FALSE 0.8296123 0.6183575 0.009507343 0.02120276
# 3   20   0.5 FALSE 0.7536796 0.4535413 0.010408810 0.02276280
# 4   25   0.1 FALSE 0.8435991 0.6480282 0.008844141 0.02194408
# 5   25   0.2 FALSE 0.8347037 0.6291966 0.010648039 0.02313318
# 6   25   0.5 FALSE 0.7566642 0.4600126 0.014553994 0.03144282

# Primer modelo. Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 6835 / 30 ~ 228 parametros
# Si k = 5, entonces 7 * h + 1 = 228, es decir, 32-33 nodos

# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 6835 / 20 ~ 342 parametros
# Si k = 5, entonces 7 * h + 1 = 342, es decir, 48-49 nodos
size.candidato.2 <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
decay.candidato.2 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.2 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.2,decay=decay.candidato.2,repeticiones=5,itera=200)

# 1
cvnnet.candidato.1.1.bis <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=50,decay=0.01,repeticiones=5,itera=200)

medias1.1.df.bis <- data.frame(cvnnet.candidato.1.1.bis[1])
medias1.1.df.bis$modelo="NODOS: 50 - DECAY: 0.01"

# 2
cvnnet.candidato.1.2.bis.2 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=55,decay=0.01,repeticiones=5,itera=200)

medias1.2.df.bis <- data.frame(cvnnet.candidato.1.2.bis.2[1])
medias1.2.df.bis$modelo="NODOS: 55 - DECAY: 0.01"

# 1
cvnnet.candidato.1.1 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=30,decay=0.01,repeticiones=5,itera=200)

medias1.1.df <- data.frame(cvnnet.candidato.1.1[1])
medias1.1.df$modelo="NODOS: 30 - DECAY: 0.01"

# 2
cvnnet.candidato.1.2 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=35,decay=0.01,repeticiones=5,itera=200)

medias1.2.df <- data.frame(cvnnet.candidato.1.2[1])
medias1.2.df$modelo="NODOS: 35 - DECAY: 0.01"

# 3
cvnnet.candidato.1.3 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=40,decay=0.01,repeticiones=5,itera=200)

medias1.3.df <- data.frame(cvnnet.candidato.1.3[1])
medias1.3.df$modelo="NODOS: 40 - DECAY: 0.01"

# 4
cvnnet.candidato.1.4 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=45,decay=0.01,repeticiones=5,itera=200)


medias1.4.df <- data.frame(cvnnet.candidato.1.4[1])
medias1.4.df$modelo="NODOS: 45 - DECAY: 0.01"

# 5
cvnnet.candidato.1.5 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=25,decay=0.001,repeticiones=5,itera=200)


medias1.5.df <- data.frame(cvnnet.candidato.1.5[1])
medias1.5.df$modelo="NODOS: 25 - DECAY: 0.001"

# 6
cvnnet.candidato.1.6 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=20,decay=0.001,repeticiones=5,itera=200)


medias1.6.df <- data.frame(cvnnet.candidato.1.6[1])
medias1.6.df$modelo="NODOS: 20 - DECAY: 0.001"

# 7
cvnnet.candidato.1.7 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=15,decay=0.01,repeticiones=5,itera=200)


medias1.7.df <- data.frame(cvnnet.candidato.1.7[1])
medias1.7.df$modelo="NODOS: 15 - DECAY: 0.01"

# 8
cvnnet.candidato.1.8 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=10,decay=0.01,repeticiones=5,itera=200)


medias1.8.df <- data.frame(cvnnet.candidato.1.8[1])
medias1.8.df$modelo="NODOS: 10 - DECAY: 0.01"

# 9
cvnnet.candidato.1.9 <- cruzadaavnnetbin(data=medic.data.final,vardep=vardep,listconti=conjunto.2, listclass=c(""),
                                         grupos=5,sinicio=1234,repe=5, size=5,decay=0.01,repeticiones=5,itera=200)


medias1.9.df <- data.frame(cvnnet.candidato.1.9[1])
medias1.9.df$modelo="NODOS: 5 - DECAY: 0.01"

union2<-rbind(logistico.1,medias1.1.df.bis,medias1.2.df.bis,medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df,medias1.5.df,medias1.6.df,medias1.7.df,medias1.8.df,medias1.9.df)

save.image("redes_neuronales.RData")

par(cex.axis=0.5) 
boxplot(data=union2,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union2,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

rm(medias1.1.df.bis,medias1.2.df.bis,medias1.1.df,medias1.2.df,medias1.3.df,medias1.4.df,medias1.5.df,medias1.6.df,medias1.7.df,medias1.8.df,medias1.9.df)
rm(cvnnet.candidato.1.1, cvnnet.candidato.1.2, cvnnet.candidato.1.3, cvnnet.candidato.1.4, cvnnet.candidato.1.5, cvnnet.candidato.1.6, cvnnet.candidato.1.7,
   cvnnet.candidato.1.8, cvnnet.candidato.1.9, cvnnet.candidato.1.1.bis, cvnnet.candidato.1.2.bis.2)

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
