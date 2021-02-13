source("./librerias/funcion steprepetido binaria.R")
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)
rm(tabla); rm(tabla.orden); rm(media); rm(desv.tipica); rm(factores); rm(mejor.lambda); rm(vector.mejor.lambda); 
rm(columnas.dummy); rm(salida.woe); rm(salida.woe.copia); rm(tablamis.numericas); rm(tablamis.factores); rm(col);
rm(numericas)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

# SELECCION DE VARIABLES
# Metodo 1. Stepwise AIC y BIC
columnas <- names(telco.data.final)[c(-37, -38)]
vardep <- c("target")
lista.variables.aic <- steprepetidobinaria(data=telco.data.final,
                          vardep=vardep,listconti=columnas,sinicio=12345,
                          sfinal=12445,porcen=0.8,criterio="AIC")
tabla.aic <- lista.variables.aic[[1]]

lista.variables.bic <- steprepetidobinaria(data=telco.data.final,
                                           vardep=vardep,listconti=columnas,sinicio=12345,
                                           sfinal=12445,porcen=0.8,criterio="BIC")
tabla.bic <- lista.variables.bic[[1]]

# Ya tenemos las variables, pero...
candidato.aic <- unlist(strsplit(tabla.aic[order(tabla.aic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))

# Opcion 2. Random Feature Elimination
vector.semillas <- vector(mode="list", length=6)
for(i in seq(1,5)) {
 vector.semillas[[i]] <- seq(12345, 12365)
}
vector.semillas[6] <- 12345

# Con regresion logistica
control.lr <- rfeControl(functions = lrFuncs, method = "cv", number = 5, repeats = 5, seeds = vector.semillas)
salida.rfe.lr <- rfe(telco.data.final[, c(columnas, "tenure.cont")], telco.data.final[, vardep], sizes = c(1:20), rfeControl = control.rf)
predictors(salida.rfe.rf)
plot(salida.rfe.rf, type=c("g", "o"))

# Con random forest
control.rf <- rfeControl(functions = rfFuncs, method = "cv", number = 5, repeats = 5, seeds = vector.semillas)
salida.rfe.rf <- rfe(telco.data.final[, c(columnas, "tenure.cont")], telco.data.final[, vardep], sizes = c(1:20), rfeControl = control.rf)
predictors(salida.rfe.rf)
plot(salida.rfe.rf, type=c("g", "o"))

# ************************************ 
# APLICANDO cruzadalogistica a los modelos candidatos 
# ************************************
medias.base<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("tenure.cont", "TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", "TechSupport.No"), listclass=c(""), grupos=5,sinicio=1234,repe=5)

medias.base.df <- data.frame(medias.base[1])
medias.base.df$modelo="RFE LR-RF TOP 5"

medias1<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                                                                             "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                                                                             "OnlineSecurity.No", "tenure.0.5", "TechSupport.No", "Dependents"), listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias1.df <- data.frame(medias1[1])
medias1.df$modelo="LOGISTICA AIC"

medias2<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                                                                             "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                                                                             "OnlineSecurity.No"), listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias2.df <- data.frame(medias2[1])
medias2.df$modelo="LOGISTICA BIC"

# Diferencia entre ambos modelos
setdiff(predictors(salida.rfe.rf), candidato.bic)
setdiff(candidato.bic, predictors(salida.rfe.rf))

medias3<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", "OnlineSecurity.No", "tenure.18", "tenure.0.5", 
                                                                             "TechSupport.No", "Contract.Month.to.month", "InternetService.DSL"), listclass=, c(""), grupos=5,sinicio=1234,repe=5)
medias3.df <- data.frame(medias3[1])
medias3.df$modelo="LOGISTICA RFE"

union1<-rbind(medias1.df,medias2.df,medias3.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

medias4<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                                                                             "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                                                                             "OnlineSecurity.No", "tenure.cont", "TechSupport.No", "Dependents"), listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias4.df <- data.frame(medias4[1])
medias4.df$modelo="LOGISTICA AIC CON tenure.cont"

medias5<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", "OnlineSecurity.No", "tenure.cont",
                                                                             "TechSupport.No", "Contract.Month.to.month"), listclass=, c(""), grupos=5,sinicio=1234,repe=5)
medias5.df <- data.frame(medias5[1])
medias5.df$modelo="LOGISTICA RFE CON tenure.cont"

union1<-rbind(medias.base.df, medias1.df,medias2.df,medias3.df, medias4.df, medias5.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Opcion 3. Grafico VCramer
# Calcula el V de Cramer
par(mai=c(3,1,1,1))
Vcramer<-function(v,target){
  greybox::cramer(telco.data.final[, v], telco.data.final[, target])$value
}

# Gr?fico con el V de cramer de todas las variables input para saber su importancia
graficoVcramer<-function(varsInd, varDep){
  vector.cramer <- c(sapply(varsInd, function(x) {Vcramer(x, varDep)}))
  barplot(sort(vector.cramer,decreasing =T),las=3,ylim=c(0,1), names.arg = varsInd, cex.names=0.8)
}
graficoVcramer(c("PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                 "StreamingMovies.No", "OnlineSecurity.No", "TechSupport.No"), vardep)


# ?Y si eliminamos OnlineSecurity.No?
medias6<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                                                                             "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check"), 
                                                                              listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias6.df <- data.frame(medias6[1])
medias6.df$modelo="LOGISTICA BIC SIN OnlineSecurity.No"

union1<-rbind(medias.base.df, medias1.df,medias2.df,medias3.df, medias4.df, medias5.df, medias6.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

for(modelo in c(medias.base, medias1,medias2,medias3, medias4, medias5, medias6)) {
  print(modelo$table)
}

# Al finalizar
stopCluster(cluster) 
registerDoSEQ()