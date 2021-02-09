source("./librerias/funcion steprepetido binaria.R")
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)
library(mlbench)
library(ggplot2)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

# SELECCION DE VARIABLES
# Metodo 1. Stepwise AIC y BIC
columnas <- names(telco.data.final)[-38]
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
# Con regresion logistica
set.seed(12345)
control.lr <- rfeControl(functions = lrFuncs, method = "cv", number = 5, repeats = 20)
salida.rfe.lr <- rfe(telco.data.final[, columnas], telco.data.final[, vardep], sizes = c(1:20), rfeControl = control.lr)
predictors(salida.rfe.lr)
plot(salida.rfe.lr, type=c("g", "o"))

# Y con random forest
set.seed(12345)
control.rf <- rfeControl(functions = rfFuncs, method = "cv", number = 5, repeats = 20)
salida.rfe.rf <- rfe(telco.data.final[, columnas], telco.data.final[, vardep], sizes = c(1:20), rfeControl = control.rf)
predictors(salida.rfe.rf)
plot(salida.rfe.rf, type=c("g", "o"))

# Opcion 3. Grafico VCramer
# Calcula el V de Cramer
par(mai=c(3,1,1,1))
Vcramer<-function(v,target){
  greybox::cramer(telco.data.final[, v], telco.data.final[, target])$value
}

# Gráfico con el V de cramer de todas las variables input para saber su importancia
graficoVcramer<-function(varsInd, varDep){
  vector.cramer <- c(sapply(varsInd, function(x) {Vcramer(x, varDep)}))
  barplot(sort(vector.cramer,decreasing =T),las=3,ylim=c(0,1), names.arg = varsInd, cex.names=0.8)
}
graficoVcramer(columnas, vardep)

# Se diferencian en variables como prop_missings, la temperatura de la estrella y su error (koi_steff) y koi_kepmag
setdiff(candidato.aic, candidato.bic)

# ************************************ 
# APLICANDO cruzadalogistica a los modelos candidatos 
# ************************************

medias1<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                          "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                          "OnlineSecurity.No", "tenure.0.5", "TechSupport.No", "Dependents"), listclass=c(""), grupos=5,sinicio=1234,repe=15)
medias1.df <- data.frame(medias1[1])
medias1.df$modelo="Logística1"

medias2<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("Contract.Month.to.month", "InternetService.Fiber.optic", "TotalCharges", "InternetService.DSL",
                          "StreamingMovies.No", "PaperlessBilling", "Contract.One.year", "StreamingTV.No", "PaymentMethod.Electronic.check",
                          "OnlineSecurity.No"), listclass=c(""), grupos=5,sinicio=1234,repe=15)
medias2.df <- data.frame(medias2[1])
medias2.df$modelo="Logística2"

medias3<-cruzadalogistica(data=telco.data.final, vardep="target",listconti=c("TotalCharges", "MonthlyCharges", "tenure.0.5", "InternetService.Fiber.optic", "OnlineSecurity.No"), 
                          listclass=, c(""), grupos=5,sinicio=1234,repe=15)
medias3.df <- data.frame(medias3[1])
medias3.df$modelo="Logística3"

union1<-rbind(medias1.df,medias2.df,medias3.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS")

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC")




# Al finalizar
stopCluster(cluster) 
registerDoSEQ()