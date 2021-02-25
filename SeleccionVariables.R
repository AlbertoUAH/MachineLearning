setwd("/Users/alberto/UCM/Machine Learning/Practica ML/MachineLearning/")
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
columnas <- names(medic.data.final)[c(-38)]
vardep <- c("target")
lista.variables.aic <- steprepetidobinaria(data=medic.data.final,
                          vardep=vardep,listconti=columnas,sinicio=12345,
                          sfinal=12445,porcen=0.8,criterio="AIC")
tabla.aic <- lista.variables.aic[[1]]

lista.variables.bic <- steprepetidobinaria(data=medic.data.final,
                                           vardep=vardep,listconti=columnas,sinicio=12345,
                                           sfinal=12445,porcen=0.8,criterio="BIC")
tabla.bic <- lista.variables.bic[[1]]

# Ya tenemos las variables, pero...
candidato.aic <- unlist(strsplit(tabla.aic[order(tabla.aic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))

# Las variables comunes son las que aparecen en candidato.bic
intersect(candidato.aic, candidato.bic)

# AIC - Age, ahrq_css.1, baseline_charlson.0, baseline_cvd, bmi, 
# ccsComplicationRate..0.0.1., ccsComplicationRate..0.2.more., 
# ccsMort30Rate..0.0.001., ccsMort30Rate..0.001.0.002., ccsMort30Rate..0.009.more.
# complication_rsi, dow.0, month.8, moonphase, mortality_rsi
table(unlist(strsplit(tabla.aic$modelo, split = "+", fixed = TRUE)))

# BIC - Age, baseline_charlson.0, baseline_cvd, bmi, ccsComplicationRate..0.0.1., 
# ccsComplicationRate..0.2.more., ccsMort30Rate..0.0.001.,  ccsMort30Rate..0.009.more.,
# complication_rsi, dow.0, month.8, moonphase, mortality_rsi
table(unlist(strsplit(tabla.bic$modelo, split = "+", fixed = TRUE)))

# 13 variables son comunes en ambos casos como las mas empleadas
# Age, baseline_charlson.0, baseline_cvd, bmi, ccsComplicationRate..0.0.1., 
# ccsComplicationRate..0.2.more., ccsMort30Rate..0.0.001., ccsMort30Rate..0.009.more., 
# complication_rsi, dow.0, month.8, moonphase, mortality_rsi

# Opcion 2. Recursive Feature Elimination
vector.semillas <- vector(mode="list", length=6)
for(i in seq(1,5)) {
 vector.semillas[[i]] <- seq(12345, 12365)
}
vector.semillas[6] <- 12345

# Con regresion logistica
control.lr <- rfeControl(functions = lrFuncs, method = "cv", number = 5, repeats = 5, seeds = vector.semillas)
salida.rfe.lr <- rfe(medic.data.final[, columnas], medic.data.final[, vardep], sizes = c(1:20), rfeControl = control.lr)
predictors(salida.rfe.lr)
plot(salida.rfe.lr, type=c("g", "o"))

# TOP 8 RFE LR - 0.7402
# ccsComplicationRate..0.0.1., complication_rsi, bmi, ccsComplicationRate..0.16.0.2., 
# ccsComplicationRate..0.1.0.16., month.8, Age, mortality_rsi
candidato.rfe.lr <- c("ccsComplicationRate..0.0.1.", "complication_rsi", "bmi", 
                      "ccsComplicationRate..0.16.0.2.", "ccsComplicationRate..0.1.0.16.", 
                      "month.8", "Age", "mortality_rsi")

# Con random forest
control.rf <- rfeControl(functions = rfFuncs, method = "cv", number = 5, repeats = 5, seeds = vector.semillas)
salida.rfe.rf <- rfe(medic.data.final[, columnas], medic.data.final[, vardep], sizes = c(1:20), rfeControl = control.rf)
predictors(salida.rfe.rf)
plot(salida.rfe.rf, type=c("g", "o"))

# TOP 5 RFE RF - 0.8616
# Age, complication_rsi, ccsComplicationRate..0.0.1., mortality_rsi, bmi
candidato.rfe.rf <- c("Age", "complication_rsi", "ccsComplicationRate..0.0.1.", 
                      "mortality_rsi", "bmi")

# ************************************ 
# APLICANDO cruzadalogistica a los modelos candidatos 
# ************************************
medias1<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=candidato.aic, listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias1.df <- data.frame(medias1[1])
medias1.df$modelo="LOGISTICA AIC"

medias2<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=candidato.bic, listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias2.df <- data.frame(medias2[1])
medias2.df$modelo="LOGISTICA BIC"

medias3<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=candidato.rfe.lr, listclass= c(""), grupos=5,sinicio=1234,repe=5)
medias3.df <- data.frame(medias3[1])
medias3.df$modelo="RFE LR TOP 8"

medias4<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=candidato.rfe.rf, listclass= c(""), grupos=5,sinicio=1234,repe=5)
medias4.df <- data.frame(medias4[1])
medias4.df$modelo="RFE RF TOP 5"

union1<-rbind(medias1.df,medias2.df,medias3.df, medias4.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

# Opcion 3. Grafico VCramer
# Calcula el V de Cramer
par(mai=c(3,1,1,1))
Vcramer<-function(v,target){
  greybox::cramer(medic.data.final[, v], medic.data.final[, target])$value
}

# Gr?fico con el V de cramer de todas las variables input para saber su importancia
graficoVcramer<-function(varsInd, varDep){
  vector.cramer <- c(sapply(varsInd, function(x) {Vcramer(x, varDep)}))
  barplot(sort(vector.cramer,decreasing =T),las=3,ylim=c(0,1), names.arg = varsInd, cex.names=0.8)
}
graficoVcramer(candidato.bic, vardep)

# ?Y si eliminamos OnlineSecurity.No?
medias5<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=setdiff(candidato.bic, c("ccsMort30Rate..0.009.more.", "baseline_charlson.0", "ahrq_ccs.2.10")), 
                                                                listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias5.df <- data.frame(medias5[1])
medias5.df$modelo="LOGISTICA BIC SIN 3 ULTIMAS VARIABLES"

union1<-rbind(medias1.df,medias2.df,medias3.df, medias4.df, medias5.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

medias6<-cruzadalogistica(data=medic.data.final, vardep="target",listconti=setdiff(candidato.bic, c("ccsMort30Rate..0.009.more.", "baseline_charlson.0", "ahrq_ccs.2.10", "ccsMort30Rate..0.0.001.", "dow.0", "ccsMort30Rate..0.0.001.0.002.")), 
                          listclass=c(""), grupos=5,sinicio=1234,repe=5)
medias6.df <- data.frame(medias6[1])
medias6.df$modelo="LOGISTICA BIC SIN 6 ULTIMAS VARIABLES"

union1<-rbind(medias1.df,medias2.df,medias3.df, medias4.df, medias5.df, medias6.df)

par(cex.axis=0.5) 
boxplot(data=union1,tasa~modelo,main="TASA FALLOS", col = "#F28773", lwd = 1)

par(cex.axis=0.5) 
boxplot(data=union1,auc~modelo,main="AUC", col = "#F28773", lwd = 1)

setdiff(setdiff(candidato.bic, c("ccsMort30Rate..0.009.more.", "baseline_charlson.0", "ahrq_ccs.2.10", "ccsMort30Rate..0.0.001.", "dow.0", "ccsMort30Rate..0.0.001.0.002.")),
                candidato.rfe.lr)

setdiff(candidato.rfe.lr,
        setdiff(candidato.bic, c("ccsMort30Rate..0.009.more.", "baseline_charlson.0", "ahrq_ccs.2.10", "ccsMort30Rate..0.0.001.", "dow.0", "ccsMort30Rate..0.0.001.0.002.")))

for(modelo in c(medias1,medias2,medias3, medias4, medias5, medias6)) {
  print(modelo$table)
}

# Elegimos el modelo BIC dado que segun la V de Cramer sus variables parecen ser mas significativas
graficoVcramer(c("ccsComplicationRate..0.2.more.", "moonphase", "ccsMort30Rate..0.001.0.002.",
                 "ccsComplicationRate..0.16.0.2.", "ccsComplicationRate..0.1.0.16."), vardep)

# Una vez obtenido el modelo final, podremos comprobar su accuracy con el conjunto de variables obtenido en RFE-LR (con 8 variables)

# Conjuntos de variables finales
conjunto.1 <- setdiff(candidato.bic, c("ccsMort30Rate..0.009.more.", "baseline_charlson.0", "ahrq_ccs.2.10", 
                                       "ccsMort30Rate..0.0.001.", "dow.0", "ccsMort30Rate..0.0.001.0.002."))
conjunto.2 <- candidato.rfe.rf

# Coinciden en complication_rsi, ccsComplicationRate..0.0.1., Age, bmi y mortality_rsi
intersect(conjunto.1, conjunto.2)

# ¿Influira realmente ccsComplicationRate..0.2.more., month.8, moonphase y ccsMort30Rate..0.001.0.002.?
setdiff(conjunto.1, conjunto.2)

logistico.1 <- medias6.df
logistico.1$modelo <- "LOG. 1"
logistico.2 <- medias4.df
logistico.2$modelo <- "LOG. 2"

# Al finalizar
stopCluster(cluster)
registerDoSEQ()

save.image("seleccion_variables.RData")