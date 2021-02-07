source("./librerias/funcion steprepetido binaria.R")
library(parallel)
library(doParallel)
library(caret)
library(mlbench)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 

# SELECCION DE VARIABLES
# Metodo 1. Stepwise AIC y BIC
columnas <- names(datos.elecciones)[-1]
vardep <- c("Derecha")
lista.variables.aic <- steprepetidobinaria(data=datos.elecciones,
                          vardep=vardep,listconti=columnas,sinicio=12345,
                          sfinal=12445,porcen=0.8,criterio="AIC")
tabla.aic <- lista.variables.aic[[1]]

lista.variables.bic <- steprepetidobinaria(data=datos.elecciones,
                                           vardep=vardep,listconti=columnas,sinicio=12345,
                                           sfinal=12445,porcen=0.8,criterio="BIC")
tabla.bic <- lista.variables.bic[[1]]

# Opcion 2. Random Feature Elimination
set.seed(12345)
control <- rfeControl(functions = lrFuncs, method = "cv", number = 5, repeats = 20)
salida.rfe <- rfe(datos.elecciones[, columnas], datos.elecciones[, vardep], sizes = c(1:36), rfeControl = control)
predictors(salida.rfe)
plot(salida.rfe, type=c("g", "o"))

# Ya tenemos las variables, pero...
candidato.aic <- unlist(strsplit(tabla.aic[order(tabla.aic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic.1 <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic.2 <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][2,1], split = "+", fixed = TRUE))

# Se diferencian en variables como prop_missings, la temperatura de la estrella y su error (koi_steff) y koi_kepmag
Reduce(setdiff, c(candidato.aic, candidato.bic.1, candidato.bic.2))





# Al finalizar
stopCluster(cluster) 
registerDoSEQ()