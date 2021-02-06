source("./librerias/funcion steprepetido binaria.R")
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 


# SELECCION DE VARIABLES
columnas <- names(datos.exoplanetas)[-1]
vardep <- c("is_exoplanet")
lista.variables.aic <- steprepetidobinaria(data=datos.exoplanetas,
                          vardep=vardep,listconti=columnas,sinicio=12345,
                          sfinal=12445,porcen=0.8,criterio="AIC")
tabla.aic <- lista.variables.aic[[1]]

lista.variables.bic <- steprepetidobinaria(data=datos.exoplanetas,
                                           vardep=vardep,listconti=columnas,sinicio=12345,
                                           sfinal=12445,porcen=0.8,criterio="BIC")
tabla.bic <- lista.variables.bic[[1]]

# Ya tenemos las variables, pero...
candidato.aic <- unlist(strsplit(tabla.aic[order(tabla.aic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic.1 <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
candidato.bic.2 <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][2,1], split = "+", fixed = TRUE))

# Se diferencian en variables como prop_missings, la temperatura de la estrella y su error (koi_steff) y koi_kepmag
Reduce(setdiff, c(candidato.aic, candidato.bic.1, candidato.bic.2))





# Al finalizar
stopCluster(cluster) 
registerDoSEQ()