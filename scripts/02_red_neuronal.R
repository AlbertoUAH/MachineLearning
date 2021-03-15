# ------------- Redes Neuronales ---------------
# Objetivo: elaborar el mejor modelo de red neuronal de acuerdo
#           a los valores de prediccion obtenidos tras variar el
#           numero de nodos y ratio de aprendizaje
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  
  source("./librerias/librerias_propias.R")
})

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"

#--- Variables de los modelos candidatos
#--  Modelo 1
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0", 
                 "Age", "moonphase.0", "baseline_osteoart")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

#---- Comenzamos con el modelo 1
#     Nota: por el momento: itera = 200
# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 20 ~ 292 parametros
# Si k = 8, entonces 10 * h + 1 = 292 => 29.1, es decir, 29-30 nodos

# Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 30 ~ 195 parametros
# Si k = 8, entonces 10 * h + 1 = 195 => 19.4, es decir, 19-20 nodos
size.candidato.1 <- c(5, 10, 15, 20, 25, 30, 35)
decay.candidato.1 <- c(0.1, 0.01, 0.001)

cvnnet.candidato.1 <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                       listconti=var_modelo1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.1,
                                       decay=decay.candidato.1,repeticiones=5,itera=200)


#---- Continuamos con el modelo 2
# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 20 ~ 292 parametros
# Si k = 5, entonces 7 * h + 1 = 292 => 41.57, es decir, 41-42 nodos

# Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 30 ~ 195 parametros
# Si k = 5, entonces 7 * h + 1 = 195 => 27.71, es decir, 27-28 nodos
size.candidato.2 <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
decay.candidato.2 <- c(0.1, 0.01, 0.001)

cvnnet.candidato.2 <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                       listconti=var_modelo2, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.2,
                                       decay=decay.candidato.2,repeticiones=5,itera=200)


#---- Guardamos el fichero RData
save.image(file = "./rdata/RedesNeuronales.RData")










