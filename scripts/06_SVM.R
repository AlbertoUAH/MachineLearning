# ------------- SVM ---------------
# Objetivo: elaborar el mejor modelo de SVM de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           C, sigma y grado
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  
  source("./librerias/librerias_propias.R")
  source("./librerias/cruzada SVM binaria lineal.R")
  source("./librerias/cruzada SVM binaria polinomial.R")
  source("./librerias/cruzada SVM binaria RBF.R")
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

#--- SVM binaria-lineal
#    Modelo 1
C_binaria_1 <- expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))


svm_bin_1 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                           listconti = var_modelo1, listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,C=C_binaria_1)

#    Modelo 2
C_binaria_2 <- expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))


svm_bin_2 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                           listconti = var_modelo2, listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,C=C_binaria_2)


#--- SVM polinomial
#    Modelo 1
C_poly_1 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
degree_poly_1 <- c(2,3)
scale_poly_1  <- c(0.1, 0.5, 1, 2, 5)

svm_pol_1 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                               listconti = var_modelo1, listclass=c(""),
                               grupos=5,sinicio=1234,repe=5,C = C_poly_1,
                               degree = degree_poly_1, scale = scale_poly_1)

#    Modelo 2
C_poly_2 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
degree_poly_2 <- c(2,3)
scale_poly_2  <- c(0.1, 0.5, 1, 2, 5)

svm_pol_2 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                               listconti = var_modelo2, listclass=c(""),
                               grupos=5,sinicio=1234,repe=5,C = C_poly_2,
                               degree = degree_poly_2, scale = scale_poly_2)



