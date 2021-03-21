# ------------- Seleccion de variables ---------------
# Objetivo: realizar una seleccion previa de variables empleando
# -> metodos step aic + bic
# -> rfe
# -> Por ultimo, elegir las variables candidatas por medio de glm
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(ggplot2)       # Libreria grafica
  
  source("./librerias/librerias_propias.R")
  source("./librerias/funcion steprepetido binaria.R")
})

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final_completo.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"
vars   <- setdiff(names(surgical_dataset), target)

#-- Seleccion de variables con step aic y bic
lista.variables.aic <- steprepetidobinaria(data=surgical_dataset,
                                           vardep=target,listconti=vars,sinicio=1234,
                                           sfinal=1344,porcen=0.8,criterio="AIC")
tabla.aic <- lista.variables.aic[[1]]

lista.variables.bic <- steprepetidobinaria(data=surgical_dataset,
                                           vardep=target,listconti=vars,sinicio=1234,
                                           sfinal=1334,porcen=0.8,criterio="BIC")
tabla.bic <- lista.variables.bic[[1]]

# Ya tenemos las variables, pero...
# aic -> 14 parametros (con todas las observaciones 17)
candidato.aic <- unlist(strsplit(tabla.aic[order(tabla.aic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))
# bic -> 9 parametros (con todas las observaciones 10)
candidato.bic <- unlist(strsplit(tabla.bic[order(tabla.bic$Freq, decreasing = TRUE), ][1,1], split = "+", fixed = TRUE))

# Las variables comunes son las que aparecen en candidato.bic
# "mortality_rsi"     "ccsMort30Rate"     "bmi"               "month.8"           "asa_status.0"     
# "baseline_osteoart" "moonphase.0"       "Age"               "dow.0"
# Con todas las observaciones (bic), el modelo es el mismo salvo con el añadido de baseline_cancer
intersect(candidato.aic, candidato.bic) %in% candidato.bic

#--- Seleccion de variables con RFE
#--  Logistic Regression
# Creamos un vector de semillas aleatorias
vector.semillas <- vector(mode="list", length=6)
for(i in seq(1,5)) {
  vector.semillas[[i]] <- seq(1234, 1254)
}
vector.semillas[6] <- 1234

control.lr <- rfeControl(functions = lrFuncs, method = "cv", 
                         number = 5, repeats = 5, seeds = vector.semillas)

salida.rfe.lr <- rfe(surgical_dataset[, vars], surgical_dataset[, target], 
                     sizes = c(1:20), rfeControl = control.lr)
salida.rfe.lr

# Mejores variables RFE - LR (18 variables). Accuracy => 0.7796
# Con todas las observaciones 0.7880
salida.rfe.lr$optVariables
# Top 5: ccsComplicationRate, mortality_rsi, bmi, month.8, Age
# Las variables obtenidas en la interseccion anterior se situan practicamente entre las primeras del RFE - LR
ggplot(salida.rfe.lr) + ggtitle("Variable importance Logistic Regression RFE")
ggsave('./charts/01_feature_selection_RFE_LR_whole_dataset.png')

candidato.rfe.lr <- salida.rfe.lr$optVariables
candidato.rfe.lr.2 <- c("ccsMort30Rate", "mortality_rsi", "bmi")

#--- Random Forest
control.rf <- rfeControl(functions = rfFuncs, method = "cv", 
                         number = 5, repeats = 5, seeds = vector.semillas)

salida.rfe.rf <- rfe(surgical_dataset[, vars], surgical_dataset[, target], 
                     sizes = c(1:20), rfeControl = control.rf)

# Mejores variables RFE - RF (5 variables). Accuracy => 0.8902
# Con el dataset completo, con 4 variables se obtiene un 0.9040 (sin ahrq_ccs)
salida.rfe.rf$optVariables
# Top 5: Age, mortality_rsi, ccsMort30Rate, bmi, ahrq_ccs
ggplot(salida.rfe.rf) + ggtitle("Variable importance Random Forest RFE")
ggsave('./charts/01_feature_selection_RFE_RF_whole_dataset.png')

candidato.rfe.rf <- salida.rfe.rf$optVariables


# ************************************ 
# APLICANDO cruzadalogistica a los modelos candidatos 
# ************************************
# Candidatos a seleccion de variables: 
# step aic, bic, RFE LR, RFE RF
candidatos         <- list(candidato.aic, candidato.bic, candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos <- c("LOGISTICA AIC", "LOGISTICA BIC", "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")

union1 <- cruzada_logistica(surgical_dataset, target, candidatos, nombres_candidatos,
                            grupos = 5, repe = 5)
# Conclusion
# llaman la atencio los candidatos bic y rfe rf top 5
# Problemas: -> alta variabilidad en tasa fallos logistica bic
#            -> bajo valor AUC en rfe rf top 5 (aunque con RF se obtiene mejor accuracy)
#--- Revisamos logistica bic ¿Podriamos eliminar aquellas variables con menor poder predictivo?
candidato.bic.2 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0",           
                     "Age", "moonphase.0", "baseline_osteoart")
candidatos_2         <- list(candidato.aic, candidato.bic, candidato.bic.2, candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_2 <- c("LOGISTICA AIC", "LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" , "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")
union2 <- cruzada_logistica(surgical_dataset, target, candidatos_2, nombres_candidatos_2,
                            grupos = 5, repe = 5)

#--- Revisamos logistica aic ¿Podriamos eliminar aquellas variables con menor poder predictivo?
candidato.aic.2 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "baseline_charlson", 
                     "baseline_osteoart", "moonphase.0", "Age", "dow.0", "month.0", 
                     "ahrq_ccs")

candidatos_3         <- list(candidato.aic, candidato.aic.2, candidato.bic, candidato.bic.2, 
                             candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_3 <- c("LOGISTICA AIC", "LOGISTICA AIC (sin 2 variables)" ,"LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" , "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")
union3 <- cruzada_logistica(surgical_dataset, target, candidatos_3, nombres_candidatos_3,
                            grupos = 5, repe = 5)

#-- Si ejecutamos un random forest...
rf_modelo_bic <- train_rf_model(surgical_dataset, 
                                    as.formula(paste0("target~", paste0(candidato.bic.2, collapse = "+"))),
                                    mtry = c(3:8), ntree = 300, grupos = 5, repe = 5, nodesize = 10,
                                    seed = 1234)

# Importancia de las variables en un random forest
final<-rf_modelo_bic$finalModel
tabla<-as.data.frame(importance(final))
tabla<-tabla[order(-tabla$MeanDecreaseAccuracy),]
tabla

# ¿Pueden sobrar dow.0, moonphase.0?
barplot(tabla$MeanDecreaseAccuracy,names.arg=rownames(tabla))

candidato.bic.3 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8",           
                     "baseline_osteoart", "Age")
candidatos_4         <- list(candidato.aic, candidato.aic.2, candidato.bic, candidato.bic.2, candidato.bic.3, 
                             candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_4 <- c("LOGISTICA AIC", "LOGISTICA AIC (sin 2 variables)" ,"LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" ,
                          "LOGISTICA BIC (TOP 6)", "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")
union4 <- cruzada_logistica(surgical_dataset, target, candidatos_4, nombres_candidatos_4,
                            grupos = 5, repe = 5)

#-- De ahora en adelante probaremos con dos modelos candidatos
#   LOGISTICA BIC (TOP 6) -> seleccion1
#   "mortality_rsi" "ccsMort30Rate" "bmi" "month.8" "Age" "baseline_osteoart"

#   RFE RF TOP 5 -> seleccion2
#   "Age" "mortality_rsi" "ccsMort30Rate" "bmi" "ahrq_ccs"
#

surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

matriz_conf_1 <- matriz_confusion_predicciones(formula = paste0(target, "~" , 
                                                                paste0(candidato.bic.3, collapse = "+")),
                                                corte = 0.5,
                                                dataset = surgical_test_data
                                                )

matriz_conf_2 <- matriz_confusion_predicciones(formula = paste0(target, "~" , 
                                                                paste0(candidato.rfe.rf, collapse = "+")),
                                               corte = 0.5,
                                               dataset = surgical_test_data
)

#--- Predicciones
#     Modelo 1
#     Reference
#     Prediction   No  Yes
#     No         6215  379
#     Yes        1440  747

#      Modelo 2
#      Reference
#      Prediction   No  Yes
#      No         6250  344
#      Yes        1443  744
#

#---- Estadisticas
# Por tasa fallos --------------- auc
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/SeleccionVariables.RData")



