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
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Por el momento, eliminamos ccsComplicationRate y complication_rsi
ccsComplicationRate <- surgical_dataset$ccsComplicationRate
complication_rsi    <- surgical_dataset$complication_rsi

surgical_dataset$complication_rsi <- NULL; surgical_dataset$ccsComplicationRate <- NULL

# Separamos variable objetivo del resto
target <- "target"
vars   <- setdiff(names(surgical_dataset), target)

#-- Seleccion de variables con step aic y bic
lista.variables.aic <- steprepetidobinaria(data=surgical_dataset,
                                           vardep=target,listconti=vars,sinicio=1234,
                                           sfinal=1334,porcen=0.8,criterio="AIC")
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
# Por tanto, ambos modelos se diferencian por 5 parametros (propios del candidato aic)
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

# Sin embargo, con tan solo 3 variables el modelo alcanza un accuracy de 0.77 en logistica
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

# Con un Random Forest, dadas las relaciones no lineales entre las variables
# obtenemos un modelo con mayor precision, aunque sean con 2 variables adicionales
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
# llaman la atencion los candidatos bic y rfe rf top 5
# RFE TOP 5 presenta un menor valor AUC, aunque el random forest nos ha dado indicios de la
# no linealidad de los datos, mejorando claramente en modelos no lineales como Random Forests
# Por otro lado, en relacion con el modelo stepwise AIC, el hecho de disponer de 14 variables
# adicionales, no ha mejorado relativamente el modelo frente a un modelo BIC con 9  variables

# Del mismo modo sucede con el modelo LR TOP 18, donde con 18 variables la mejora con respecto
# a BIC es de apenas unas milesimas

#--- Revisamos logistica bic ¿Podriamos eliminar aquellas variables con menor poder predictivo? Visto en IV.pdf
candidato.bic.2 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0",           
                     "Age", "moonphase.0", "baseline_osteoart")
candidatos_2         <- list(candidato.aic, candidato.bic, candidato.bic.2, candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_2 <- c("LOGISTICA AIC", "LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" , "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")
union2 <- cruzada_logistica(surgical_dataset, target, candidatos_2, nombres_candidatos_2,
                            grupos = 5, repe = 5)

# No baja demasiado, pero aunque la tasa de error se mantenga por encima, parece estabilizarse

#--- Revisamos logistica aic ¿Podriamos eliminar aquellas variables con menor poder predictivo?
candidato.aic.2 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "baseline_charlson", 
                     "baseline_osteoart", "moonphase.0", "Age", "dow.0", "month.0", 
                     "ahrq_ccs")

candidatos_3         <- list(candidato.aic, candidato.aic.2, candidato.bic, candidato.bic.2, 
                             candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_3 <- c("LOGISTICA AIC", "LOGISTICA AIC (sin 2 variables)" ,"LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" , "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5")
union3 <- cruzada_logistica(surgical_dataset, target, candidatos_3, nombres_candidatos_3,
                            grupos = 5, repe = 5)

# Incluso dejando el modelo con 11 variables, el valor AUC es practicamente identico, aunque la varianza
# del modelo aumenta

#-- Por otro lado, ¿Que ocurriria si ejecutamos un modelo random forest inicial de 1000 arboles con
#   las variables del candidato bic?
rf_modelo_bic <- train_rf_model(surgical_dataset, 
                                    as.formula(paste0("target~", paste0(candidato.bic.2, collapse = "+"))),
                                    mtry = c(3:8), ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                    seed = 1234)

rf_modelo_aic_2 <- train_rf_model(surgical_dataset, 
                                as.formula(paste0("target~", paste0(candidato.aic.2, collapse = "+"))),
                                mtry = c(3:11), ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                seed = 1234)

rf_modelo_rfe    <- train_rf_model(surgical_dataset, 
                                  as.formula(paste0("target~", paste0(candidato.rfe.rf, collapse = "+"))),
                                  mtry = c(3:5), ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                  seed = 1234)

# Importancia de las variables en un random forest
# ¿Pueden sobrar dow.0, moonphase.0? y baseline osteoart?
show_vars_importance(rf_modelo_bic, "Importancia variables Random Forest (modelo BIC)")

show_vars_importance(rf_modelo_aic_2, "Importancia variables Random Forest (modelo AIC)")

show_vars_importance(rf_modelo_rfe, "Importancia variables Random Forest (modelo RFE RF TOP 5)")

candidato.bic.3 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
candidato.bic.4 <- c("mortality_rsi", "bmi", "month.8", "Age")
candidato.aic.3 <- c("mortality_rsi", "ahrq_ccs", "bmi", "month.8", "Age")
candidato.rfe.2 <- c("Age", "mortality_rsi", "bmi", "ahrq_ccs")

candidatos_4         <- list(candidato.aic, candidato.aic.2, candidato.aic.3, candidato.bic, candidato.bic.2, candidato.bic.3, 
                             candidato.bic.4,candidato.rfe.lr, candidato.rfe.lr.2, candidato.rfe.rf, candidato.rfe.2)
nombres_candidatos_4 <- c("LOGISTICA AIC", "LOGISTICA AIC (sin 2 variables)" , "LOGISTICA AIC (TOP 5)" ,"LOGISTICA BIC", "LOGISTICA BIC (sin asa.status)" ,
                          "LOGISTICA BIC (TOP 5)", "LOGISTICA BIC (TOP 4)", "RFE LR TOP 18", "RFE LR TOP 3", "RFE RF TOP 5", "RFE RF TOP 4")
union4 <- cruzada_logistica(surgical_dataset, target, candidatos_4, nombres_candidatos_4,
                            grupos = 5, repe = 5)

#-- De ahora en adelante probaremos con dos modelos candidatos
#   LOGISTICA BIC (TOP 5) -> seleccion1
#   "Age" "mortality_rsi" "ccsMort30Rate" "bmi" "month.8" 

#   RFE RF TOP 5 -> seleccion2
#   "Age" "mortality_rsi" "ccsMort30Rate" "bmi" "ahrq_ccs"
#

#-- Previamente, dejamos a un lado variables como ccsComplicationRate y complication_rsi
#   ¿Como se ve afectado el modelo si incluimos ambas variables?
surgical_dataset$ccsComplicationRate <- ccsComplicationRate
surgical_dataset$complication_rsi    <- complication_rsi

candidatos_final_aux         <- list(c(candidato.bic.3, "ccsComplicationRate", "complication_rsi"), candidato.bic.3,
                                     c(candidato.rfe.rf, "ccsComplicationRate", "complication_rsi"), candidato.rfe.rf)

nombres_candidatos_final_aux <- c("LOGISTICA BIC + comp.", "LOGISTICA BIC (TOP 5)", "RFE RF + comp.", "RFE RF TOP 5")
union_final_aux <- cruzada_logistica(surgical_dataset, target, candidatos_final_aux, nombres_candidatos_final_aux,
                                 grupos = 5, repe = 5)
rm(candidatos_final_aux)
rm(nombres_candidatos_final_aux)

rf_1 <- train_rf_model(surgical_dataset, 
                                as.formula(paste0("target~", paste0(candidato.bic.3, collapse = "+"))),
                                mtry = 5, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                seed = 1234)

rf_2 <- train_rf_model(surgical_dataset, 
                                  as.formula(paste0("target~", paste0(candidato.rfe.rf, collapse = "+"))),
                                  mtry = 5, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                  seed = 1234)

rf_3 <- train_rf_model(surgical_dataset, 
                                as.formula(paste0("target~", paste0(c("mortality_rsi", "bmi", "ccsComplicationRate", "complication_rsi"), collapse = "+"))),
                                mtry = 4, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                seed = 1234)

rf_4 <- train_rf_model(surgical_dataset, 
                       as.formula(paste0("target~", paste0(candidato.rfe.lr.2, collapse = "+"))),
                       mtry = 3, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                       seed = 1234)

# Incluso con las variables "complication" un modelo RandomForest empeora, por lo que me decanto por descartar dichas variables
surgical_dataset$ccsComplicationRate <- NULL; surgical_dataset$complication_rsi <- NULL;

write.csv(surgical_dataset, "data/surgical_dataset_final.csv", row.names = FALSE)

#-- Nota: de cara a las comparaciones con el resto de modelos, aumentamos el numero de repeticiones a 10
candidatos_final         <- list(candidato.bic.3, candidato.rfe.rf, candidato.rfe.lr.2)
nombres_candidatos_final <- c("LOGISTICA BIC (TOP 5)", "RFE RF TOP 5", "RFE LR TOP 3")
union_final <- cruzada_logistica(surgical_dataset, target, candidatos_final, nombres_candidatos_final,
                            grupos = 5, repe = 10)
rm(candidatos_final)
rm(nombres_candidatos_final)

union_10_rep <- rbind(union4[union4$modelo %in% c("LOGISTICA BIC (TOP 5)", "RFE RF TOP 5", "RFE LR TOP 3"), ], union_final)
union_10_rep$rep <- c(rep("5", 15), rep("10", 30))

# Tasa de fallos
ggplot(union_10_rep, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/01_boxplot_log_modelo1_error_10rep.jpeg')

# AUC
ggplot(union_10_rep, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/01_boxplot_log_modelo1_auc_10rep.jpeg')

surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
surgical_test_data$ccsComplicationRate <- NULL; surgical_test_data$complication_rsi <- NULL
write.csv(surgical_test_data, "./data/surgical_test_data.csv", row.names = FALSE)

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



