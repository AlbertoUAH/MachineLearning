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
surgical_dataset$baseline_dementia <- NULL; surgical_dataset$mort30 <- NULL

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
# Top 5: "ccsMort30Rate", "mortality_rsi", "bmi", "month.8", "Age" 
# Las variables obtenidas en la interseccion anterior se situan practicamente entre las primeras del RFE - LR
colors <- c("Dataset original" = "red", "Subconjunto (40 %)" = "darkgreen")

ggplot(NULL, aes(x = Variables, y = Accuracy)) +
  geom_line(data = salida.rfe.lr$results, aes(color = "Dataset original")) +
  geom_point(data = salida.rfe.lr$results) +
  geom_line(data = salida.rfe.lr1$results, aes(color = "Subconjunto (40 %)")) +
  geom_point(data = salida.rfe.lr1$results) +
  ggtitle("Variable importance Logistic Regression RFE") +
  scale_color_manual(values = colors) + 
  labs(color='Dataset')+
  theme(text = element_text(size=14, face = "bold"))
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
ggplot(NULL, aes(x = Variables, y = Accuracy)) +
  geom_line(data = salida.rfe.rf$results, aes(color = "Dataset original")) +
  geom_point(data = salida.rfe.rf$results) +
  geom_line(data = salida.rfe.rf1$results, aes(color = "Subconjunto (40 %)")) +
  geom_point(data = salida.rfe.rf1$results) +
  ggtitle("Variable importance Random Forest RFE") +
  scale_color_manual(values = colors) + 
  labs(color='Dataset')  +
  theme(text = element_text(size=14, face = "bold"))
ggsave('./charts/01_feature_selection_RFE_RF_whole_dataset.png')

# Con un Random Forest, dadas las relaciones no lineales entre las variables
# obtenemos un modelo con mayor precision, aunque sean con 2 variables adicionales
candidato.rfe.rf <- salida.rfe.rf$optVariables


# ************************************ 
# APLICANDO cruzadalogistica a los modelos candidatos 
# ************************************
# Candidatos a seleccion de variables: 
# step aic, bic, RFE LR, RFE RF
candidatos         <- list(candidato.aic, candidato.bic, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos <- c("LOGISTICA AIC", "LOGISTICA BIC", "RFE LR TOP 3", "RFE RF")

union1 <- cruzada_logistica(surgical_dataset, target, candidatos, nombres_candidatos,
                            grupos = 5, repe = 5)

union_final <- rbind(union1, union_subset)

union_final$Dataset <- c(rep("Dataset original", 20), rep("Subconjunto (40 %)", 20))
t <- ggplot(union_final, aes(x = modelo, y = tasa, color = Dataset)) +
  geom_boxplot() +
  ggtitle("Comparacion (tasa fallos)") +
  scale_color_manual(values = colors) + 
  labs(color='Dataset')  +
  theme(text = element_text(size=12, face = "bold"))

a <- ggplot(union_final, aes(x = modelo, y = auc, color = Dataset)) +
  geom_boxplot() +
  ggtitle("Comparacion (AUC)") +
  scale_color_manual(values = colors) + 
  labs(color='Dataset')  +
  theme(text = element_text(size=12, face = "bold"))

ggpubr::ggarrange(t, a, common.legend = TRUE)
ggsave('./charts/01_feature_selection_primera_comparacion.png')

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
candidato.aic.2 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", 
                     "dow.0", "Age", "moonphase.0", "month.0", "baseline_osteoart", 
                     "baseline_charlson", "ahrq_ccs")

candidatos_3         <- list(candidato.aic, candidato.aic.2, candidato.bic, candidato.bic.2, candidato.rfe.lr.2, candidato.rfe.rf)
nombres_candidatos_3 <- c("LOGISTICA AIC", "LOGISTICA AIC (11 vars)" ,"LOGISTICA BIC", "LOGISTICA BIC (8 vars)" , "RFE LR TOP 3", "RFE RF TOP 5")
union3 <- cruzada_logistica(surgical_dataset, target, candidatos_3, nombres_candidatos_3,
                            grupos = 5, repe = 5)

union3$modelo <- with(union3, reorder(modelo,tasa, mean))
t <- ggplot(union3, aes(x = modelo, y = tasa)) +
  geom_boxplot() +
  ggtitle("Comparacion (tasa fallos)") +
  labs(color='Dataset')  +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))

union3$modelo <- with(union3, reorder(modelo,auc, mean))
a <- ggplot(union3, aes(x = modelo, y = auc)) +
  geom_boxplot() +
  ggtitle("Comparacion (AUC)") +
  labs(color='Dataset')  +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))

ggpubr::ggarrange(t, a, common.legend = TRUE)
ggsave('./charts/01_feature_selection_segunda_comparacion.png')



# Incluso dejando el modelo con 11 variables, el valor AUC es practicamente identico, aunque la varianza
# del modelo aumenta
rf_modelo_bic <- train_rf_model(surgical_dataset, 
                                as.formula(paste0("target~", paste0(candidato.bic.2, collapse = "+"))),
                                mtry = 5, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                seed = 1234)

rf_modelo_aic   <- train_rf_model(surgical_dataset, 
                                 as.formula(paste0("target~", paste0(candidato.aic.2, collapse = "+"))),
                                 mtry = 5, ntree = 1000, grupos = 5, repe = 5, nodesize = 10,
                                 seed = 1234)


# Importancia de las variables en un random forest
# ¿Pueden sobrar dow.0, moonphase.0? y baseline osteoart?
imp1 <- show_vars_importance(rf_modelo_aic, "Importancia variables (AIC)")
imp2 <- show_vars_importance(rf_modelo_bic, "Importancia variables (BIC)")
ggpubr::ggarrange(imp1, imp2, common.legend = TRUE)
ggsave('./charts/01_feature_selection_comparacion_random_forest.png')

top4 <- c("Age", "mortality_rsi", "bmi", "ccsMort30Rate")
candidato.aic.top5 <- c("Age", "mortality_rsi", "bmi", "ccsMort30Rate", "ahrq_ccs")
candidato.bic.4 <- c("Age", "mortality_rsi", "bmi", "ccsMort30Rate", "baseline_osteoart")
candidato.bic.5 <- c("Age", "mortality_rsi", "bmi", "ccsMort30Rate", "month.8")

candidatos_4         <- list(candidato.aic, candidato.bic, top4, candidato.rfe.rf, 
                             candidato.bic.4, candidato.bic.5)
nombres_candidatos_4 <- c("AIC" , "BIC" , "AIC-BIC-TOP 4", "RFE RF TOP 5 (AIC TOP 5)",
                          "BIC (TOP 5 - baseline_osteoart)", "BIC (TOP 5 - month.8)")
union4 <- cruzada_logistica(surgical_dataset, target, candidatos_4, nombres_candidatos_4,
                            grupos = 5, repe = 5)
union4$modelo <- with(union4, reorder(modelo,tasa, mean))
t <- ggplot(union4, aes(x = modelo, y = tasa)) +
  geom_boxplot() +
  ggtitle("Comparacion (tasa fallos)") +
  labs(color='Dataset')  +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))

union4$modelo <- with(union4, reorder(modelo,auc, mean))
a <- ggplot(union4, aes(x = modelo, y = auc)) +
  geom_boxplot() +
  ggtitle("Comparacion (AUC)") +
  labs(color='Dataset')  +
  theme(text = element_text(size=13, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.8))

ggpubr::ggarrange(t, a, common.legend = TRUE)
ggsave('./charts/01_feature_selection_comparacion_final.png')

#-- De ahora en adelante probaremos con dos modelos candidatos
#   LOGISTICA BIC (TOP 5) -> seleccion1
#   "Age" "mortality_rsi" "ccsMort30Rate" "bmi" "month.8" 

#   LOGISTICA BIC (TOP 4) -> seleccion2
#   "Age" "mortality_rsi" "bmi" "month.8"
#
write.csv(surgical_dataset, "data/surgical_dataset_final.csv", row.names = FALSE)

#-- Nota: de cara a las comparaciones con el resto de modelos, aumentamos el numero de repeticiones a 10
candidatos_final         <- list(candidato.bic.5, top4)
nombres_candidatos_final <- c("BIC (TOP 5 - month.8)", "AIC-BIC-TOP 4")
union_final <- cruzada_logistica(surgical_dataset, target, candidatos_final, nombres_candidatos_final,
                            grupos = 5, repe = 10)
rm(candidatos_final)
rm(nombres_candidatos_final)

union_10_rep <- rbind(union4[union4$modelo %in% c("BIC (TOP 5 - month.8)", "AIC-BIC-TOP 4"), ], union_final)
union_10_rep$rep <- c(rep("5", 10), rep("10", 20))

# Tasa de fallos
p <- ggplot(union_10_rep, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo") +
  theme(text = element_text(size=13, face = "bold"))
p
ggsave('./charts/01_boxplot_log_modelo1_error_10rep.jpeg')

# AUC
g <- ggplot(union_10_rep, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo") +
  theme(text = element_text(size=13, face = "bold"))
g
ggsave('./charts/01_boxplot_log_modelo1_auc_10rep.jpeg')

ggpubr::ggarrange(p, g, common.legend = TRUE)
ggsave('./charts/01_feature_selection_comparacion_5_10_rep.png')

surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)

names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

control<-trainControl(method = "repeatedcv",number=5,repeats=10,
                      savePredictions = "all",classProbs=TRUE)

modelo1 <- train(as.formula(paste0(target, "~" , paste0(candidato.bic.3, collapse = "+"))),data=surgical_dataset,
                   trControl=control,method="glm",family = binomial(link="logit"))

modelo2 <- train(as.formula(paste0(target, "~" , paste0(candidato.bic.5, collapse = "+"))),data=surgical_dataset,
                 trControl=control,method="glm",family = binomial(link="logit"))

matriz_conf_1 <- matriz_confusion_predicciones(modelo1, formula = paste0(target, "~" , 
                                                                paste0(candidato.bic.3, collapse = "+")),
                                                corte = 0.5,
                                                dataset = surgical_test_data
                                                )

matriz_conf_2 <- matriz_confusion_predicciones(modelo2, formula = paste0(target, "~" , 
                                                                paste0(candidato.bic.5, collapse = "+")),
                                               corte = 0.5,
                                               dataset = surgical_test_data
)

#--- Predicciones
#     Modelo 1
#     Reference
#     Prediction    No  Yes
#             No  6250  344
#             Yes 1465  722

#      Modelo 2
#      Reference
#      Prediction   No  Yes
#             No  6319  275
#             Yes 1636  551
#

#---- Estadisticas
# Por tasa fallos --------------- auc
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/SeleccionVariables.RData")



