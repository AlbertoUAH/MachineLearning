# ------------- Punto de corte ---------------
# Objetivo: estudiar el mejor punto de corte
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  library(ggrepel)       # Labels ggplot2
  library(stringi)       # Tratamiento de strings
  library(ggplotgui)     # Interfaz ggplot2
  library(visualpred)    # Visualizacion algoritmos clasificacion
  library(egg)           # Layouts de ggplot2
  
  source("./librerias/librerias_propias.R")
  source("./librerias/funcion resultadosrf.R")
  source("./librerias/funcion resultadosgbm.R")
  source("./librerias/funcion resultadosxgboost.R")
})

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)
surgical_dataset_completo <- rbind(surgical_dataset, surgical_test_data)
rm(surgical_test_data)

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 1
var_modelo1 <- c("mortality_rsi", "bmi", "ahrq_ccs", "Age")
#-- Modelo 2
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

#-- Probamos con diferentes puntos de corte
puntos_corte <- seq(0, 1, 0.05)
dataframe_puntos_corte_rf      <- data.frame()
dataframe_puntos_corte_bagging <- data.frame()
dataframe_puntos_corte_gbm     <- data.frame()
dataframe_puntos_corte_xgboost <- data.frame()

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo2,mtry=2,ntree=2000,sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_rf <- rbind(dataframe_puntos_corte_rf, result_rf)
  print(punto_corte)
}

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo2,mtry=4,ntree=900, sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_bagging <- rbind(dataframe_puntos_corte_bagging, result_rf)
  print(punto_corte)
}

for(punto_corte in puntos_corte) {
  result_gbm <- resultadosgbm(dataf=surgical_dataset,
                              vardep=target,listconti=var_modelo2,
                              n.minobsinnode=20,shrink=0.2,n.trees=100,
                              bag.fraction=0.5,corte=punto_corte)
  result_gbm$AUC <- as.numeric(result_gbm$AUC)
  result_gbm$tasa <- as.numeric(result_gbm$tasa)
  result_gbm$pto_corte <- punto_corte
  
  dataframe_puntos_corte_gbm <- rbind(dataframe_puntos_corte_gbm, result_gbm)
  print(punto_corte)
}

for(punto_corte in puntos_corte) {
  result_xgboost <- resultadosxgboost(dataf=surgical_dataset,
                              vardep=target,listconti=var_modelo2,
                              corte=punto_corte)
  result_xgboost$AUC <- ifelse(is.null(result_xgboost$AUC), 0, as.numeric(result_xgboost$AUC))
  result_xgboost$tasa <- as.numeric(result_xgboost$tasa)
  result_xgboost$pto_corte <- punto_corte
  
  dataframe_puntos_corte_xgboost <- rbind(dataframe_puntos_corte_xgboost, result_xgboost)
  print(punto_corte)
}

#-- ¿Y si probamos con el modelo 1?
#-- Probamos con diferentes puntos de corte
dataframe_puntos_corte_rf_1 <- data.frame()
dataframe_puntos_corte_bagging_1 <- data.frame()

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo1,mtry=3,ntree=2000,sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_rf_1 <- rbind(dataframe_puntos_corte_rf_1, result_rf)
  print(punto_corte)
}

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo1,mtry=5,ntree=900, sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_bagging_1 <- rbind(dataframe_puntos_corte_bagging_1, result_rf)
  print(punto_corte)
}
rm(result_rf); rm(punto_corte)

colors <- c("Sentividad" = "red", "Especificidad" = "darkblue")

#-- Comparacion sub-dataset y original (RF y Bagging)
#-- Modelo 2
p <- ggplot(dataframe_puntos_corte_rf, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

q <- ggplot(dataframe_puntos_corte_gbm, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (gbm)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

ggpubr::ggarrange(p, q, common.legend = TRUE)

#-- Modelo 2
p <- ggplot(dataframe_puntos_corte_rf_1, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest) - Modelo 1") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

q <- ggplot(dataframe_puntos_corte_bagging_1, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Bagging) - Modelo 1") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))
ggpubr::ggarrange(p, q, common.legend = TRUE)

#-----------------------------------------------------------------------------------------------------
#-- Probamos con diferentes semillas (10)
semillas <- sample(1:10000, size = 10)
dataframe_puntos_corte_rf_multiple_seeds <- data.frame()
dataframe_puntos_corte_gbm_multiple_seeds <- data.frame()

for(semilla in semillas) {
  for(punto_corte in puntos_corte) {

    result_rf <- resultadosgbm(dataf=surgical_dataset,
                               vardep=target,listconti=var_modelo2,
                               n.minobsinnode=20,shrink=0.2,n.trees=100,
                               bag.fraction=0.5,corte=punto_corte)
    result_rf$AUC <- as.numeric(result_rf$AUC)
    result_rf$tasa <- as.numeric(result_rf$tasa)
    result_rf$pto_corte <- punto_corte
    
    dataframe_puntos_corte_gbm_multiple_seeds <- rbind(dataframe_puntos_corte_gbm_multiple_seeds, as.data.frame(t(unlist(result_rf))))
  }
  print(semilla)
}

#-- Solo con el subconjunto
p <- ggplot(dataframe_puntos_corte_rf_multiple_seeds, aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

q <- ggplot(dataframe_puntos_corte_rf_multiple_seeds[dataframe_puntos_corte_rf_multiple_seeds$pto_corte > 0.1 & dataframe_puntos_corte_rf_multiple_seeds$pto_corte< 0.5, ], aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (RF - Ampliado)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

p <- ggplot(dataframe_puntos_corte_gbm_multiple_seeds, aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (gbm)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))

r <- ggplot(dataframe_puntos_corte_gbm_multiple_seeds[dataframe_puntos_corte_gbm_multiple_seeds$pto_corte > 0.1 & dataframe_puntos_corte_gbm_multiple_seeds$pto_corte< 0.5, ], aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (gbm - Ampliado)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors) +  theme(text = element_text(face = "bold", size = 11))


#-- ¿En que punto de corte existe un equilibrio entre sensitividad y especificidad?
#   Random Forest
obs_rf <- which(abs(dataframe_puntos_corte_rf_multiple_seeds$sensitividad - 
                    dataframe_puntos_corte_rf_multiple_seeds$especificidad) == min(abs(dataframe_puntos_corte_rf_multiple_seeds$sensitividad - 
                                                                                  dataframe_puntos_corte_rf_multiple_seeds$especificidad)))
dataframe_puntos_corte_rf_multiple_seeds[obs_rf, ]

#   Bagging
obs_bagging <- which(abs(dataframe_puntos_corte_bagging_multiple_seeds$sensitividad - 
                         dataframe_puntos_corte_bagging_multiple_seeds$especificidad) == min(abs(dataframe_puntos_corte_bagging_multiple_seeds$sensitividad - 
                                                                                           dataframe_puntos_corte_bagging_multiple_seeds$especificidad)))
dataframe_puntos_corte_bagging_multiple_seeds[obs_bagging, ]

#-- Indice de Youden
indice_youden_rf      <- dataframe_puntos_corte_rf_multiple_seeds$especificidad + dataframe_puntos_corte_rf_multiple_seeds$sensitividad - 1
indice_youden_gbm <- dataframe_puntos_corte_gbm_multiple_seeds$especificidad + dataframe_puntos_corte_gbm_multiple_seeds$sensitividad - 1

dataframe_puntos_corte_rf_multiple_seeds[which(indice_youden_rf == max(indice_youden_rf)), ]
dataframe_puntos_corte_gbm_multiple_seeds[which(indice_youden_gbm == max(indice_youden_gbm)), ]

# F-1 score
f1_rf <- dataframe_puntos_corte_rf_multiple_seeds$VP / (dataframe_puntos_corte_rf_multiple_seeds$VP + 
                                                          0.5 * (dataframe_puntos_corte_rf_multiple_seeds$FP + 
                                                                   dataframe_puntos_corte_rf_multiple_seeds$FN))

f1_bagging <- dataframe_puntos_corte_bagging_multiple_seeds$VP / (dataframe_puntos_corte_bagging_multiple_seeds$VP + 
                                                                  0.5 * (dataframe_puntos_corte_bagging_multiple_seeds$FP + 
                                                                           dataframe_puntos_corte_bagging_multiple_seeds$FN))
f1_rf <- data.frame(pto_corte = dataframe_puntos_corte_rf_multiple_seeds$pto_corte,
                    f1_score  = f1_rf)

f1_bagging <- data.frame(pto_corte = dataframe_puntos_corte_bagging_multiple_seeds$pto_corte,
                         f1_score  = f1_bagging)

ggplot(NULL, aes(x = factor(pto_corte), y = f1_score)) + 
  geom_boxplot(data = f1_rf, aes(colour = "RF")) +
  geom_boxplot(data = f1_bagging, aes(color = "Bagging")) +
  ggtitle("F1-score (random forest vs bagging)") +
  labs(x ="Punto de corte", y = "F1") +
  scale_colour_manual(name="Modelo",
                      values=c(RF="red", Bagging="darkblue"))

dataframe_puntos_corte_rf_multiple_seeds[which(f1_rf$f1_score == max(f1_rf$f1_score)), ]
dataframe_puntos_corte_bagging_multiple_seeds[which(f1_bagging$f1_score == max(f1_bagging$f1_score)), ]


result_rf <- famdcontour(dataf=surgical_dataset,listconti=var_modelo1,listclass=c(""),vardep=target,
                         title="Random Forest",title2=" ",selec=0,modelo="rf",classvar=0,mtry=2,ntree=2000,sampsize=1000,
                         nodesize=20, alpha1 = 1, alpha2 = 1, alpha3 = 1)

result_bagging <-famdcontour(dataf=surgical_dataset,listconti=var_modelo2,listclass=c(""),vardep=target,
                             title="gbm",title2=" ",selec=0,modelo="rf",classvar=0,n.minobsinnode=20,shrink=0.2,ntreegbm = 100,
                             bag.fraction=0.5, alpha1 = 1, alpha2 = 1, alpha3 = 1)

candidato_bic <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "dow.0", 
                                    "Age", "moonphase.0", "baseline_osteoart", "asa_status.0")

result_rf_2 <- famdcontour(dataf=surgical_dataset,listconti=candidato_bic,listclass=c(""),vardep=target,
                         title="Random Forest",title2=" ",selec=0,modelo="rf",classvar=0,mtry=2,ntree=2000,sampsize=1000,
                         nodesize=20, alpha1 = 1, alpha2 = 1, alpha3 = 1)

result_bagging_2 <-famdcontour(dataf=surgical_dataset,listconti=candidato_bic,listclass=c(""),vardep=target,
                             title="gbm",title2=" ",selec=0,modelo="rf",classvar=0,n.minobsinnode=20,shrink=0.2,ntreegbm = 100,
                             bag.fraction=0.5, alpha1 = 1, alpha2 = 1, alpha3 = 1)




stopCluster(cluster)














