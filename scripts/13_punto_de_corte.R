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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
#-- Modelo 2
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

#-- Probamos con diferentes puntos de corte
puntos_corte <- seq(0, 1, 0.05)
dataframe_puntos_corte_rf <- data.frame()
dataframe_puntos_corte_bagging <- data.frame()

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
ggplot(dataframe_puntos_corte_rf, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest) - Modelo 2") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

ggplot(dataframe_puntos_corte_bagging, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Bagging) - Modelo 2") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

#-- Modelo 2
ggplot(dataframe_puntos_corte_rf_1, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest) - Modelo 1") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

ggplot(dataframe_puntos_corte_bagging_1, aes(x = factor(pto_corte))) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Bagging) - Modelo 1") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

#-----------------------------------------------------------------------------------------------------
#-- Probamos con diferentes semillas (10)
semillas <- sample(1:10000, size = 10)
dataframe_puntos_corte_rf_multiple_seeds <- data.frame()
dataframe_puntos_corte_bagging_multiple_seeds <- data.frame()

for(semilla in semillas) {
  for(punto_corte in puntos_corte) {
    result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo2,mtry=2,ntree=2000,sampsize=1000,
                              nodesize=20,corte=punto_corte, sinicio = semilla)
    result_rf$AUC <- as.numeric(result_rf$AUC)
    result_rf$tasa <- as.numeric(result_rf$tasa)
    result_rf$pto_corte <- punto_corte
    
    dataframe_puntos_corte_rf_multiple_seeds <- rbind(dataframe_puntos_corte_rf_multiple_seeds, result_rf)

    result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,listconti=var_modelo2,mtry=4,ntree=900, sampsize=1000,
                              nodesize=20,corte=punto_corte, sinicio = semilla)
    result_rf$AUC <- as.numeric(result_rf$AUC)
    result_rf$tasa <- as.numeric(result_rf$tasa)
    result_rf$pto_corte <- punto_corte
    
    dataframe_puntos_corte_bagging_multiple_seeds <- rbind(dataframe_puntos_corte_bagging_multiple_seeds, result_rf)
  }
  print(semilla)
}

#-- Solo con el subconjunto
ggplot(dataframe_puntos_corte_rf_multiple_seeds, aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

ggplot(dataframe_puntos_corte_bagging_multiple_seeds, aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Bagging)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)


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
indice_youden_bagging <- dataframe_puntos_corte_bagging_multiple_seeds$especificidad + dataframe_puntos_corte_bagging_multiple_seeds$sensitividad - 1

dataframe_puntos_corte_rf_multiple_seeds[which(indice_youden_rf == max(indice_youden_rf)), ]
dataframe_puntos_corte_bagging_multiple_seeds[which(indice_youden_bagging == max(indice_youden_bagging)), ]

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


result_rf <- famdcontour(dataf=surgical_dataset,listconti=var_modelo2,listclass=c(""),vardep=target,
                         title="Random Forest",title2=" ",selec=0,modelo="rf",classvar=0,mtry=2,ntree=2000,sampsize=1000,
                         nodesize=20)

result_bagging <-famdcontour(dataf=surgical_dataset,listconti=var_modelo2,listclass=c(""),vardep=target,
                             title="Bagging",title2=" ",selec=0,modelo="rf",classvar=0,mtry=4,ntree=900,sampsize=1000,
                             nodesize=20)




stopCluster(cluster)














