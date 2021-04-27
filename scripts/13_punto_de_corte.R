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

#-- Modelo 2
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

#-- Probamos con diferentes puntos de corte
puntos_corte <- seq(0, 1, 0.05)
dataframe_puntos_corte_rf <- data.frame()
dataframe_puntos_corte_bagging <- data.frame()

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,mtry=2,ntree=2000,sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_rf <- rbind(dataframe_puntos_corte_rf, result_rf)
  print(punto_corte)
}

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,mtry=4,ntree=900, sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_bagging <- rbind(dataframe_puntos_corte_bagging, result_rf)
  print(punto_corte)
}

# Probamos tambien con el dataset completo
for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset_completo,vardep=target,mtry=2,ntree=2000,sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_rf <- rbind(dataframe_puntos_corte_rf, result_rf)
  print(punto_corte)
}

dataframe_puntos_corte_rf$dataset <- c(rep("subconjunto", 21), rep("original", 21))

for(punto_corte in puntos_corte) {
  result_rf <- resultadosrf(dataf=surgical_dataset_completo,vardep=target,mtry=4,ntree=900, sampsize=1000,
                            nodesize=20,corte=punto_corte)
  result_rf$AUC <- as.numeric(result_rf$AUC)
  result_rf$tasa <- as.numeric(result_rf$tasa)
  result_rf$pto_corte <- punto_corte
  
  dataframe_puntos_corte_bagging <- rbind(dataframe_puntos_corte_bagging, result_rf)
  print(punto_corte)
}

dataframe_puntos_corte_bagging$dataset <- c(rep("subconjunto", 21), rep("original", 21))

colors <- c("Sentividad" = "red", "Especificidad" = "darkblue")

#-- Comparacion sub-dataset y original (RF y Bagging)
ggplot(dataframe_puntos_corte_rf, aes(x = pto_corte)) + 
  geom_point(aes(y = sensitividad, colour = dataset)) +
  geom_point(aes(y = especificidad, colour = dataset)) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  theme_bw()

ggplot(dataframe_puntos_corte_bagging, aes(x = pto_corte)) + 
  geom_point(aes(y = sensitividad, colour = dataset)) +
  geom_point(aes(y = especificidad, colour = dataset)) +
  ggtitle("Especificidad vs Sensitividad (Bagging)") +
  labs(x ="Punto de corte", y = "Valor") +
  theme_bw()

#-- Solo con el subconjunto
ggplot(dataframe_puntos_corte_rf[dataframe_puntos_corte_rf$dataset == "subconjunto", ], aes(x = pto_corte)) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

ggplot(dataframe_puntos_corte_bagging[dataframe_puntos_corte_bagging$dataset == "subconjunto", ], aes(x = pto_corte)) + 
  geom_point(aes(y = sensitividad, color = "Sentividad")) +
  geom_point(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Bagging)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

#-- Probamos con diferentes semillas (15)
semillas <- c(1234, sample(1:10000, size = 15))
dataframe_puntos_corte_rf_multiple_seeds <- data.frame()
dataframe_puntos_corte_bagging_multiple_seeds <- data.frame()

for(semilla in semillas) {
  for(punto_corte in puntos_corte) {
    result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,mtry=2,ntree=2000,sampsize=1000,
                              nodesize=20,corte=punto_corte, sinicio = semilla)
    result_rf$AUC <- as.numeric(result_rf$AUC)
    result_rf$tasa <- as.numeric(result_rf$tasa)
    result_rf$pto_corte <- punto_corte
    
    dataframe_puntos_corte_rf_multiple_seeds <- rbind(dataframe_puntos_corte_rf_multiple_seeds, result_rf)

    result_rf <- resultadosrf(dataf=surgical_dataset,vardep=target,mtry=4,ntree=900, sampsize=1000,
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

#-- Hagamos zoom
ggplot(dataframe_puntos_corte_rf_multiple_seeds[dataframe_puntos_corte_rf_multiple_seeds$pto_corte <= 0.3, ], aes(x = factor(pto_corte))) + 
  geom_boxplot(aes(y = sensitividad, color = "Sentividad")) +
  geom_boxplot(aes(y = especificidad, color = "Especificidad")) +
  ggtitle("Especificidad vs Sensitividad (Random Forest)") +
  labs(x ="Punto de corte", y = "Valor") +
  scale_color_manual(values = colors)

result_rf <- famdcontour(dataf=surgical_dataset,listconti=var_modelo2,listclass=c(""),vardep=target,
                         title="Random Forest",title2=" ",selec=0,modelo="rf",classvar=0,mtry=2,ntree=2000,sampsize=1000,
                         nodesize=20)

result_bagging <-famdcontour(dataf=surgical_dataset,listconti=var_modelo2,listclass=c(""),vardep=target,
                             title="Bagging",title2=" ",selec=0,modelo="rf",classvar=0,mtry=4,ntree=900,sampsize=1000,
                             nodesize=20)






















