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
  library(readxl)        # Lectura de ficheros Excel
  
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

#-- Modelo 1: interesan nodes 30, 25, 20, 15 y decay 0.1
size.candidato.1 <- c(15, 20, 25, 30)
decay.candidato.1 <- c(0.1, 0.1, 0.1, 0.1)

union_2 <- comparar_modelos_red(
                                surgical_dataset,
                                target,
                                var_modelo1,
                                sizes = size.candidato.1,
                                decays = decay.candidato.1,
                                grupos = 5,
                                repe = 5,
                                iteraciones = 200
)

# Nos decantamos por 20 nodos y decay 0.1

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

#-- Modelo 2: interesan nodes 40,30,25,20 y decay 0.01
size.candidato.2 <- c(20, 25, 30, 35, 40, 45, 50)
decay.candidato.2 <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

union_3 <- comparar_modelos_red(
  surgical_dataset,
  target,
  var_modelo2,
  sizes = size.candidato.2,
  decays = decay.candidato.2,
  grupos = 5,
  repe = 5,
  iteraciones = 200
)

# ¿Y si bajamos a 5, 10 o 15?
size.candidato.2_1 <- c(5, 10, 15, 20)
decay.candidato.2_1 <- c(0.01, 0.01, 0.01, 0.01)

union_3_1 <- comparar_modelos_red(
  surgical_dataset,
  target,
  var_modelo2,
  sizes = size.candidato.2_1,
  decays = decay.candidato.2_1,
  grupos = 5,
  repe = 5,
  iteraciones = 200
)

# No mejora practicamente disminuyendo el numero de nodos de la red
# Nos decantamos por 20 nodos y decay 0.01

#--- Modificamos el numero de iteraciones
union_it_1  <- data.frame(tasa = numeric(), auc = numeric(), num_iter = character())
union_it_2  <- data.frame(tasa = numeric(), auc = numeric(), num_iter = character())
for(num_iteraciones in c(50, 100, 200, 300, 400, 500)) {
  union_it_1_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                      listconti=var_modelo1, listclass=c(""),
                                      grupos=5,sinicio=1234,repe=5, size=20,
                                      decay=0.1,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_1_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_1_temp$modelo   <- NULL
  union_it_1 <- rbind(union_it_1, union_it_1_temp)
  
  union_it_2_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                      listconti=var_modelo2, listclass=c(""),
                                      grupos=5,sinicio=1234,repe=5, size=20,
                                      decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_2_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_2_temp$modelo   <- NULL
  union_it_2 <- rbind(union_it_2, union_it_2_temp)
  
  union_it_2_temp$num_iter <- rep(as.character(num_iteraciones), nrow(union_it_2_temp))
  
  print(paste0(num_iteraciones, " - done!"))
}
rm(union_it_1_temp)
rm(union_it_2_temp)

#-- Mostramos graficamente los resultados
#-  Modelo 1
ggplot(union_it_1, aes(x = num_iter, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/02_boxplot_nnet_modelo1_iteraciones_error.jpeg')

ggplot(union_it_1, aes(x = num_iter, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/02_boxplot_nnet_modelo1_iteraciones_AUC.jpeg')

# Conclusion: con el modelo 1 mantenemos 200 iteraciones

#- Modelo 2
ggplot(union_it_2, aes(x = num_iter, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/02_boxplot_nnet_modelo2_iteraciones_error.jpeg')

ggplot(union_it_2, aes(x = num_iter, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/02_boxplot_nnet_modelo2_iteraciones_AUC.jpeg')

# Conclusion: con el modelo 2 mantenemos 200 iteraciones

#---- Guardamos el fichero RData
save.image(file = "./rdata/RedesNeuronales.RData")

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

# Aplico caret y construyo modelos finales
control <- trainControl(method = "repeatedcv",number=5,repeats=5,
                      savePredictions = "all",classProbs=TRUE) 

avnnetgrid_1 <-  expand.grid(size=20,decay=0.1,bag=FALSE)
set.seed(1234)
avnnet_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                  data=surgical_dataset, method="avNNet",linout = FALSE,maxit=200,repeats=5,
                  trControl=control,tuneGrid=avnnetgrid_1,trace=FALSE)

matriz_conf_1 <- matriz_confusion_predicciones(avnnet_1, NULL, surgical_test_data, 0.5)

avnnetgrid_2 <-  expand.grid(size=20,decay=0.01,bag=FALSE)
set.seed(1234)
avnnet_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                  data=surgical_dataset, method="avNNet",linout = FALSE,maxit=200,repeats=5,
                  trControl=control,tuneGrid=avnnetgrid_2,trace=FALSE)

matriz_conf_2 <- matriz_confusion_predicciones(avnnet_2, NULL, surgical_test_data, 0.5)
rm(avnnetgrid_1)
rm(avnnetgrid_2)

#--- Predicciones
#     Modelo 1
#     Reference
#     Prediction   No  Yes
#     No         6442  152
#     Yes        809  1378

#     Modelo 2
#     Reference
#     Prediction   No  Yes
#     No         6462  132
#     Yes        746  1441
#

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)

ggplot(modelos_actuales, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/comparativas/02_log_avnnet_tasa.jpeg')

ggplot(modelos_actuales, aes(x = modelo, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/comparativas/02_log_avnnet_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#   avnnet modelo 2           avnnet modelo 2
#   avnnet modelo 1           avnnet modelo 1
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/RedesNeuronales.RData")

