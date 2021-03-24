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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", 
                 "Age")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "ccsMort30Rate", "bmi", "ahrq_ccs")

#---- Comenzamos con el modelo 1
#     Nota: por el momento: itera = 200
# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 20 ~ 292 parametros
# Si k = 5, entonces 7 * h + 1 = 292 => es decir, 41-42 nodos

# Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 30 ~ 195 parametros
# Si k = 5, entonces 7 * h + 1 = 195 => es decir, 27-28 nodos
size.candidato.1 <- c(5, 10, 15, 20, 25, 30, 35, 40)
decay.candidato.1 <- c(0.1, 0.01, 0.001)

cvnnet.candidato.1 <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                       listconti=var_modelo1, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.1,
                                       decay=decay.candidato.1,repeticiones=5,itera=200)

#-- Modelo 1: interesan nodes 35, 30, 25, 20, 15 y decay 0.01
# Da buenos resultados con 25-30 nodos (aunque con 30 presenta mayor variabilidad en cuanto a AUC se refiere)
size.candidato.1 <- c(15, 20, 25, 30, 35, 40)
decay.candidato.1 <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

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

# Llama la atencion 25-30 y 40 nodos, aunque con 25 nodos la tasa de fallos es bastante menor
# aunque con alta varianza en el AUC
# ¿Y si aumentamos el numero de repeticiones?
union_2_bis <- comparar_modelos_red(
  surgical_dataset,
  target,
  var_modelo1,
  sizes = c(15, 20, 25, 30, 40),
  decays = c(0.01, 0.01, 0.01, 0.01, 0.01),
  grupos = 5,
  repe = 10,
  iteraciones = 200
)

# Nos decantamos por 25-30 nodos y decay 0.01 Por los siguientes motivos:
# -> Con tan solo 5 repeticiones, la varianza de la tasa de fallos es mayor en
#    con 25 nodos que con 30.
# -> Aunque el valor AUC sea ligeramente mayor con 25 nodos, empleando 25 nodos tendriamos
#    7 * h + 1 = 176 parametros, lo que equivale a 5854 / 176 ~ 33 observaciones por parametro
#    Con 30 nodos, tendriamos 7 * h + 1 = 211 parametros, lo que equivale a 28 observaciones por parametro

#---- Continuamos con el modelo 2
# Si quisieramos 20 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 20 ~ 292 parametros
# Si k = 5, entonces 7 * h + 1 = 292 => 41.57, es decir, 41-42 nodos

# Si quisieramos 30 observaciones por parametro:
# h * (k + 1) + h + 1 = 5854 / 30 ~ 195 parametros
# Si k = 5, entonces 7 * h + 1 = 195 => 27.71, es decir, 27-28 nodos
size.candidato.2 <- c(5, 10, 15, 20, 25, 30, 35, 40)
decay.candidato.2 <- c(0.1, 0.01, 0.001)

cvnnet.candidato.2 <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                       listconti=var_modelo2, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=5, size=size.candidato.2,
                                       decay=decay.candidato.2,repeticiones=5,itera=200)

#-- Modelo 2: interesan nodes 40,30,25,20 y decay 0.01 (tambien vamos a echar un vistazo con 15 nodos)
size.candidato.2 <- c(15, 20, 25, 30, 35, 40)
decay.candidato.2 <- c(0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

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
# A simple vista, como posibles opciones tendriamos o 20, 25 o 30 nodos en la red
# Problema: con 30 nodos, la varianza es muy elevada en comparacion con 20 o 25 nodos

# ¿Y si aumentamos el numero de repeticiones?
union_3_bis_2 <- comparar_modelos_red(
  surgical_dataset,
  target,
  var_modelo2,
  sizes = c(20, 25, 30),
  decays = c(0.01, 0.01, 0.01),
  grupos = 5,
  repe = 10,
  iteraciones = 200
)

# Nos decantamos por 20-25 nodos y decay 0.01

# Tenemos dos posibles modelos candidatos para cada seleccion de variables
# Modelo 1: 25-30 nodos
# Modelo 2: 20-25 nodos
#--- Modificamos el numero de iteraciones 
union_it_1  <- data.frame(tasa = numeric(), auc = numeric(), num_iter = character())
union_it_2  <- data.frame(tasa = numeric(), auc = numeric(), num_iter = character())
for(num_iteraciones in c(100, 200, 300, 400, 500)) {
  union_it_1_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                      listconti=var_modelo1, listclass=c(""),
                                      grupos=5,sinicio=1234,repe=5, size=25,
                                      decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_1_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_1_temp$modelo   <- NULL
  union_it_1 <- rbind(union_it_1, union_it_1_temp)
  
  union_it_1_1_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                        listconti=var_modelo1, listclass=c(""),
                                        grupos=5,sinicio=1234,repe=5, size=30,
                                        decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_1_1_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_1_1_temp$modelo   <- NULL
  union_it_1 <- rbind(union_it_1, union_it_1_1_temp)
  
  print(paste0(num_iteraciones, " - done!"))
}
rm(union_it_1_temp)
rm(union_it_1_1_temp)

union_it_1$modelo <- rep(c(rep("25 nodos", 5), rep("30 nodos", 5)), 5)

#-- Mostramos graficamente los resultados
#-  Modelo 1
ggplot(union_it_1, aes(x = num_iter, y = tasa, col = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo1_iteraciones_error.jpeg')

ggplot(union_it_1, aes(x = num_iter, y = auc, col = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo1_iteraciones_AUC.jpeg')

# Conclusion: con el modelo 1 mantenemos 200 iteraciones
# Sin embargo, no queda tan claro si mantener 25 o 30 nodos
# Me decantaria por una red de 30 nodos

# Modelo 2
# Resulta que con 300 iteraciones, tanto la tasa de fallo como el AUC
# se estabilizan
union_it_2  <- data.frame(tasa = numeric(), auc = numeric(), num_iter = character())
for(num_iteraciones in c(100, 200, 250, 300, 400)) {
  union_it_2_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                      listconti=var_modelo2, listclass=c(""),
                                      grupos=5,sinicio=1234,repe=5, size=20,
                                      decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_2_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_2_temp$modelo   <- NULL
  union_it_2 <- rbind(union_it_2, union_it_2_temp)
  
  union_it_2_1_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                        listconti=var_modelo2, listclass=c(""),
                                        grupos=5,sinicio=1234,repe=5, size=25,
                                        decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_2_1_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_2_1_temp$modelo   <- NULL
  union_it_2 <- rbind(union_it_2, union_it_2_1_temp)
  
  union_it_2_2_temp <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                        listconti=var_modelo2, listclass=c(""),
                                        grupos=5,sinicio=1234,repe=5, size=30,
                                        decay=0.01,repeticiones=5,itera=num_iteraciones)[[1]]
  
  union_it_2_2_temp$num_iter <- rep(as.character(num_iteraciones), 5)
  union_it_2_2_temp$modelo   <- NULL
  union_it_2 <- rbind(union_it_2, union_it_2_2_temp)
  
  print(paste0(num_iteraciones, " - done!"))
}
rm(union_it_2_temp)
rm(union_it_2_1_temp)
rm(union_it_2_2_temp)

# Graficamos el AUC y tasa de fallo del modelo 2
union_it_2$modelo <- rep(c(rep("20 nodos", 5), rep("25 nodos", 5), rep("30 nodos", 5)), 5)

ggplot(union_it_2, aes(x = num_iter, y = tasa, col = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo2_iteraciones_error_200_300.jpeg')

ggplot(union_it_2, aes(x = num_iter, y = auc, col = modelo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo2_iteraciones_AUC_200_300.jpeg')

###############
#-- Conclusiones
#   Modelo 1: Elegimos 30 nodos y 0.01 lr (con 200 iteraciones) (alrededor de 33 obs. por param.)
#   Mejor AUC
#   Modelo 2: Elegimos 20 nodos y 0.01 lr (con 250 iteraciones)
#   Lo empirico: menor varianza en comparacion con modelos con 25 y 30 nodos
###############
#-- ¿Como varian ambas redes si aumentamos a 10 repeticiones?
modelo1_final <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                   listconti=var_modelo1, listclass=c(""),
                   grupos=5,sinicio=1234,repe=10, size=30,
                   decay=0.01,repeticiones=5,itera=200)[[1]]

modelo2_final <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                                  listconti=var_modelo2, listclass=c(""),
                                  grupos=5,sinicio=1234,repe=10, size=20,
                                  decay=0.01,repeticiones=5,itera=250)[[1]]

union_final <- rbind(union_it_2[union_it_2$modelo == "20 nodos" & union_it_2$num_iter == "250", c("tasa", "auc")],
                     union_it_1[union_it_1$modelo == "30 nodos" & union_it_1$num_iter == "200", c("tasa", "auc")],
                     modelo1_final, modelo2_final)
union_final$modelo <- c(rep("20+250", 5), rep("30+200", 5), rep("30+200", 10), rep("20+250", 10))
union_final$rep    <- c(rep("5", 10), rep("10", 20))

ggplot(union_final, aes(x = modelo, y = tasa, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo2_iteraciones_error_10_rep.jpeg')

ggplot(union_final, aes(x = modelo, y = auc, col = rep)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/avnnet/02_boxplot_nnet_modelo2_iteraciones_AUC_10_rep.jpeg')

rm(modelo1_final)
rm(modelo2_final)

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target     <- as.factor(surgical_test_data$target)

# Aplico caret y construyo modelos finales
control <- trainControl(method = "repeatedcv",number=5,repeats=10,
                      savePredictions = "all",classProbs=TRUE) 

avnnetgrid_1 <-  expand.grid(size=30,decay=0.01,bag=FALSE)
set.seed(1234)
avnnet_1 <- train(as.formula(paste0(target, "~" , paste0(var_modelo1, collapse = "+"))),
                  data=surgical_dataset, method="avNNet",linout = FALSE,maxit=200,repeats=5,
                  trControl=control,tuneGrid=avnnetgrid_1)

matriz_conf_1 <- matriz_confusion_predicciones(avnnet_1, NULL, surgical_test_data, 0.5)

avnnetgrid_2 <-  expand.grid(size=20,decay=0.01,bag=FALSE)
set.seed(1234)
avnnet_2 <- train(as.formula(paste0(target, "~" , paste0(var_modelo2, collapse = "+"))),
                  data=surgical_dataset, method="avNNet",linout = FALSE,maxit=250,repeats=5,
                  trControl=control,tuneGrid=avnnetgrid_2)

matriz_conf_2 <- matriz_confusion_predicciones(avnnet_2, NULL, surgical_test_data, 0.5)
rm(avnnetgrid_1)
rm(avnnetgrid_2)

#--- Predicciones
#     Modelo 1
#     Reference
#     Prediction   No  Yes
#     No         6474  120
#     Yes        731  1456

#     Modelo 2
#     Reference
#     Prediction   No  Yes
#     No         6515  79
#     Yes        766  1421
#

modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx"))
modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
ggplot(modelos_actuales, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo")

ggsave('./charts/comparativas/02_log_avnnet_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
ggplot(modelos_actuales, aes(x = modelo, y = auc)) +
  geom_boxplot(fill =  "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo")

ggsave('./charts/comparativas/02_log_avnnet_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#   avnnet modelo 1           avnnet modelo 2
#   avnnet modelo 2           avnnet modelo 1
#   log.   modelo 2           log.   modelo 1
#   log.   modelo 1           log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/RedesNeuronales.RData")

