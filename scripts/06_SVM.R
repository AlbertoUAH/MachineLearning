# ------------- SVM ---------------
# Objetivo: elaborar el mejor modelo de SVM de acuerdo
#           a los valores de prediccion obtenidos tras variar los parametros
#           C, grado, escala y sigma
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(parallel)      # Paralelizacion de funciones (I)
  library(doParallel)    # Paralelizacion de funciones (II)
  library(caret)         # Recursive Feature Elimination
  library(readxl)        # Lectura ficheros .xlsx
  library(DescTools)     # Reordenacion de variales categoricas
  
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
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", 
                 "Age")

#-- Modelo 2
var_modelo2 <- c("Age", "mortality_rsi", "bmi", "month.8")

#--- SVM binaria-lineal
#    Modelo 1
C_binaria_1 <- expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))


svm_bin_1 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                           listconti = var_modelo1, listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,C=C_binaria_1, label = "Modelo 1")

#    Modelo 2
C_binaria_2 <- expand.grid(C=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10))


svm_bin_2 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                           listconti = var_modelo2, listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,C=C_binaria_2, label = "Modelo 2")

#-- Analizamos los resultados obtenidos
#   Modelo 1
svm_5_rep <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_lineal"))
svm_5_rep$C <- as.numeric(svm_5_rep$C)

p <- ggplot(svm_5_rep_v2[svm_5_rep_v2$Modelo == 1, ], aes(x = C, y = Accuracy)) + geom_point() + ggtitle("SVM Lineal Modelo 1") + theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/lineal/svm_lineal_modelo1.png")

q <- ggplot(svm_5_rep_v2[svm_5_rep_v2$Modelo == 2, ], aes(x = C, y = Accuracy)) + geom_point() + ggtitle("SVM Lineal Modelo 2") + theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/lineal/svm_lineal_modelo2.png")

# En el Modelo 1, parecen ser buenas alternativas 0.50, 0.10, 0.05 y 0.01
# En el Modelo 2, parecen ser buenas alternativas 10, 1, 0.05 y 0.01 (tambien interesan valores C altos)
# Por lo general, podemos probar a reducir el valor de C
C_binaria <- expand.grid(C=c(0.001,0.005,0.01))

svm_bin_1_v2 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                              listconti = var_modelo1, listclass=c(""),
                              grupos=5,sinicio=1234,repe=5,C=C_binaria, label = "Modelo 1")

svm_bin_2_v2 <- cruzadaSVMbin(data=surgical_dataset, vardep=target,
                           listconti = var_modelo2, listclass=c(""),
                           grupos=5,sinicio=1234,repe=5,C=C_binaria, label = "Modelo 2")

svm_5_rep_v2 <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_lineal_2"))
svm_5_rep_v2$C <- as.numeric(svm_5_rep_v2$C)

svm_5_rep_v2[svm_5_rep_v2$Modelo == 1, ] %>% ggplot() + geom_point(aes(C, Accuracy)) + ggtitle("SVM Lineal Modelo 1")
ggsave("./charts/SVM/lineal/svm_lineal_modelo1_v2.png")

svm_5_rep_v2[svm_5_rep_v2$Modelo == 2, ] %>% ggplot() + geom_point(aes(C, Accuracy)) + ggtitle("SVM Lineal Modelo 2")
ggsave("./charts/SVM/lineal/svm_lineal_modelo2_v2.png")


# Parece que C = 0.01-0.02 da buenos resultados, en general (nos quedamos con 0.01)

# ------------------------ SVM Polinomial --------------------------
#    Modelo 1
C_poly_1 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
degree_poly_1 <- c(2)
scale_poly_1  <- c(0.1, 0.5, 1, 2, 5)

svm_pol_1 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                               listconti = var_modelo1, listclass=c(""),
                               grupos=5,sinicio=1234,repe=5,C = C_poly_1,
                               degree = degree_poly_1, scale = scale_poly_1,
                               label = "Modelo 1")

#    Modelo 2
# Nota: dado que con grado 3 el proceso se alarga, una vez filtrado el SVM Polinomial
# comparamos los mejores modelos de grado 2 con grado 3
C_poly_2 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
degree_poly_2 <- c(2)
scale_poly_2  <- c(0.1, 0.5, 1, 2, 5)

svm_pol_2 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                               listconti = var_modelo2, listclass=c(""),
                               grupos=5,sinicio=1234,repe=5,C = C_poly_2,
                               degree = degree_poly_2, scale = scale_poly_2,
                               label = "Modelo 2")

svm_poly_5_rep   <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_polinomial"))
svm_poly_5_rep$C <- reorder.factor(svm_poly_5_rep[svm_poly_5_rep$Modelo == 1, "C"], new.order = c(1,2,3,4,5,6,8,9,7))
p <- ggplot(svm_poly_5_rep[svm_poly_5_rep$Modelo == 1, ], aes(x = factor(C), y = Accuracy,
                                                        color = factor(scale), pch = factor(scale))) + 
            geom_point(position = position_dodge(width = 0.5), size = 3) + theme(text = element_text(size=15, face = "bold"))
            ggtitle("SVM Poly Modelo 1 (Grado 2)")
ggsave("./charts/SVM/polinomica/svm_polinomica_modelo1.png")

# Patrones observados
# A excepcion de una escala de 2,5,10, conforme aumenta el factor C de regularizacion, aumenta el valor de Accuracy
# con factor = 0.5-1. En cuanto al resto de escalas comprobamos que el Accuracy comienza a disminuir conforme aumenta C
# Con una escala 2, 5, se obtiene un maximo con un factor menor (entre 0.2 y 1)

q <- svm_poly_5_rep[svm_poly_5_rep$Modelo == 2, ] %>% ggplot(aes(x = factor(C), y = Accuracy,
                                                            color = factor(scale), pch = factor(scale))) + 
                    geom_point(position = position_dodge(width = 0.5), size = 3) + 
                    ggtitle("SVM Poly Modelo 2 (Grado 2)")  + theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/polinomica/svm_polinomica_modelo2.png")

# En relacion al segundo modelo, llama la atencion la necesidad de un factor de escala pequeño (en torno a 0.01 y 0.1)
# Bien es cierto que con una escala de 2 es necesario un valor de 1 (mas alto que el resto), pero por lo general se
# se situan en torno a valores C pequeños, especialmente en torno a una escala de 0.1 e incluso 0.5


# ¿Y si aumentamos el factor C en el modelo 1? Mantenemos escalas de 0.5 y 1
#    Modelo 1
C_poly_1 <- c(5,10,15,20)
degree_poly_1 <- c(2)
scale_poly_1  <- c(0.5, 1, 2)

svm_pol_1_2 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                                 listconti = var_modelo1, listclass=c(""),
                                 grupos=5,sinicio=1234,repe=5,C = C_poly_1,
                                 degree = degree_poly_1, scale = scale_poly_1,
                                 label = "Modelo 1")

C_poly_1 <- c(0.001,0.005)
degree_poly_1 <- c(2)
scale_poly_1  <- c(1, 2, 5)

svm_pol_1_3 <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target,
                                 listconti = var_modelo1, listclass=c(""),
                                 grupos=5,sinicio=1234,repe=5,C = C_poly_1,
                                 degree = degree_poly_1, scale = scale_poly_1)

# ¿Y si reducimos el factor C en el modelo 1? Mantenemos las escalas de 0.1 y 0.5
#    Modelo 2
C_poly_2 <- c(0.001,0.005,0.01,0.05)
degree_poly_2 <- c(2)
scale_poly_2  <- c(0.1, 0.5)

svm_pol_2_2 <- data.frame("id" = c(21, 22, 23, 24, 25, 26),
                          "C" = c(0.001, 0.005, 0.001, 0.005, 0.001, 0.005), 
                          "degree" = rep(2, 6), "scale" = c(1, 1, 2, 2, 5, 5),
                          "Accuracy" = c(0.7713008, 0.7740677, 0.7729747, 0.7752296, 
                                         0.7748536, 0.7758443), "Kappa" = rep(1,6), "AccuracySD" = rep(1,6), "KappaSD" = rep(1,6), "Modelo" = rep(1,6))

# Modelo 1
svm_poly_5_rep_v2   <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_polinomial_2"))
svm_poly_5_rep_v2[, "C"] <- as.numeric(svm_poly_5_rep_v2[, "C"])
svm_poly_5_rep_v2 <- rbind(svm_poly_5_rep_v2, svm_pol_2_2)
p <- ggplot(svm_poly_5_rep_v2[svm_poly_5_rep_v2$Modelo == 1 & svm_poly_5_rep_v2$C > 0.005, ], aes(x = factor(C), y = Accuracy,
                                                         color = factor(scale), pch = factor(scale))) + 
  geom_point(position = position_dodge(width = 0.5), size = 3) + 
  ggtitle("SVM Polinomial Modelo 1 (Grado 2)")  + theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/polinomica/svm_polinomica_modelo1_v2.png")

# Dado que la decision esta basada por patrones y no en valores puntuales, 
# por lo que empleando otra semilla y con otros valores
# training los parametros seran "aproximadamente" similares, aunque no iguales
# Elegimos un factor C = 0.2 y escala = 1

# Modelo 2
q <- ggplot(svm_poly_5_rep_v2[svm_poly_5_rep_v2$Modelo == 2, ], aes(x = factor(C), y = Accuracy,
                                                               color = factor(scale), pch = factor(scale))) + 
  geom_point(position = position_dodge(width = 0.5), size = 3) + 
  ggtitle("SVM Polinomial Modelo 2 (Grado 2)")  + theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/polinomica/svm_polinomica_modelo2_v2.png")

# Podemos comprobar que disminuyendo el factor C no mejora el modelo. De nuevo, la decision esta basada por patrones y no
# en valores puntuales, por lo que empleando otra semilla y con otros valores training los parametros seran
# "aproximadamente" similares, aunque no iguales
# Elegimos un factor C = 0.01 y escala = 0.1

#--- SVM RBF
C_rbf_1 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
sigma_rbf_1  <- c(0.1, 0.5, 1, 2, 5, 10)

svm_rbf_1 <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target,
                              listconti = var_modelo1, listclass=c(""),
                              grupos=5,sinicio=1234,repe=5,C = C_rbf_1,
                              sigma = sigma_rbf_1, label = "Modelo 1")


C_rbf_2 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10)
sigma_rbf_2  <- c(0.1, 0.5, 1, 2, 5, 10)

svm_rbf_2 <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target,
                              listconti = var_modelo2, listclass=c(""),
                              grupos=5,sinicio=1234,repe=5,C = C_rbf_2,
                              sigma = sigma_rbf_2, label = "Modelo 2")

svm_rbf_5_rep   <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_RBF"))
svm_rbf_5_rep$Accuracy <- as.numeric(svm_rbf_5_rep$Accuracy)
svm_rbf_5_rep$C <- as.numeric(svm_rbf_5_rep$C)
svm_rbf_5_rep$sigma <- as.numeric(svm_rbf_5_rep$sigma)

# Modelo 1
p <- ggplot(svm_rbf_5_rep[svm_rbf_5_rep$Modelo == 1, ], aes(x = factor(C), y = Accuracy,
                                                       color = factor(sigma))) + 
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  ggtitle("SVM RBF Modelo 1") +  theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/RBF/svm_rbf_modelo1.png")

# Destaca el patron que presenta, con valores sigmas grandes (entre 5 y 10) y un C entre 0.01 y 0.5 (para sigma = 10)
# y un C entre 0.5 y 2 (para sigma = 5)
# O bien un valor sigma = 2 y C alrededor de 10

t <- ggplot(svm_rbf_5_rep[svm_rbf_5_rep$Modelo == 2, ], aes(x = factor(C), y = Accuracy,
                                                       color = factor(sigma))) + 
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  ggtitle("SVM RBF Modelo 2") +  theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/RBF/svm_rbf_modelo2.png")

# Destaca un parametro sigma grande (entre 5 y 10) y un C entre 0.2 y 1 o bien sigma = 2 y C = 10

# ¿Y si aumentamos el valor sigma? Por lo general se obtienen buenos resultados con valores sigma altos
C_rbf_1 <- c(0.01,0.05,0.1,0.2,0.5,1,2,5,10,15,20,30)
sigma_rbf_1  <- c(2, 5, 10, 20, 30)

svm_rbf_1_2 <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target,
                                listconti = var_modelo1, listclass=c(""),
                                grupos=5,sinicio=1234,repe=5,C = C_rbf_1,
                                sigma = sigma_rbf_1, label = "Modelo 1")

svm_rbf_2_2 <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target,
                                listconti = var_modelo2, listclass=c(""),
                                grupos=5,sinicio=1234,repe=5,C = C_rbf_1,
                                sigma = sigma_rbf_1, label = "Modelo 2")

svm_rbf_5_rep_v2   <- as.data.frame(read_xlsx("./data/modelos_svm.xlsx", sheet = "SVM_RBF_2"))
svm_rbf_5_rep_v2$Accuracy <- as.numeric(svm_rbf_5_rep_v2$Accuracy)
svm_rbf_5_rep_v2$C <- as.numeric(svm_rbf_5_rep_v2$C)

# Modelo 1
p <- ggplot(svm_rbf_5_rep_v2[svm_rbf_5_rep_v2$Modelo == 1 & svm_rbf_5_rep_v2$sigma == 2 & svm_rbf_5_rep_v2$C > 1, ], aes(x = factor(C), y = Accuracy,
                                                       )) + 
            geom_point(position = position_dodge(width = 0.5), size = 3) +
            ggtitle("SVM RBF Modelo 1 (sigma = 2)") +  theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/RBF/svm_rbf_modelo1_v2.png")


# Modelo 2
t <- ggplot(svm_rbf_5_rep_v2[svm_rbf_5_rep_v2$Modelo == 2 & svm_rbf_5_rep_v2$sigma == 2 & svm_rbf_5_rep_v2$C > 1, ], aes(x = factor(C), y = Accuracy,
                                                       )) + 
            geom_point(position = position_dodge(width = 0.5), size = 3) +
            ggtitle("SVM RBF Modelo 2 (sigma = 2)") +  theme(text = element_text(size=15, face = "bold"))

ggsave("./charts/SVM/RBF/svm_rbf_modelo2_v2.png")

# En ambos casos, podemos comprobar que conforme aumenta el valor sigma de 10 (20, 30 por ejemplo),
# el valor de Accuracy disminuye. Por otro lado, conforme aumenta el parametro C, la accuracy disminuye
# por lo que es preferible un valor C en torno a 0.5 y 5

# Modelo 1: en general los valores de sigma deben ser altos (aunque no superiores a 10), por ejemplo
# 5 o 10 y y valores C entre 0.5 y 2-5 (incluso hasta 10 para sigma = 2)
# Aceptamos la recomendacion de caret y elegimos C = 1 y sigma = 5

# Modelo 2: en general, sucede un patron/patrones similar/es a los del modelo 1: es preferible un sigma
# alto, en torno a 5-10 y un parametro C en torno a 0.5 y 2-5, aunque en el caso de sigma = 2 bien es cierto
# que alcanza su maximo en C = 5
# Aunque caret nos recomienda C = 2 y sigma = 0.5, la diferencia es tan pequeña que podemos incluso escoger
# la misma configuracion que que en el primer modelo (C = 1  y sigma = 5)

#-------------------------- Configuracion Final --------------------------------
control<-trainControl(method = "repeatedcv",number=5,repeats=10,
                      savePredictions = "all",classProbs=TRUE, allowParallel = TRUE)

#-- SVM Lineal
set.seed(1234)
SVM_lineal_modelo1     <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmLinear",trControl=control,
                                tuneGrid=expand.grid(C=0.01),replace=TRUE)

SVM_lineal_modelo1_df  <- cruzadaSVMbin(surgical_dataset, vardep=target, listconti=var_modelo1,
                                        listclass = c(""), grupos=5, repe=10, C=0.01, sinicio=1234)

set.seed(1234)
SVM_lineal_modelo2     <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmLinear",trControl=control,
                                tuneGrid=expand.grid(C=0.01),replace=TRUE)

SVM_lineal_modelo2_df  <- cruzadaSVMbin(surgical_dataset, vardep=target, listconti=var_modelo2,
                                        listclass = c(""), grupos=5, repe=10, C=0.01, sinicio=1234)

#-- SVM Polinomial
set.seed(1234)
SVM_polinomial_modelo1 <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmPoly",trControl=control,
                                tuneGrid=expand.grid(C=0.5,degree=2,scale=1),
                                replace=TRUE)

SVM_poly_modelo1_df    <- cruzadaSVMbinPoly(surgical_dataset, vardep=target, listconti=var_modelo1,
                                            listclass = c(""), grupos=5, repe=10, C=0.5, degree=2,
                                            scale=1, sinicio=1234)

set.seed(1234)
SVM_polinomial_modelo2 <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmPoly",trControl=control,
                                tuneGrid=expand.grid(C=0.01,degree=2,scale=0.1),
                                replace=TRUE)

SVM_poly_modelo2_df    <- cruzadaSVMbinPoly(surgical_dataset, vardep=target, listconti=var_modelo2,
                                            listclass = c(""), grupos=5, repe=10, C=0.01, degree=2,
                                            scale=0.1, sinicio=1234)

#-- SVM RBF
set.seed(1234)
SVM_RBF_modelo1        <- train(as.formula(paste0(target,"~",paste0(var_modelo1, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmRadial",trControl=control,
                                tuneGrid=expand.grid(C=0.5,sigma=5),
                                replace=TRUE)

SVM_RBF_modelo1_df     <- cruzadaSVMbinRBF(surgical_dataset, vardep=target, listconti=var_modelo1,
                                           listclass = c(""), grupos=5, repe=10, C=0.5, sigma=5, sinicio=1234)

set.seed(1234)
SVM_RBF_modelo2        <- train(as.formula(paste0(target,"~",paste0(var_modelo2, collapse = "+"))),
                                data=surgical_dataset,
                                method="svmRadial",trControl=control,
                                tuneGrid=expand.grid(C=0.5,sigma=5),
                                replace=TRUE)

SVM_RBF_modelo2_df     <- cruzadaSVMbinRBF(surgical_dataset, vardep=target, listconti=var_modelo2,
                                           listclass = c(""), grupos=5, repe=10, C=0.5, sigma=5, sinicio=1234)

# Analizamos la Tasa de fallos y AUC
svm_10_rep_modelo1 <- rbind(SVM_lineal_modelo1_df, SVM_poly_modelo1_df, SVM_RBF_modelo1_df)
svm_10_rep_modelo1$modelo <- c(rep("lineal", 10), rep("poly", 10), rep("rbf", 10))
svm_10_rep_modelo2 <- rbind(SVM_lineal_modelo2_df, SVM_poly_modelo2_df, SVM_RBF_modelo2_df)
svm_10_rep_modelo2$modelo <- c(rep("lineal", 10), rep("poly", 10), rep("rbf", 10))
svm_10_rep_final   <- rbind(svm_10_rep_modelo1, svm_10_rep_modelo2)
svm_10_rep_final$set <- c(rep("Modelo 1", 30), rep("Modelo 2", 30))

svm_10_rep_final$modelo <- with(svm_10_rep_final,
                                reorder(modelo,tasa, mean))
p <- ggplot(svm_10_rep_final, aes(x = modelo, y = tasa, colour = set)) +
  geom_boxplot() +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos modelos SVM")  +  theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/comparacion_tasa_fallos_modelo1.png")

svm_10_rep_final$modelo <- with(svm_10_rep_final,
                                reorder(modelo,auc, mean))
q <- ggplot(svm_10_rep_final, aes(x = modelo, y = auc, colour = set)) +
  geom_boxplot() +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC modelos SVM")  +  theme(text = element_text(size=15, face = "bold"))
ggsave("./charts/SVM/comparacion_auc_modelo1.png")

ggplot(svm_10_rep_modelo2, aes(x = modelo, y = tasa)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos modelo 2")
ggsave("./charts/SVM/comparacion_tasa_fallos_modelo2.png")

ggplot(svm_10_rep_modelo2, aes(x = modelo, y = auc)) +
  geom_boxplot(fill = "#4271AE", colour = "#1F3552",
               alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC modelo 2")
ggsave("./charts/SVM/comparacion_auc_modelo2.png")

# El modelo RBF, en ambos candidatos, es la mejor opcion

#--- Estadisticas
surgical_test_data <- fread("./data/surgical_test_data.csv", data.table = FALSE)
names(surgical_test_data)[35] <- "target"
surgical_test_data$target <- as.factor(surgical_test_data$target)

#-- SVM Lineal
matriz_conf_svm_lineal_1 <- matriz_confusion_predicciones(SVM_lineal_modelo1, NULL, surgical_test_data, 0.5)
matriz_conf_svm_lineal_2 <- matriz_confusion_predicciones(SVM_lineal_modelo2, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No  Yes
#        No  6234  360
#        Yes 1493  694

# Modelo 2
# print(table(pred_vector)) -> No (8781) No converge con el numero de iteraciones por defecto

matriz_conf_svm_poly_1   <- matriz_confusion_predicciones(SVM_polinomial_modelo1, NULL, surgical_test_data, 0.5)
matriz_conf_svm_poly_2   <- matriz_confusion_predicciones(SVM_polinomial_modelo2, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No  Yes
#        No  6353  241
#        Yes 1568  619

# Modelo 2
# Prediction   No  Yes
#        No  6206  388
#        Yes 1521  666

matriz_conf_svm_rbf_1   <- matriz_confusion_predicciones(SVM_RBF_modelo1, NULL, surgical_test_data, 0.5)
matriz_conf_svm_rbf_2   <- matriz_confusion_predicciones(SVM_RBF_modelo2, NULL, surgical_test_data, 0.5)

# Modelo 1
# Prediction   No  Yes
#        No  6304  290
#        Yes  998 1189

# Modelo 2
# Prediction   No  Yes
#        No  6404  190
#        Yes  869 1318

#-- Comparativa de modelos
modelos_actuales <- as.data.frame(read_excel("./ComparativaModelos.xlsx",
                                             sheet = "svm"))

modelos_actuales$tasa <- as.numeric(modelos_actuales$tasa)
modelos_actuales$auc <- as.numeric(modelos_actuales$auc)
modelos_actuales$tipo <- c(rep("LOGISTICA", 20), rep("RED NEURONAL", 20), rep("BAGGING", 20), rep("RANDOM FOREST", 20),
                           rep("GBM", 20), rep("SVM", 60))

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,tasa, mean))
ggplot(modelos_actuales, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo") + theme(text = element_text(size=15, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))

ggsave('./charts/comparativas/06_log_avnnet_bagging_rf_gbm_svm_tasa.jpeg')

modelos_actuales$modelo <- with(modelos_actuales,
                                reorder(modelo,auc, mean))
ggplot(modelos_actuales, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo") + theme(text = element_text(size=15, face = "bold"), axis.text.x = element_text(angle = 45, vjust = 0.5))

ggsave('./charts/comparativas/06_log_avnnet_bagging_rf_gbm_svm_auc.jpeg')

#---- Estadisticas
# Por tasa fallos --------------- auc
#          svm
#      gbm modelo 1               rf. modelo 1
#      gbm modelo 2           bagging modelo 1
#  bagging modelo 2           bagging modelo 2
#  bagging modelo 1               gbm modelo 1
#   avnnet modelo 2               rf. modelo 2
#      rf. modelo 2            avnnet modelo 1
#      rf. modelo 1               gbm modelo 2
#   avnnet modelo 1            avnnet modelo 2
#   log.   modelo 1            log.   modelo 1
#   log.   modelo 2            log.   modelo 2

#---- Detenemos el cluster
stopCluster(cluster)

#---- Guardamos el fichero RData
save.image(file = "./rdata/SVM.RData")

