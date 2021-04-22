# ------------- Ensamblado ---------------
# Objetivo: elaborar el mejor modelo de ensamblado en base
#           a los mejores modelos obtenidos anteriormente
# Autor: Alberto Fernandez Hernandez

# Elegimos la seleccion de variables nº 2
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
  library(corrplot)      # Matriz de correlacion (grafico)
  
  source("./librerias/cruzadas ensamblado binaria fuente.R")
})

# Funcion para calcular la tasa de fallos
tasafallos<-function(x,y) {
  confu<-confusionMatrix(x,y)
  tasa<-confu[[3]][1]
  return(tasa)
}

# Funcion para calcular el AUC
auc<-function(x,y) {
  curvaroc<-roc(response=x,predictor=y)
  auc<-curvaroc$auc
  return(auc)
}

#--- Creamos el cluster
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

#--- Lectura dataset depurado
surgical_dataset <- fread("./data/surgical_dataset_final.csv", data.table = FALSE)
surgical_dataset$target <- as.factor(surgical_dataset$target)

# Separamos variable objetivo del resto
target <- "target"

#-- Modelo 2
var_modelo1 <- c("mortality_rsi", "ccsMort30Rate", "bmi", "month.8", "Age")
var_modelo2 <- c("mortality_rsi", "bmi", "month.8", "Age")

#-- Parametros generales: semilla, numero de grupos y repeticiones
sinicio <- 1234; grupos <- 5; repe <- 10

# Importante: para el ensamblado no vamos a unir algoritmos "malos", sino las mejores versiones de cada
# uno de ellos. Ademas, se benefician de aquellos algoritmos que no presentan una alta correlación entre si.
# Ejemplo: redes neuronales y un modelo random forest (se benefician entre si ya que son modelos con una
# naturaleza diferente que un modelo bagging y un modelo random forest, donde no podria reducirse tanto la 
# varianza).

#-- Tuneo de los modelos finales
#   Regresion logistica
logistica <- cruzadalogistica(data=surgical_dataset, vardep=target,
                              listconti=var_modelo2, listclass=c(""),
                              grupos=grupos,sinicio=1234,repe=repe)

medias_logistica    <- as.data.frame(logistica[1])
medias_logistica$modelo <-"Logistica"
pred_logistica      <- as.data.frame(logistica[2])
pred_logistica$logi <- pred_logistica$Yes

#  Avnnet
avnnet  <- cruzadaavnnetbin(data=surgical_dataset,vardep=target,
                             listconti=var_modelo2, listclass=c(""),
                             grupos=grupos,sinicio=sinicio,repe=repe, 
                             size=10,decay=0.01,repeticiones=5,itera=200)

medias_avnnet    <- as.data.frame(avnnet[1])
medias_avnnet$modelo <-"Avnnet"
pred_avnnet      <- as.data.frame(avnnet[2])
pred_avnnet$avnnet <- pred_avnnet$Yes

#  Bagging
bagging <- cruzadarfbin(data=surgical_dataset, vardep=target,
                        listconti=var_modelo2,listclass=c(""),
                        grupos=grupos,sinicio=sinicio,repe=repe,nodesize=20,
                        mtry=4,ntree=900, sampsize=1000, replace = TRUE)

medias_bagging    <- as.data.frame(bagging[1])
medias_bagging$modelo <-"Bagging"
pred_bagging      <- as.data.frame(bagging[2])
pred_bagging$bagging <- pred_bagging$Yes

#  Random Forest
random_forest <- cruzadarfbin(data=surgical_dataset,
                              vardep=target,listconti=var_modelo2,
                              listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                              mtry=2,ntree=2000,nodesize=20,replace=TRUE)

medias_random_forest    <- as.data.frame(random_forest[1])
medias_random_forest$modelo <-"Random_Forest"
pred_random_forest      <- as.data.frame(random_forest[2])
pred_random_forest$rf <- pred_random_forest$Yes

# Gradient Boosting
gradient_boosting <- cruzadagbmbin(data=surgical_dataset,
                                   vardep=target,listconti=var_modelo2,
                                   listclass=c(""),grupos=grupos,sinicio=sinicio,repe=repe,
                                   n.minobsinnode=20,shrinkage=0.2,n.trees=100,
                                   interaction.depth=2, bag.fraction=0.5)

medias_gradient_boosting    <- as.data.frame(gradient_boosting[1])
medias_gradient_boosting$modelo <-"Gradient_Boosting"
pred_gradient_boosting      <- as.data.frame(gradient_boosting[2])
pred_gradient_boosting$gbm <- pred_gradient_boosting$Yes

# XGboost
xgboost <- cruzadaxgbmbin(data=surgical_dataset,vardep=target,
           listconti=var_modelo2,listclass=c(""),
           grupos=grupos,sinicio=sinicio,repe=repe,
           min_child_weight=20,eta=0.1,nrounds=100,max_depth=6,
           gamma=0,colsample_bytree=1,subsample=0.5)

medias_xgboost    <- as.data.frame(xgboost[1])
medias_xgboost$modelo <-"XGboost"
pred_xgboost      <- as.data.frame(xgboost[2])
pred_xgboost$xgboost <- pred_xgboost$Yes

svm_lineal <- cruzadaSVMbin(data=surgical_dataset, vardep=target, listconti=var_modelo1,
                            listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                            C=0.01, replace=TRUE)

medias_svm_lineal    <- as.data.frame(svm_lineal[1])
medias_svm_lineal$modelo <-"SVM_Lin"
pred_svm_lineal      <- as.data.frame(svm_lineal[2])
pred_svm_lineal$svmlin <- pred_svm_lineal$Yes

svm_poly   <- cruzadaSVMbinPoly(data=surgical_dataset, vardep=target, listconti=var_modelo2,
                            listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                            C=0.01, degree=2, scale=0.1)

medias_svm_poly    <- as.data.frame(svm_poly[1])
medias_svm_poly$modelo <-"SVM_Poly"
pred_svm_poly      <- as.data.frame(svm_poly[2])
pred_svm_poly$svmpoly <- pred_svm_poly$Yes

svm_rbf   <- cruzadaSVMbinRBF(data=surgical_dataset, vardep=target, listconti=var_modelo2,
                              listclass=c(""), grupos=grupos, sinicio=sinicio, repe=repe,
                              C=1, sigma=5)

medias_svm_rbf    <- as.data.frame(svm_rbf[1])
medias_svm_rbf$modelo <-"SVM_RBF"
pred_svm_rbf      <- as.data.frame(svm_rbf[2])
pred_svm_rbf$svmrbf <- pred_svm_rbf$Yes

#-- Construccion de los modelos de ensamblado
unipredi <- cbind(pred_logistica,pred_avnnet,pred_bagging,pred_random_forest,
                  pred_gradient_boosting,pred_xgboost,pred_svm_lineal,pred_svm_poly, pred_svm_rbf)
#  Eliminamos columnas duplicadas
unipredi <- unipredi[, !duplicated(colnames(unipredi))]

#-- Analisis previo (nos quedamos con la primera repeticion)
unigraf <- unipredi[unipredi$Rep == "Rep01", ]

#-- Analisis de correlacion entre predicciones de cada algoritmo individual
modelos  <- c("logi","avnnet", "bagging", "rf", "gbm", "xgboost", "svmlin", "svmpoly", "svmrbf")

mat <- unigraf[,modelos] 
matrizcorr <- cor(mat)
colmat <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(matrizcorr, method="color", col=colmat(200),  
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         p.mat = matrizcorr, sig.level = 0.99,
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)

#-- Comentarios: cabe destacar que el modelo de regresion logistica presenta una correlación moderada con 
#   la mayoría de los modelos, a excepción del kernel SVM Lineal.
#   En relacion al resto de parametros, el kernel polinomial presenta una correlacion moderada con la mayoria
#   de los modelos
#   Los modelos svmrbf, avnnet, xgboost, gbm, bagging y rf presenta una alta correlacion entre si

#-- Combinacion modelos con logistica
modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmlin", "svmpoly", "svmrbf")

for(modelo in modelos) {
  nombre_modelo  <- paste0("logi-",modelo)
  unipredi[, nombre_modelo] <- (unipredi[, modelo] + unipredi[, "logi"]) / 2
}

#-- Combinacion entre los 6 mejores modelos (excluimos logistica y polinomial)
modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmlin", "svmpoly", "svmrbf")

for(modelo in modelos) {
  for(modelo_aux in setdiff(modelos, modelo)) {
    nombre_modelo  <- paste0(modelo,"-",modelo_aux)
    cat("Baking... ", nombre_modelo)
    unipredi[, nombre_modelo] <- (unipredi[, modelo] + unipredi[, modelo_aux]) / 2
  }
  modelos <- modelos[!modelos %in% c(modelo)]
}

# Listado de modelos a considerar
dput(names(unipredi))

listado           <- c("logi","avnnet","bagging", "rf", "gbm", "xgboost", "svmlin", 
                       "svmpoly", "svmrbf", "logi-avnnet", "logi-bagging", "logi-rf", 
                       "logi-gbm", "logi-xgboost", "logi-svmlin", "logi-svmpoly", 
                       "logi-svmrbf", "avnnet-bagging", "avnnet-rf", "avnnet-gbm", "avnnet-xgboost", 
                       "avnnet-svmlin", "avnnet-svmpoly", "avnnet-svmrbf", "bagging-rf", 
                       "bagging-gbm", "bagging-xgboost", "bagging-svmlin", "bagging-svmpoly", 
                       "bagging-svmrbf", "rf-gbm", "rf-xgboost", "rf-svmlin", "rf-svmpoly", 
                       "rf-svmrbf", "gbm-xgboost", "gbm-svmlin", "gbm-svmpoly", "gbm-svmrbf", 
                       "xgboost-svmlin", "xgboost-svmpoly", "xgboost-svmrbf", "svmlin-svmpoly", 
                       "svmlin-svmrbf", "svmpoly-svmrbf")

# Cambio a Yes, No, todas las predicciones
repeticiones<-nlevels(factor(unipredi$Rep))
unipredi$Rep<-as.factor(unipredi$Rep)
unipredi$Rep<-as.numeric(unipredi$Rep)


medias0<-data.frame(c())
for (prediccion in listado)
{
  unipredi$proba<-unipredi[,prediccion]
  unipredi[,prediccion]<-ifelse(unipredi[,prediccion]>0.5,"Yes","No")
  for (repe in 1:repeticiones)
  {
    paso <- unipredi[(unipredi$Rep==repe),]
    pre<-factor(paso[,prediccion])
    archi<-paso[,c("proba","obs")]
    archi<-archi[order(archi$proba),]
    obs<-paso[,c("obs")]
    tasa=1-tasafallos(pre,obs)
    t<-as.data.frame(tasa)
    t$modelo<-prediccion
    auc<-suppressMessages(auc(archi$obs,archi$proba))
    t$auc<-auc
    medias0<-rbind(medias0,t)
  }
}

medias0$tipo <- c(rep("Logistica", 10),
                  rep("Original",  80),
                  rep("Logistica", 80),
                  rep("Ensamblado", 280))



medias0$modelo <- with(medias0,
                       reorder(modelo,tasa, mean))
ggplot(medias0, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias0$modelo <- with(medias0,
                       reorder(modelo,auc, mean))
ggplot(medias0, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

#-- Comentarios:
#   SVM Lineal y SVM Polinomial no aportan mejorias a los modelos de Ensamblado, mientras que RBF se situa entre los
#   mejores modelos de ensamblado
#   La logistica combina bien con modelos de arboles como bagging o random forest
#-- Con los detalles comentados, aumentamos a tres modelos por ensamblado

unipredi_2 <- cbind(pred_logistica,pred_avnnet,pred_bagging,pred_random_forest,
                  pred_gradient_boosting,pred_xgboost,pred_svm_lineal,pred_svm_poly, pred_svm_rbf)

#  Eliminamos columnas duplicadas
unipredi_2 <- unipredi_2[, !duplicated(colnames(unipredi_2))]

#-- Combinacion modelos con logistica
modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmrbf")

for(modelos in combn(modelos, m=2, simplify = FALSE)) {
  nombre_modelo  <- paste0("logi-",modelos[1], "-", modelos[2])
  print(nombre_modelo)
  unipredi_2[, nombre_modelo] <- (unipredi_2[, modelos[1]] + unipredi_2[, modelos[2]] + unipredi_2[, "logi"]) / 3
}

#-- Combinacion entre los 6 mejores modelos (excluimos logistica y polinomial)
modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmrbf")

for(modelos in combn(modelos, m=3, simplify = FALSE)) {
  nombre_modelo  <- paste0(modelos[1],"-",modelos[2],"-",modelos[3])
  print(nombre_modelo)
  unipredi_2[, nombre_modelo] <- (unipredi_2[, modelos[1]] + unipredi_2[, modelos[2]] + unipredi_2[, modelos[3]]) / 3
}

# Cambio a Yes, No, todas las predicciones
repeticiones<-nlevels(factor(unipredi_2$Rep))
unipredi_2$Rep<-as.factor(unipredi_2$Rep)
unipredi_2$Rep<-as.numeric(unipredi_2$Rep)

dput(names(unipredi_2))

listado           <- c("logi", "avnnet", "bagging", "rf", "gbm", "xgboost", "svmlin", 
                       "svmpoly", "svmrbf", "logi-avnnet-bagging", 
                       "logi-avnnet-rf", "logi-avnnet-gbm", "logi-avnnet-xgboost", "logi-avnnet-svmrbf", 
                       "logi-bagging-rf", "logi-bagging-gbm", "logi-bagging-xgboost", 
                       "logi-bagging-svmrbf", "logi-rf-gbm", "logi-rf-xgboost", "logi-rf-svmrbf", 
                       "logi-gbm-xgboost", "logi-gbm-svmrbf", "logi-xgboost-svmrbf", 
                       "avnnet-bagging-rf", "avnnet-bagging-gbm", "avnnet-bagging-xgboost", 
                       "avnnet-bagging-svmrbf", "avnnet-rf-gbm", "avnnet-rf-xgboost", 
                       "avnnet-rf-svmrbf", "avnnet-gbm-xgboost", "avnnet-gbm-svmrbf", 
                       "avnnet-xgboost-svmrbf", "bagging-rf-gbm", "bagging-rf-xgboost", 
                       "bagging-rf-svmrbf", "bagging-gbm-xgboost", "bagging-gbm-svmrbf", 
                       "bagging-xgboost-svmrbf", "rf-gbm-xgboost", "rf-gbm-svmrbf", 
                       "rf-xgboost-svmrbf", "gbm-xgboost-svmrbf")

medias1<-data.frame(c())
for (prediccion in listado)
{
  unipredi_2$proba<-unipredi_2[,prediccion]
  unipredi_2[,prediccion]<-ifelse(unipredi_2[,prediccion]>0.5,"Yes","No")
  for (repe in 1:repeticiones)
  {
    paso <- unipredi_2[(unipredi_2$Rep==repe),]
    pre<-factor(paso[,prediccion])
    archi<-paso[,c("proba","obs")]
    archi<-archi[order(archi$proba),]
    obs<-paso[,c("obs")]
    tasa=1-tasafallos(pre,obs)
    t<-as.data.frame(tasa)
    t$modelo<-prediccion
    auc<-suppressMessages(auc(archi$obs,archi$proba))
    t$auc<-auc
    medias1<-rbind(medias1,t)
  }
}

medias1$tipo <- c(rep("Logistica", 10),
                  rep("Original",  80),
                  rep("Logistica", 150),
                  rep("Ensamblado", 200))



medias1$modelo <- with(medias1,
                       reorder(modelo,tasa, mean))
ggplot(medias1, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias1$modelo <- with(medias1,
                       reorder(modelo,auc, mean))
ggplot(medias1, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

#-- Pese a añadir mas de 3 modelos por ensamblado, continuan siendo mejor los modelos de ensamblado
#   Podemos aumentar a 4 grupos, juntando xgboost, gbm, bagging y rf
unipredi_3 <- cbind(pred_logistica,pred_avnnet,pred_bagging,pred_random_forest,
                    pred_gradient_boosting,pred_xgboost,pred_svm_lineal,pred_svm_poly, pred_svm_rbf)

#  Eliminamos columnas duplicadas
unipredi_3 <- unipredi_3[, !duplicated(colnames(unipredi_3))]
unipredi_3[, "xgboost-gbm-bagging-rf"] <- (unipredi_3[, "xgboost"] + unipredi_3[, "gbm"] + unipredi_3[, "bagging"] + unipredi_3[, "rf"]) / 4



medias_final <- rbind(
  medias0[medias0$modelo %in% c("logi", "avnnet", "bagging", "rf", "gbm", "xgboost", "svmlin", "svmpoly", "svmrbf",
                                "bagging-rf", "rf-xgboost"), ],
  medias1[medias1$modelo %in% c("bagging-rf-gbm", "bagging-rf-xgboost"), ]
)

# Cambio a Yes, No, todas las predicciones
repeticiones<-nlevels(factor(unipredi_3$Rep))
unipredi_3$Rep<-as.factor(unipredi_3$Rep)
unipredi_3$Rep<-as.numeric(unipredi_3$Rep)

dput(names(unipredi_3))

listado           <- c("xgboost-gbm-bagging-rf")

medias2<-data.frame(c())
for (prediccion in listado)
{
  unipredi_3$proba<-unipredi_3[,prediccion]
  unipredi_3[,prediccion]<-ifelse(unipredi_3[,prediccion]>0.5,"Yes","No")
  for (repe in 1:repeticiones)
  {
    paso <- unipredi_3[(unipredi_3$Rep==repe),]
    pre<-factor(paso[,prediccion])
    archi<-paso[,c("proba","obs")]
    archi<-archi[order(archi$proba),]
    obs<-paso[,c("obs")]
    tasa=1-tasafallos(pre,obs)
    t<-as.data.frame(tasa)
    t$modelo<-prediccion
    auc<-suppressMessages(auc(archi$obs,archi$proba))
    t$auc<-auc
    medias2<-rbind(medias2,t)
  }
}

medias_final <- rbind(medias_final[, -4], medias2)

medias_final$tipo <- c(rep("Original", 90), rep("Ensamblado (2)", 20),
                       rep("Ensamblado (3)", 20), rep("Ensamblado (4)", 10))

medias_final$modelo <- with(medias_final,
                       reorder(modelo,tasa, mean))
ggplot(medias_final, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias_final$modelo <- with(medias_final,
                       reorder(modelo,auc, mean))
ggplot(medias_final, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))


#-- Hagamos zoom sobre los modelos Ensamblado...
medias_final$modelo <- with(medias_final,
                            reorder(modelo,tasa, mean))
ggplot(medias_final[!medias_final$tipo %in% c("Original"), ], aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias_final$modelo <- with(medias_final,
                            reorder(modelo,auc, mean))
ggplot(medias_final[!medias_final$tipo %in% c("Original"), ], aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

#-- Entre los modelos de ensamblado no hay mucha diferencia
#   Podemos quedarnos con
#   rf-xgboost
#   bagging-rf-xgboost
#   xgboost-gbm-bagging-rf

#-- Sin embargo, tenemos una posibilidad de poder mejorar los modelos de regresion logistica
#   Asignando pesos a las predicciones
medias_final_2 <- medias_final


#-- Veamos los dos modelos de ensamblado

qplot(bagging,rf,data=unigraf,colour=obs)+
  geom_hline(yintercept=0.5, color="black", size=1)+
  geom_vline(xintercept=0.5, color="black", size=1)

qplot(rf,xgboost,data=unigraf,colour=obs)+
  geom_hline(yintercept=0.5, color="black", size=1)+
  geom_vline(xintercept=0.5, color="black", size=1)

unipredi_4 <- cbind(pred_logistica,pred_avnnet,pred_bagging,pred_random_forest,
                    pred_gradient_boosting,pred_xgboost,pred_svm_lineal,pred_svm_poly, pred_svm_rbf)

#  Eliminamos columnas duplicadas
unipredi_4 <- unipredi_4[, !duplicated(colnames(unipredi_4))]

modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmrbf", "svmlin", "svmpoly")

for(modelo in modelos) {
  nombre_modelo  <- paste0("logi-",modelo)
  print(nombre_modelo)
  unipredi_4[, nombre_modelo] <- (unipredi_4[, modelo] * 0.8 + unipredi_4[, "logi"] * 0.2)
}

# Cambio a Yes, No, todas las predicciones
repeticiones<-nlevels(factor(unipredi_4$Rep))
unipredi_4$Rep<-as.factor(unipredi_4$Rep)
unipredi_4$Rep<-as.numeric(unipredi_4$Rep)

dput(names(unipredi_4))

listado           <- c("logi-avnnet", "logi-bagging", "logi-rf", "logi-gbm", "logi-xgboost", 
                       "logi-svmrbf", "logi-svmlin", "logi-svmpoly")

medias3<-data.frame(c())
for (prediccion in listado)
{
  unipredi_4$proba<-unipredi_4[,prediccion]
  unipredi_4[,prediccion]<-ifelse(unipredi_4[,prediccion]>0.5,"Yes","No")
  for (repe in 1:repeticiones)
  {
    paso <- unipredi_4[(unipredi_4$Rep==repe),]
    pre<-factor(paso[,prediccion])
    archi<-paso[,c("proba","obs")]
    archi<-archi[order(archi$proba),]
    obs<-paso[,c("obs")]
    tasa=1-tasafallos(pre,obs)
    t<-as.data.frame(tasa)
    t$modelo<-prediccion
    auc<-suppressMessages(auc(archi$obs,archi$proba))
    t$auc<-auc
    medias3<-rbind(medias3,t)
  }
}

medias_final_2 <- rbind(medias_final_2[, -4], medias3)

medias_final_2$tipo <- c(rep("Original", 90), rep("Ensamblado (2)", 20),
                       rep("Ensamblado (3)", 20), rep("Ensamblado (4)", 10),
                       rep("Weighted Average", 80))

medias_final_2$modelo <- with(medias_final_2,
                            reorder(modelo,tasa, mean))
ggplot(medias_final_2, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias_final_2$modelo <- with(medias_final_2,
                            reorder(modelo,auc, mean))
ggplot(medias_final_2, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

#-- ¿Y empleando tres modelos?
unipredi_5 <- cbind(pred_logistica,pred_avnnet,pred_bagging,pred_random_forest,
                    pred_gradient_boosting,pred_xgboost,pred_svm_lineal,pred_svm_poly, pred_svm_rbf)

#  Eliminamos columnas duplicadas
unipredi_5 <- unipredi_5[, !duplicated(colnames(unipredi_5))]

modelos  <- c("avnnet", "bagging", "rf", "gbm", "xgboost", "svmrbf", "svmlin", "svmpoly")

for(modelo in combn(modelos, m=2, simplify = FALSE)) {
  nombre_modelo  <- paste0("logi-",modelo[1],"-",modelo[2])
  print(nombre_modelo)
  unipredi_5[, nombre_modelo] <- unipredi_5[, modelo[1]] * 0.35 + unipredi_5[, modelo[2]] * 0.35 + unipredi_5[, "logi"] * 0.2
}

# Cambio a Yes, No, todas las predicciones
repeticiones<-nlevels(factor(unipredi_5$Rep))
unipredi_5$Rep<-as.factor(unipredi_5$Rep)
unipredi_5$Rep<-as.numeric(unipredi_5$Rep)

dput(names(unipredi_5))

listado           <- c("logi-avnnet-bagging", 
                       "logi-avnnet-rf", "logi-avnnet-gbm", "logi-avnnet-xgboost", "logi-avnnet-svmrbf", 
                       "logi-avnnet-svmlin", "logi-avnnet-svmpoly", "logi-bagging-rf", 
                       "logi-bagging-gbm", "logi-bagging-xgboost", "logi-bagging-svmrbf", 
                       "logi-bagging-svmlin", "logi-bagging-svmpoly", "logi-rf-gbm", 
                       "logi-rf-xgboost", "logi-rf-svmrbf", "logi-rf-svmlin", "logi-rf-svmpoly", 
                       "logi-gbm-xgboost", "logi-gbm-svmrbf", "logi-gbm-svmlin", "logi-gbm-svmpoly", 
                       "logi-xgboost-svmrbf", "logi-xgboost-svmlin", "logi-xgboost-svmpoly", 
                       "logi-svmrbf-svmlin", "logi-svmrbf-svmpoly", "logi-svmlin-svmpoly")

medias4<-data.frame(c())
for (prediccion in listado)
{
  unipredi_5$proba<-unipredi_5[,prediccion]
  unipredi_5[,prediccion]<-ifelse(unipredi_5[,prediccion]>0.5,"Yes","No")
  for (repe in 1:repeticiones)
  {
    paso <- unipredi_5[(unipredi_5$Rep==repe),]
    pre<-factor(paso[,prediccion])
    archi<-paso[,c("proba","obs")]
    archi<-archi[order(archi$proba),]
    obs<-paso[,c("obs")]
    tasa=1-tasafallos(pre,obs)
    t<-as.data.frame(tasa)
    t$modelo<-prediccion
    auc<-suppressMessages(auc(archi$obs,archi$proba))
    t$auc<-auc
    medias4<-rbind(medias4,t)
  }
}

medias_final_3 <- rbind(medias_final[, -4], medias4)

medias_final_3$tipo <- c(rep("Original", 90), rep("Ensamblado (2)", 20),
                         rep("Ensamblado (3)", 20), rep("Ensamblado (4)", 10),
                         rep("Weighted Average", 280))

medias_final_3$modelo <- with(medias_final_3,
                              reorder(modelo,tasa, mean))
ggplot(medias_final_3, aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias_final_3$modelo <- with(medias_final_3,
                              reorder(modelo,auc, mean))
ggplot(medias_final_3, aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

#-- Entre los modelos de ensamblado no hay mucha diferencia
#   Podemos quedarnos con
#   rf-xgboost
#   bagging-rf-xgboost
#   xgboost-gbm-bagging-rf
medias_final$modelo <- with(medias_final,
                            reorder(modelo,tasa, mean))
ggplot(medias_final[!medias_final$tipo %in% c("Original"), ], aes(x = modelo, y = tasa, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("Tasa de fallos por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

medias_final$modelo <- with(medias_final,
                            reorder(modelo,auc, mean))
ggplot(medias_final[!medias_final$tipo %in% c("Original"), ], aes(x = modelo, y = auc, col = tipo)) +
  geom_boxplot(alpha = 0.7) +
  scale_x_discrete(name = "Modelo") +
  ggtitle("AUC por modelo (modelos originales + Ensamblado)") + 
  theme(axis.text.x = element_text(angle = 45))

