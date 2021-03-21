#-- Funciones propias

suppressPackageStartupMessages({
  library(DataExplorer)  # EDAs automaticos
  library(gridExtra)     # Representar tablas graficamente
  library(ggplot2)       # Libreria grafica
  
  source("./librerias/cruzadas avnnet y log binaria.R")
  source("./librerias/cruzada rf binaria.R")
})

# libreria para EDA
make_eda_report <- function(df, target, title, file_name, path) {
  create_report(df,
                y = target,
                report_title = title,
                output_file = file_name,
                output_dir = path)
}

# tuneo regresion logistica
cruzada_logistica <- function(dataset, target, candidatos, nombres_candidatos, grupos, repe) {
  union  <- data.frame(tasa = numeric(), auc = numeric(), modelo = character())
  
  for (i in seq(1, length(candidatos))) {
    medias <- cruzadalogistica(data=dataset, vardep=target,
                               listconti=candidatos[[i]], listclass=c(""),
                               grupos=grupos,sinicio=1234,repe=repe)
    medias.df <- data.frame(medias[1])
    medias.df$modelo <- nombres_candidatos[i]
    
    union <- rbind(union, medias.df)
    
    print(paste0(nombres_candidatos[i], ": completed!"))
  }
  
  fill <- "#4271AE"
  line <- "#1F3552"
  
  print(ggplot(union, aes(x = modelo, y = tasa)) +
              geom_boxplot(fill = fill, colour = line,
                           alpha = 0.7) +
              scale_x_discrete(name = "Modelo") +
              ggtitle("Tasa de fallos por modelo"))
  
  print(ggplot(union, aes(x = modelo, y = auc)) +
              geom_boxplot(fill = fill, colour = line,
                           alpha = 0.7) +
              scale_x_discrete(name = "Modelo") +
              ggtitle("AUC por modelo"))
  
  return(union)
}

# Funcion para obtener la matriz de confusion de las predicciones resultantes
matriz_confusion_predicciones <- function(modelo = "glm", formula, dataset, corte) {
  
  if (modelo == "glm") {
    modelo <- glm(formula,
                  data = dataset,
                  family = binomial(link="logit")
    )
  }
  pred <- predict(modelo, dataset, type = "prob")
  pred_vector <- as.factor(ifelse(
    pred$No > corte,
    "No",
    "Yes"
  ))
  matriz_confusion <- confusionMatrix(dataset$target, pred_vector)
  return(matriz_confusion)
}

# comparacion modelos redes neuronales
comparar_modelos_red <- function(dataset, target, lista.continua, sizes, 
                                 decays, grupos, repe, iteraciones) {
  
  union  <- data.frame(tasa = numeric(), auc = numeric(), modelo = character())
  for (i in seq(1:length(sizes))) {
    cvnnet.candidato <- cruzadaavnnetbin(data=dataset,vardep=target,
                                         listconti=lista.continua, 
                                         listclass=c(""),
                                         grupos=grupos,sinicio=1234,
                                         repe=repe, size=sizes[i],
                                         decay=decays[i],repeticiones=repe,
                                         itera=iteraciones)
    
    modelo <- paste0("NODOS: ", sizes[i] , " - DECAY: ", decays[i])
    
    medias.df <- data.frame(cvnnet.candidato[1])
    medias.df$modelo <- paste0(modelo)
    
    union <- rbind(union, medias.df)
    
    print(paste0(modelo, ": completed!"))
    
  }
  
  fill <- "#4271AE"
  line <- "#1F3552"
  
  print(ggplot(union, aes(x = modelo, y = tasa)) +
          geom_boxplot(fill = fill, colour = line,
                       alpha = 0.7) +
          scale_x_discrete(name = "Modelo") +
          ggtitle("Tasa de fallos por modelo"))
  
  print(ggplot(union, aes(x = modelo, y = auc)) +
          geom_boxplot(fill = fill, colour = line,
                       alpha = 0.7) +
          scale_x_discrete(name = "Modelo") +
          ggtitle("AUC por modelo"))
  
  return(union)
}

# Funcion para mostrar el err. rate en Bagging
mostrar_err_rate <- function(train.err.rate1, train.err.rate2) {
  plot(train.err.rate1, col = 'red', type = 'l', 
       main = 'Error rate by nº trees', xlab = 'Number of trees', ylab = 'Error rate', ylim = c(0.09, 0.13))
  lines(train.err.rate2, col = 'blue')
  #lines(test.err.rate1, col = 'green')
  #lines(test.err.rate2, col = 'purple')
  abline(h = 0.1)
  legend("topright", legend = c("OOB: 5 variables (train)","OOB: 6 variables (train)") , 
         col = c('red', 'blue') , bty = "n", horiz = FALSE, 
         lty=1, cex = 0.75)
}

# Funcion para el tuneo de un modelo bagging
tuneo_bagging <- function(dataset, target, lista.continua, nodesizes, sampsizes,
                          mtry, ntree, grupos, repe, replace = TRUE) {
  lista.rf <- list()
  for(x in apply(data.frame(expand.grid(nodesizes, sampsizes)),1,as.list)) {
    salida <- cruzadarfbin(data=dataset, vardep=target,
                           listconti=lista.continua,
                           listclass=c(""),
                           grupos=grupos,sinicio=1234,repe=repe,nodesize=x$Var1,
                           mtry=mtry,ntree=ntree, sampsize=x$Var2, replace = replace)
    cat(x$Var1, "+",  x$Var2 , "-> FINISHED\n")
    salida$modelo <- paste0(x$Var1, "+",  x$Var2)
    lista.rf <- c(lista.rf, list(salida))
  }
  
  union <- do.call(rbind.data.frame, lista.rf)
  
  fill <- "#4271AE"
  line <- "#1F3552"
  print(colnames(union))
  print(ggplot(union, aes(x = modelo, y = tasa)) +
          geom_boxplot(fill = fill, colour = line,
                       alpha = 0.7) +
          scale_x_discrete(name = "Modelo") +
          ggtitle("Tasa de fallos por modelo"))
  
  print(ggplot(union, aes(x = modelo, y = auc)) +
          geom_boxplot(fill = fill, colour = line,
                       alpha = 0.7) +
          scale_x_discrete(name = "Modelo") +
          ggtitle("AUC por modelo"))
  
  return(union)
}

train_rf_model <- function(dataset, formula, mtry, ntree, grupos, repe,
                           nodesize, seed) {
  set.seed(seed)
  rfgrid <- expand.grid(mtry=mtry)
  control <- trainControl(method = "repeatedcv",number=grupos, repeats=repe,
                          savePredictions = "all",classProbs = TRUE)
                        
  rf<- train(formula,data=dataset,
           method="rf",trControl=control,tuneGrid=rfgrid,
           linout = FALSE,ntree=ntree,nodesize=nodesize,replace=TRUE,
           importance=TRUE)
  
  return(rf)
}














