#-- Funciones propias

suppressPackageStartupMessages({
  library(DataExplorer)  # EDAs automaticos
  library(gridExtra)     # Representar tablas graficamente
  library(ggplot2)       # Libreria grafica
  library(tidyr)         # Ordenacion datos
  
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
  
  union$modelo <- with(union, reorder(modelo,tasa, mean))
  print(ggplot(union, aes(x = modelo, y = tasa)) +
              geom_boxplot(fill = fill, colour = line,
                           alpha = 0.7) +
              scale_x_discrete(name = "Modelo") +
              ggtitle("Tasa de fallos por modelo"))
  
  union$modelo <- with(union, reorder(modelo,auc, mean))
  print(ggplot(union, aes(x = modelo, y = auc)) +
              geom_boxplot(fill = fill, colour = line,
                           alpha = 0.7) +
              scale_x_discrete(name = "Modelo") +
              ggtitle("AUC por modelo"))
  
  return(union)
}

# Funcion para obtener la matriz de confusion de las predicciones resultantes
matriz_confusion_predicciones <- function(modelo, formula , dataset, corte) {
  
  pred <- predict(modelo, dataset, type = "prob")
  
  pred_vector <- as.factor(ifelse(
    pred$No > corte,
    "No",
    "Yes"
  ))

  matriz_confusion <- confusionMatrix(factor(dataset$target), pred_vector, positive = "Yes")
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
                                         decay=decays[i],repeticiones=5,
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
  legend("top", legend = c("OOB: MODELO 2","OOB: MODELO 1") , 
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
  rfgrid <- expand.grid(mtry=c(mtry))
  control <- trainControl(method = "repeatedcv",number=grupos, repeats=repe,
                          savePredictions = "all",classProbs = TRUE)
                        
  rf<- train(formula,data=dataset,
           method="rf",trControl=control,tuneGrid=rfgrid,
           linout = FALSE,ntree=ntree,nodesize=nodesize,
           replace=TRUE, importance=TRUE)
  
  return(rf)
}

show_vars_importance <- function(modelo, title) {
  final<-modelo$finalModel
  tabla<-as.data.frame(final$importance)
  tabla<-tabla[order(tabla$MeanDecreaseAccuracy),]
  vars <- rownames(tabla)
  tabla$vars <- factor(vars, levels=unique(vars))
  rownames(tabla) <- NULL
  
  print(tabla %>% arrange(.,-MeanDecreaseAccuracy))
  
  tabla %>% arrange(.,-MeanDecreaseAccuracy) %>% 
    ggplot(aes(MeanDecreaseAccuracy, vars)) +
    geom_col() +
    geom_text(aes(label=round(MeanDecreaseAccuracy, 3), x=0.5*MeanDecreaseAccuracy), size=3, colour="white") +
    scale_x_continuous(expand=expansion(c(0,0.04))) +
    ggtitle(title) +
    theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          axis.title=element_blank())
}


review_ntrees <- function(dataset, formula, mtry, ntree, nodesize, seed) {
  
    set.seed(seed)
    rf_modelo <-randomForest(formula,
                                       data=dataset,
                                       mtry=mtry[length(mtry)],ntree=ntree,nodesize=nodesize,replace=TRUE)
    
    p <- data.frame(err.rate.5 = rf_modelo$err.rate[, 1])
    
    plot_vectors <- list()
    for (i in mtry[-length(mtry)]) {
      set.seed(seed)
      rf_modelo <-randomForest(formula,
                               data=dataset,
                               mtry=i,ntree=ntree,nodesize=nodesize,replace=TRUE)
  
      error_rates  <- data.frame(x = rf_modelo$err.rate[,1])
      names(error_rates) <- paste0("err.rate.",i)
      p <- cbind(p, error_rates)
    }
  
    return(p)
}

best_minbucket_dt <- function(dataset, formula, minbuckets, grupos, repe) {
  set.seed(1234)
  control<-trainControl(method = "cv",number=grupos,savePredictions = "all")
  arbolgrid <- expand.grid(cp=c(0))
  
  for (minbu in minbuckets)
  {
    arbolcaret <- train(formula, 
    data=dataset,method="rpart",minbucket=minbu,
    trControl = control, tuneGrid = arbolgrid)
    print(minbu)
    print(arbolcaret)
  }
}





