#-- Funciones propias

suppressPackageStartupMessages({
  library(DataExplorer)  # EDAs automaticos
  library(gridExtra)     # Representar tablas graficamente
  library(ggplot2)       # Libreria grafica
  
  source("./librerias/cruzadas avnnet y log binaria.R")
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
  
  return(union1)
}

# Funcion para obtener la matriz de confusion de las predicciones resultantes
matriz_confusion_predicciones <- function(formula, dataset, corte) {
  
  modelo <- glm(formula,
                data = dataset,
                family = binomial(link="logit")
  )
  pred <- predict(modelo, dataset, type = "response")
  pred_vector <- as.factor(ifelse(
    pred < corte,
    "No",
    "Yes"
  ))
  
  matriz_confusion <- confusionMatrix(dataset$target, pred_vector)
  return(matriz_confusion)
}


