setwd("/Users/alberto/UCM/Machine Learning/Practica ML/")
library(corrplot)
library(readxl)
library(ggplot2)
library(dplyr)

telco.data <- read.csv("teleco.csv")
dim(telco.data) # 7043 filas y 21 columnas

dput(names(telco.data))

# 1. Eliminacion de columnas
library(DataExplorer)
create_report(telco.data)

# El nombre del exoplaneta tambien debemos descartarlo, ya que solo los planetas marcados como "CONFIRMED" presentan nombre, por lo que no aporta informacion alguna al modelo
length(unique(telco.data$customerID))
telco.data <- telco.data[, !colnames(telco.data) %in% c("customerID")]

dim(telco.data) # 7043 filas y 20 columnas

summary(telco.data)

# 2. RECATEGORIZACION DE VARIABLES
apply(telco.data, 2, function(x) {length(unique(x))})
telco.data[,c(1:17,20)] <- lapply(telco.data[, c(1:17,20)], factor)
summary(telco.data[, c(1:17,20)])

target<-"Churn"
factores<-names(Filter(is.factor, telco.data))[-18]
numericas<-names(Filter(is.numeric, telco.data))

# 3. EDA
# Variables cualitativas
mosaico_targetbinaria<-function(var,target,nombreEje,sort=2:1){
  ds <- table(var, target)
  ord <- order(apply(ds, 1, sum), decreasing=TRUE)
  mosaicplot(ds[ord,], color=c("darkturquoise","indianred1"), main="",xlab=nombreEje)
}

for(columna in factores) {
  mosaico_targetbinaria(telco.data[, columna], telco.data[, "Churn"],columna)
}



tabla <- table(telco.data$tenure, telco.data$Churn)
tabla.orden <- order(tabla[,2])
barplot(tabla[, 2], type = "l", xlab = "Month", ylab = "Churn", col = "red")


library(scorecard)
salida.woe <- woebin(telco.data, y = "Churn", positive = "Yes")
woebin_plot(salida.woe)

# Podemos descartar determinadas columnas
telco.data$MultipleLines <- NULL
telco.data$PhoneService <- NULL
telco.data$gender <- NULL

factores <- setdiff(factores, c("MultipleLines", "PhoneService", "gender"))

# Variables cuantitativas
grafico_barras_cuantitativa <- function(columna) {
  ggplot(telco.data, aes_string(x = columna)) + geom_histogram(aes(color = Churn, fill = Churn),
                                                  alpha = 0.4, position = "identity") +
    scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
    scale_color_manual(values = c("#00AFBB", "#E7B800"))
}

for(columna in numericas) {
  print(grafico_barras_cuantitativa(columna))
}

# 3. VALORES MISSING
# Observar gráficamente missing y estructura
library(naniar)
gg_miss_var(telco.data)

# Observaciones missing por variables
tablamis.numericas<-as.data.frame(sapply(telco.data[,numericas],function(x) sum(is.na(x))))
names(tablamis.numericas)[1]<-"nmiss"
max(tablamis.numericas)/nrow(telco.data) # 0.1 % (en el peor de los casos)

tablamis.factores<-as.data.frame(sapply(telco.data[,factores],function(x) sum(is.na(x))))
names(tablamis.factores)[1]<-"nmiss"
max(tablamis.factores)/ nrow(telco.data) # Existen variables que llegan al 72 % (en el peor de los casos)

# La media NO es representativa
psych::describe(Filter(is.numeric, telco.data))

# Imputaciones
ImputacionCuant<-function(vv,tipo){
  if (tipo=="mediana"){
    vv[is.na(vv)]<-round(median(vv,na.rm=T),4)
  } else if (tipo=="media"){
    vv[is.na(vv)]<-round(mean(vv,na.rm=T),4)
  }
  vv
}
telco.data[, numericas] <- sapply(Filter(is.numeric, telco.data),function(x) ImputacionCuant(x,"mediana"))
sum(is.na(telco.data))


# 5. RE-CATEGORIZACION DE VARIABLES CUALITATIVAS
# tenure
DescTools::Freq(telco.data$tenure)
mosaico_targetbinaria(telco.data$tenure, telco.data$Churn, "Churn")

telco.data$tenure <- car::recode(telco.data$tenure, "seq(0,5)='0-5'; seq(6,25)='6-25'; seq(26,43)='26-43'; else='44+'")

DescTools::Freq(telco.data$tenure)
mosaico_targetbinaria(telco.data$tenure, telco.data$Churn, "Churn")

# 5. TRANSFORMACION DE VARIABLES
telco.data.copia <- telco.data

library(bestNormalize)
vector.mejor.lambda <- c()
for(col in numericas) {
  mejor.lambda <- yeojohnson(unlist(telco.data.copia[, col]))$lambda
  vector.mejor.lambda <- c(vector.mejor.lambda, mejor.lambda)
  telco.data.copia[, col] <- VGAM::yeo.johnson(unlist(telco.data[, col]), lambda = mejor.lambda)
}
data.frame("columna" = numericas, "lambda" = vector.mejor.lambda)

create_report(telco.data.copia, output_file = "datos_telco_imputados_transformados")

salida.woe.copia <- woebin(telco.data.copia, "Churn", print_step = 0, positive = "Yes")
sapply(salida.woe.copia[numericas], function(x) x$total_iv[1]) - 
          sapply(salida.woe[numericas], function(x) x$total_iv[1])
psych::describe(Filter(is.numeric, telco.data))
psych::describe(Filter(is.numeric, telco.data.copia))

columnas.transformadas <- c("TotalCharges")
telco.data[, columnas.transformadas] <- telco.data.copia[, columnas.transformadas]
rm(telco.data.copia);

# 6. ESTANDARIZACION DE VARIABLES CONTINUAS
media <- sapply(telco.data[, numericas], mean)
desv.tipica <- sapply(telco.data[, numericas], sd)
telco.data[, numericas] <- scale(telco.data[, numericas], center = media, scale = desv.tipica)
psych::describe(Filter(is.numeric, telco.data))

# Renombramos la columna con la variable objetivo
colnames(telco.data)[17] <- "target"

# 7. CREACION VARIABLES DUMMY
library(dummies)
columnas.dummy <- c("tenure", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod")
telco.data.final <- dummy.data.frame(telco.data[, columnas.dummy], sep = ".")

telco.data$SeniorCitizen <- as.numeric(as.character(telco.data$SeniorCitizen))

telco.data <- telco.data %>%
  mutate(Partner = ifelse(Partner == "No",0,1))

telco.data <- telco.data %>%
  mutate(Dependents = ifelse(Dependents == "No",0,1))

telco.data <- telco.data %>%
  mutate(PaperlessBilling = ifelse(PaperlessBilling == "No",0,1))

telco.data.final <- cbind(telco.data.final, telco.data[, c("Partner", "Dependents", "PaperlessBilling")], telco.data[, numericas], telco.data[, "target"])
colnames(telco.data.final)[38] <- "target"
colnames(telco.data.final) <- gsub(" ", ".", colnames(telco.data.final))

create_report(telco.data.final, output_file = "datos_telco_final")
  