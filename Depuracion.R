setwd("/Users/alberto/UCM/Machine Learning/Practica ML/MachineLearning/")
library(corrplot)
library(readxl)
library(ggplot2)
library(dplyr)

medic.data <- read.csv("Surgical-deepnet.csv")
dim(medic.data) # 7043 filas y 21 columnas

dput(names(medic.data))

# 1. Eliminacion de columnas
library(DataExplorer)
create_report(medic.data)

summary(medic.data)

# 2. RECATEGORIZACION DE VARIABLES
apply(medic.data, 2, function(x) {length(unique(x))})
medic.data[,c(3,4,6:12,17,18,21,22,24,25)] <- lapply(medic.data[, c(3,4,6:12,17,18,21,22,24,25)], factor)
summary(medic.data[, c(3,4,6:12,17,18,21,22,24,25)])

target<-"complication"
factores<-names(Filter(is.factor, medic.data))[-15]
numericas<-names(Filter(is.numeric, medic.data))

# 3. EDA
# Variables cualitativas
mosaico_targetbinaria<-function(var,target,nombreEje,sort=2:1){
  ds <- table(var, target)
  ord <- order(apply(ds, 1, sum), decreasing=TRUE)
  mosaicplot(ds[ord,], color=c("darkturquoise","indianred1"), main="",xlab=nombreEje)
}

for(columna in factores) {
  mosaico_targetbinaria(medic.data[, columna], medic.data[, "complication"],columna)
}

library(scorecard)
salida.woe <- woebin(medic.data, y = "complication", positive = 1)
pdf("IV.pdf")
woebin_plot(salida.woe)
dev.off()

# Podemos descartar determinadas columnas
medic.data$race <- NULL
medic.data$mort30 <- NULL
medic.data$baseline_pulmonary <- NULL
medic.data$baseline_psych <- NULL
medic.data$baseline_diabetes <- NULL
medic.data$baseline_dementia <- NULL
medic.data$asa_status <- NULL

# Variables cuantitativas
grafico_barras_cuantitativa <- function(columna) {
  ggplot(medic.data, aes_string(x = columna)) + geom_histogram(aes(color = complication, fill = complication),
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
gg_miss_var(medic.data)

# La media NO es representativa
psych::describe(Filter(is.numeric, medic.data))

# 5. RE-CATEGORIZACION DE VARIABLES CUALITATIVAS
# tenure
DescTools::Freq(medic.data$baseline_charlson)
medic.data$baseline_charlson <- car::recode(medic.data$baseline_charlson, "2:3='2-3'; 4:11 = '4-11';")

DescTools::Freq(medic.data$ahrq_ccs)
medic.data$ahrq_ccs <- car::recode(medic.data$ahrq_ccs, "2:10='2-10'; 11:22 = '11-22';")

DescTools::Freq(medic.data$ccsComplicationRate)
medic.data <- medic.data %>% 
  mutate(ccsComplicationRate = case_when(
    ccsComplicationRate < 0.1 ~ "[0-0.1)",
    ccsComplicationRate >= 0.1 & ccsComplicationRate < 0.16 ~ "[0.1-0.16)",
    ccsComplicationRate >= 0.16 & ccsComplicationRate < 0.2 ~ "[0.16-0.2)",
    ccsComplicationRate >= 0.2 ~ "[0.2-more)"
  ))

DescTools::Freq(medic.data$ccsMort30Rate)
medic.data <- medic.data %>% 
  mutate(ccsMort30Rate = case_when(
    ccsMort30Rate < 0.001 ~ "[0-0.001)",
    ccsMort30Rate >= 0.001 & ccsMort30Rate < 0.002 ~ "[0.001-0.002)",
    ccsMort30Rate >= 0.002 & ccsMort30Rate < 0.003 ~ "[0.002-0.003)",
    ccsMort30Rate >= 0.003 & ccsMort30Rate < 0.009 ~ "[0.003-0.009)",
    ccsMort30Rate >= 0.009 ~ "[0.009-more)",
  ))

DescTools::Freq(medic.data$dow)
medic.data$dow <- car::recode(medic.data$dow, "1:3 = '1-3'")
mosaico_targetbinaria(medic.data[, "dow"], medic.data[, "complication"],"dow")

DescTools::Freq(medic.data$month)
mosaico_targetbinaria(medic.data[, "month"], medic.data[, "complication"],"month")
medic.data$month <- car::recode(medic.data$month, "1:7 = '1-7'; 9:12 = '9-12'")

DescTools::Freq(medic.data$moonphase)
mosaico_targetbinaria(medic.data[, "moonphase"], medic.data[, "complication"],"moonphase")
medic.data$moonphase <- car::recode(medic.data$moonphase, "1:3 = '1-3'")

medic.data$baseline_charlson <- as.factor(medic.data$baseline_charlson)
medic.data$ahrq_ccs <- as.factor(medic.data$ahrq_ccs)
medic.data$ccsComplicationRate <- as.factor(medic.data$ccsComplicationRate)
medic.data$ccsMort30Rate <- as.factor(medic.data$ccsMort30Rate)
medic.data$month <- as.factor(medic.data$month)

factores<-names(Filter(is.factor, medic.data))[-13]
numericas<-names(Filter(is.numeric, medic.data))

# 5. TRANSFORMACION DE VARIABLES
medic.data.copia <- medic.data

library(bestNormalize)
vector.mejor.lambda <- c()
for(col in numericas) {
  mejor.lambda <- yeojohnson(unlist(medic.data.copia[, col]))$lambda
  vector.mejor.lambda <- c(vector.mejor.lambda, mejor.lambda)
  medic.data.copia[, col] <- VGAM::yeo.johnson(unlist(medic.data[, col]), lambda = mejor.lambda)
}
data.frame("columna" = numericas, "lambda" = vector.mejor.lambda)

create_report(medic.data.copia, output_file = "datos_imputados_transformados")

salida.woe.copia <- woebin(medic.data.copia, "complication", print_step = 0, positive = 1)
sapply(salida.woe.copia[numericas], function(x) x$total_iv[1]) - 
          sapply(salida.woe[numericas], function(x) x$total_iv[1])
psych::describe(Filter(is.numeric, medic.data))
psych::describe(Filter(is.numeric, medic.data.copia))

columnas.transformadas <- c("bmi")
medic.data[, columnas.transformadas] <- medic.data.copia[, columnas.transformadas]
rm(medic.data.copia)

# 6. ESTANDARIZACION DE VARIABLES CONTINUAS
media <- sapply(medic.data[, numericas], mean)
desv.tipica <- sapply(medic.data[, numericas], sd)
medic.data[, numericas] <- scale(medic.data[, numericas], center = media, scale = desv.tipica)
psych::describe(Filter(is.numeric, medic.data))

# Renombramos la columna con la variable objetivo
colnames(medic.data)[18] <- "target"
levels(medic.data$target) <- c("No", "Yes")

# 7. CREACION VARIABLES DUMMY
library(dummies)
apply(medic.data, 2, function(x) {length(unique(x))})
columnas.dummy <- c("baseline_charlson", "ahrq_ccs", "ccsComplicationRate", "ccsMort30Rate",
                    "dow", "month")
medic.data.final <- dummy.data.frame(medic.data[, columnas.dummy], sep = ".")

levels(medic.data$moonphase) <- c("0", "1")

medic.data$baseline_cancer <- as.numeric(as.character(medic.data$baseline_cancer))
medic.data$baseline_cvd <- as.numeric(as.character(medic.data$baseline_cvd))
medic.data$baseline_digestive <- as.numeric(as.character(medic.data$baseline_digestive))
medic.data$baseline_osteoart <- as.numeric(as.character(medic.data$baseline_osteoart))
medic.data$gender <- as.numeric(as.character(medic.data$gender))
medic.data$moonphase <- as.numeric(as.character(medic.data$moonphase))

medic.data.final <- cbind(medic.data.final, medic.data[, c("baseline_cancer", "baseline_cvd", 
                                                           "baseline_digestive", "baseline_osteoart",
                                                           "gender", "moonphase")], medic.data[, numericas], medic.data[, "target"])
colnames(medic.data.final)[38] <- "target"
colnames(medic.data.final) <- gsub("\\.", "-", colnames(medic.data.final))
colnames(medic.data.final) <- make.names(colnames(medic.data.final))
create_report(medic.data.final, output_file = "datos_final")
save.image("depuracion.RData")

  