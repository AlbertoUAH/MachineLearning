setwd("/Users/alberto/UCM/Machine Learning/Practica ML/")
library(corrplot)

datos.exoplanetas <- read.csv("exoplanets_2018.csv", sep = ";")
dim(datos.exoplanetas) # 9564 filas y 44 columnas

dput(names(datos.exoplanetas))

# 1. Eliminacion de columnas
apply(datos.exoplanetas, 2, function(x) {length(unique(x))})
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("kepid", "kepoi_name", "koi_pdisposition", "koi_score")]
datos.exoplanetas <- datos.exoplanetas[datos.exoplanetas$koi_disposition != "CANDIDATE", ]

dim(datos.exoplanetas) # 7197 filas y 40 columnas

# El nombre del exoplaneta tambien debemos descartarlo, ya que solo los planetas marcados como "CONFIRMED" presentan nombre, por lo que no aporta informacion alguna al modelo
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("kepler_name")]
which(apply(datos.exoplanetas, 2, function(x) sum(is.na(x))) == dim(datos.exoplanetas)[1]) # koi_teq_err1 y koi_teq_err2 presentan todos los valores missing
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("koi_teq_err1", "koi_teq_err2")]

dim(datos.exoplanetas) # 7197 filas y 37 columnas

# 2. RECATEGORIZACION DE VARIABLES
apply(datos.exoplanetas, 2, function(x) {length(unique(x))})
datos.exoplanetas[,c(1:5)] <- lapply(datos.exoplanetas[,c(1:5)], factor)
summary(datos.exoplanetas[, c(1:5)])
datos.exoplanetas$koi_fpflag_nt<-questionr::recode.na(datos.exoplanetas$koi_fpflag_nt, "465") # Categoria desconocida

numericas<-names(Filter(is.numeric, datos.exoplanetas))
factores<-names(Filter(is.factor, datos.exoplanetas))[-1]

# 3. VALORES MISSING
# Observar gráficamente missing y estructura
library(naniar)
gg_miss_var(datos.exoplanetas)

# Observaciones missing por columnas
tablamis.numericas<-as.data.frame(sapply(datos.exoplanetas[,numericas],function(x) sum(is.na(x))))
names(tablamis.numericas)[1]<-"nmiss"
max(tablamis.numericas)/nrow(datos.exoplanetas) # Alrededor de un 5 %

tablamis.factores<-as.data.frame(sapply(datos.exoplanetas[,factores],function(x) sum(is.na(x))))
names(tablamis.factores)[1]<-"nmiss"
max(tablamis.factores)/ nrow(datos.exoplanetas) # Apneas llega al 0.01 %

# Observaciones missing por observaciones
datos.exoplanetas$prop_missings<-apply(datos.exoplanetas,1,function(x) sum(is.na(x)) / length(colnames(datos.exoplanetas)))
# Max = 0.72 -> Tengo alguna observacion con un 72 % de los datos perdidos
summary(datos.exoplanetas$prop_missings)
sum(boxplot(datos.exoplanetas$prop_missings, plot = FALSE)$out >= 0.5) # 258 observaciones con mas del 50 % de sus variables a missing
datos.exoplanetas<-datos.exoplanetas[datos.exoplanetas$prop_missings<=0.5,]

# Max = 0.37 -> Ahora, el maximo porcentaje es de un 37 %
summary(datos.exoplanetas$prop_missings)

psych::describe(Filter(is.numeric, datos.exoplanetas))

# 3.1 Imputaciones
# Variables cuantitativas
ImputacionCuant<-function(vv,tipo){
  if (tipo=="media"){
    vv[is.na(vv)]<-round(mean(vv,na.rm=T),4)
  } else if (tipo=="mediana"){
    vv[is.na(vv)]<-round(median(vv,na.rm=T),4)
  } else if (tipo=="aleatorio"){
    dd<-density(vv,na.rm=T,from=min(vv,na.rm = T),to=max(vv,na.rm = T))
    vv[is.na(vv)]<-round(approx(cumsum(dd$y)/sum(dd$y),dd$x,runif(sum(is.na(vv))))$y,4)
  }
  vv
}

correlacion.original <- cor(Filter(is.numeric, datos.exoplanetas), use="complete.obs", method="pearson")
datos.exoplanetas.media <- sapply(Filter(is.numeric, datos.exoplanetas),function(x) ImputacionCuant(x,"media"))
dif.correlacion.media <- abs(correlacion.original) - abs(cor(datos.exoplanetas.media, use="complete.obs", method="pearson"))

datos.exoplanetas.mediana <- sapply(Filter(is.numeric, datos.exoplanetas),function(x) ImputacionCuant(x,"mediana"))
dif.correlacion.mediana <- abs(correlacion.original) - abs(cor(datos.exoplanetas.mediana, use="complete.obs", method="pearson"))

datos.exoplanetas.aleatorio <- sapply(Filter(is.numeric, datos.exoplanetas),function(x) ImputacionCuant(x,"aleatorio"))
datos.exoplanetas.aleatorio <- apply(datos.exoplanetas.aleatorio,2,function(x) ImputacionCuant(x,"mediana"))
dif.correlacion.aleatorio <- abs(correlacion.original) - abs(cor(datos.exoplanetas.aleatorio, use="complete.obs", method="pearson"))

summary(c(dif.correlacion.media)); summary(c(dif.correlacion.mediana)); summary(c(dif.correlacion.aleatorio))

datos.exoplanetas[, numericas] <- sapply(Filter(is.numeric, datos.exoplanetas)[,-33],function(x) ImputacionCuant(x,"mediana"))

# Variables cualitativas
datos.exoplanetas[,"koi_fpflag_nt"]<-sapply(datos.exoplanetas[, "koi_fpflag_nt"],function(x) {
                                                          x[is.na(x)]<-names(sort(table(x),decreasing = T))[1]
                                                          as.factor(x)
                                                      })
summary(datos.exoplanetas)
corrplot(cor(Filter(is.numeric, datos.exoplanetas), use="complete.obs", method="pearson"), method = "ellipse",type = "upper") #No se aprecia ning?n patr?n

# 5. ESTANDARIZACION DE VARIABLES CONTINUAS
numericas <- c(numericas, "prop_missings")
media <- sapply(datos.exoplanetas[, numericas], mean)
desv.tipica <- sapply(datos.exoplanetas[, numericas], sd)
datos.exoplanetas[, numericas] <- scale(datos.exoplanetas[, numericas], center = media, scale = desv.tipica)

colnames(datos.exoplanetas)[1] <- "varObjBin"

# PARA EVITAR PROBLEMAS, MEJOR DEFINIR LA VARIABLE OUTPUT
# con valores alfanuméricos Yes, No
datos.exoplanetas$varObjBin <- ifelse(datos.exoplanetas$varObjBin=="CONFIRMED","Yes","No")
datos.exoplanetas$varObjBin <- as.factor(datos.exoplanetas$varObjBin)

# Dado que las variables cualitativas solo presentan dos valores, es mejor tratarlas como variables numericas
datos.exoplanetas$koi_fpflag_nt <- as.numeric(as.character(datos.exoplanetas$koi_fpflag_nt))
datos.exoplanetas$koi_fpflag_ss <- as.numeric(as.character(datos.exoplanetas$koi_fpflag_ss))
datos.exoplanetas$koi_fpflag_co <- as.numeric(as.character(datos.exoplanetas$koi_fpflag_co))
datos.exoplanetas$koi_fpflag_ec <- as.numeric(as.character(datos.exoplanetas$koi_fpflag_ec))
