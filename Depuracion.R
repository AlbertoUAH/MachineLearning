setwd("/Users/alberto/UCM/Machine Learning/Practica ML/")
library(corrplot)
library(readxl)
library(ggplot2)

datos.elecciones <- read_excel("DatosEleccionesEspaña.xlsx")
dim(datos.elecciones) # 8119 filas y 34 columnas

dput(names(datos.elecciones))

# 1. Eliminacion de columnas
library(DataExplorer)
create_report(datos.elecciones)

# El nombre del exoplaneta tambien debemos descartarlo, ya que solo los planetas marcados como "CONFIRMED" presentan nombre, por lo que no aporta informacion alguna al modelo
length(unique(datos.elecciones$Name))
datos.elecciones <- datos.elecciones[, !colnames(datos.elecciones) %in% c("Name")]

dim(datos.elecciones) # 8119 filas y 33 columnas

summary(datos.elecciones)

# 2. RECATEGORIZACION DE VARIABLES
apply(datos.elecciones, 2, function(x) {length(unique(x))})
datos.elecciones[,c(1,4,26,30)] <- lapply(datos.elecciones[,c(1,4,26,30)], factor)
summary(datos.elecciones[, c(1,4,26,30)])
datos.elecciones$Densidad<-questionr::recode.na(datos.elecciones$Densidad, "?") # Categoria desconocida
# Valores numericos que no corresponden con lo indicado en la documentacion
datos.elecciones$SameComAutonPtge <-replace(datos.elecciones$SameComAutonPtge, which(datos.elecciones$SameComAutonPtge > 100), NA)
datos.elecciones$ForeignersPtge <-replace(datos.elecciones$ForeignersPtge, which(datos.elecciones$ForeignersPtge < 0), NA)
datos.elecciones$PobChange_pct <-replace(datos.elecciones$PobChange_pct, which(datos.elecciones$PobChange_pct > 100), NA)
datos.elecciones$Explotaciones<-replace(datos.elecciones$Explotaciones,which(datos.elecciones$Explotaciones==99999),NA)

numericas<-names(Filter(is.numeric, datos.elecciones))
factores<-names(Filter(is.factor, datos.elecciones))[-2]

# 3. VALORES MISSING
# Observar gráficamente missing y estructura
library(naniar)
gg_miss_var(datos.elecciones)

# Observaciones missing por variables
tablamis.numericas<-as.data.frame(sapply(datos.elecciones[,numericas],function(x) sum(is.na(x))))
names(tablamis.numericas)[1]<-"nmiss"
max(tablamis.numericas)/nrow(datos.elecciones) # Alrededor de un 8 % (en el peor de los casos)

tablamis.factores<-as.data.frame(sapply(datos.elecciones[,factores],function(x) sum(is.na(x))))
names(tablamis.factores)[1]<-"nmiss"
max(tablamis.factores)/ nrow(datos.elecciones) # Apenas llega al 1 % (en el peor de los casos)

# Observaciones missing por observaciones
datos.elecciones$prop_missings<-apply(datos.elecciones,1,function(x) sum(is.na(x)) / length(colnames(datos.elecciones)))
summary(datos.elecciones$prop_missings)
sum(boxplot(datos.elecciones$prop_missings, plot = FALSE)$out > 0.3) # 5 observaciones con mas del 30 % de sus variables a missing

# La media NO es representativa
psych::describe(Filter(is.numeric, datos.elecciones))

# Columnas a imputar
columnas <- c("Explotaciones", "Industria", "Construccion", "inmuebles", "Servicios", "ComercTTEHosteleria", "Pob2010", "SameComAutonPtge", "SUPERFICIE", "PobChange_pct", "ForeignersPtge")

# 3.1 Imputaciones
# Variables cuantitativas
ImputacionCuant<-function(vv,tipo){
  if (tipo=="media"){
    vv[is.na(vv)]<-round(mean(vv,na.rm=T),4)
  } else if (tipo=="mediana"){
    vv[is.na(vv)]<-round(median(vv,na.rm=T),4)
  }
  vv
}
datos.elecciones.media <- sapply(Filter(is.numeric, datos.elecciones[, columnas]),function(x) ImputacionCuant(x,"media"))
datos.elecciones.mediana <- sapply(Filter(is.numeric, datos.elecciones[, columnas]),function(x) ImputacionCuant(x,"mediana"))

datos.elecciones[, columnas] <- sapply(Filter(is.numeric, datos.elecciones[, columnas]),function(x) ImputacionCuant(x,"mediana"))

# Imputacion manual
gg_miss_var(datos.elecciones)

personas_inmueble <- apply(datos.elecciones,1,function(x) ifelse(is.na(x["PersonasInmueble"]), round(as.numeric(x["Population"]) / as.numeric(x["inmuebles"]), 3), x["PersonasInmueble"]))
datos.elecciones[, "PersonasInmueble"] <- as.numeric(personas_inmueble)

total.empresas <- apply(datos.elecciones,1,function(x) ifelse(is.na(x["totalEmpresas"]), x["totalEmpresas"] <- as.numeric(x["Industria"]) + as.numeric(x["Construccion"]) + 
                          as.numeric(x["ComercTTEHosteleria"]) + as.numeric(x["Servicios"]), x["totalEmpresas"]))
datos.elecciones[, "totalEmpresas"] <- as.numeric(total.empresas)

corrplot(cor(Filter(is.numeric, datos.elecciones), use="complete.obs", method="pearson"), method = "ellipse",type = "upper") #No se aprecia ning?n patr?n

# Variables cualitativas
modificar.columna <- function(fila) {
  densidad <- ""
  if(is.na(fila["Densidad"])) {
    proporcion <- as.numeric(fila["Population"]) / as.numeric(fila["SUPERFICIE"])
    ifelse(proporcion < 1, densidad <- "MuyBaja", ifelse(proporcion > 1 & proporcion < 5, densidad <- "Baja", densidad <- "Alta"))
  }
  else {
    densidad <- fila["Densidad"]
  }
  as.factor(densidad)
}
datos.elecciones$Densidad <- apply(datos.elecciones, 1, modificar.columna)

summary(datos.elecciones)
corrplot(cor(Filter(is.numeric, datos.elecciones), use="complete.obs", method="pearson"), method = "ellipse",type = "upper") #No se aprecia ning?n patr?n

# 5. RE-CATEGORIZACION DE VARIABLES CUALITATIVAS
mosaico_targetbinaria<-function(var,target,nombreEje){
  ds <- table(var, target)
  ord <- order(apply(ds, 1, sum), decreasing=TRUE)
  mosaicplot(ds[ord,], color=c("darkturquoise","indianred1"), las=2, main="",xlab=nombreEje)
}
# CCAA
questionr::freq(datos.elecciones$CCAA)
mosaico_targetbinaria(datos.elecciones$CCAA, datos.elecciones$Derecha, "Derecha")

datos.elecciones$CCAA <- car::recode(datos.elecciones$CCAA, "c('Extremadura', 'Navarra', 'Asturias') = 'EX_NA_AS'; c('CastillaMancha', 'Aragón' , 'Canarias', 'Baleares') = 'CM_AR_CA_BA'; 
                                     c('Madrid', 'Rioja', 'Cantabria', 'Murcia', 'Ceuta', 'Melilla', 'Galicia', 'ComValenciana') = 'MA_CA_RI_MU_CE_ME_GA_CV'; c('PaísVasco', 'Cataluña') = 'PV_CAT'")

questionr::freq(datos.elecciones$Densidad)
questionr::freq(datos.elecciones$ActividadPpal)
mosaico_targetbinaria(datos.elecciones$ActividadPpal, datos.elecciones$Derecha, "Derecha")

datos.elecciones$ActividadPpal <- car::recode(datos.elecciones$ActividadPpal, "c('Construccion', 'Industria', 'Otro') = 'Construccion_Industria_Otro';")

create_report(datos.elecciones, output_file = "datos_elecciones_imputados")

# 5. TRANSFORMACION DE VARIABLES
datos.elecciones.copia <- datos.elecciones

library(bestNormalize)
vector.mejor.lambda <- c()
numericas <- c(numericas, "prop_missings")
for(col in numericas) {
  mejor.lambda <- yeojohnson(unlist(datos.elecciones.copia[, col]))$lambda
  vector.mejor.lambda <- c(vector.mejor.lambda, mejor.lambda)
  datos.elecciones.copia[, col] <- VGAM::yeo.johnson(unlist(datos.elecciones[, col]), lambda = mejor.lambda)
}
data.frame("columna" = numericas, "lambda" = vector.mejor.lambda)

create_report(datos.elecciones.copia, output_file = "datos_elecciones_imputados_transformados")

library(scorecard)

salida.woe <- woebin(datos.elecciones, "Derecha", print_step = 0)
salida.woe.copia <- woebin(datos.elecciones.copia, "Derecha", print_step = 0)
summary(sapply(salida.woe.copia[numericas], function(x) x$total_iv[1]) - 
          sapply(salida.woe[numericas], function(x) x$total_iv[1]))
psych::describe(Filter(is.numeric, datos.elecciones))


columnas.transformadas <- c("Population", "TotalCensus", "Age_over65_pct", "SameComAutonPtge", "AgricultureUnemploymentPtge", "IndustryUnemploymentPtge", "ConstructionUnemploymentPtge", "totalEmpresas", "inmuebles", "Pob2010", "SUPERFICIE", "PersonasInmueble", "Explotaciones")
datos.elecciones[, columnas.transformadas] <- datos.elecciones.copia[, columnas.transformadas]
rm(datos.elecciones.copia);

# 6. ESTANDARIZACION DE VARIABLES CONTINUAS
media <- sapply(datos.elecciones[, numericas], mean)
desv.tipica <- sapply(datos.elecciones[, numericas], sd)
datos.elecciones[, numericas] <- scale(datos.elecciones[, numericas], center = media, scale = desv.tipica)
psych::describe(Filter(is.numeric, datos.elecciones))

# PARA EVITAR PROBLEMAS, MEJOR DEFINIR LA VARIABLE OUTPUT
# con valores alfanuméricos Yes, No
datos.elecciones$Derecha <- ifelse(datos.elecciones$Derecha==1,"Si","No")
datos.elecciones$Derecha <- as.factor(datos.elecciones$Derecha)

# 7. CREACION VARIABLES DUMMY
library(dummies)
datos.elecciones[, factores] <- sapply(datos.elecciones[, factores], as.character)
datos.elecciones <- as.data.frame(datos.elecciones)
datos.elecciones.final <- dummy.data.frame(datos.elecciones, names = factores, sep = ".")

create_report(datos.elecciones, output_file = "datos_elecciones_final")
  