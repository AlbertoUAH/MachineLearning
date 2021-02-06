setwd("/Users/alberto/UCM/Machine Learning/Practica ML/")
library(corrplot)

datos.exoplanetas <- read.csv("exoplanets_2018.csv", sep = ";")
dim(datos.exoplanetas) # 9564 filas y 43 columnas

dput(names(datos.exoplanetas))

# 1. Eliminacion de columnas
apply(datos.exoplanetas, 2, function(x) {length(unique(x))})
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("kepid", "kepoi_name", "koi_pdisposition", "koi_score")]
datos.exoplanetas <- datos.exoplanetas[datos.exoplanetas$koi_disposition != "CANDIDATE", ]

dim(datos.exoplanetas) # 7197 filas y 39 columnas

library(DataExplorer)
create_report(datos.exoplanetas)

# El nombre del exoplaneta tambien debemos descartarlo, ya que solo los planetas marcados como "CONFIRMED" presentan nombre, por lo que no aporta informacion alguna al modelo
library(dplyr)
datos.exoplanetas  %>% group_by(koi_disposition) %>% summarise(total_na = sum(kepler_name == ""))
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("kepler_name")]
which(apply(datos.exoplanetas, 2, function(x) sum(is.na(x))) == dim(datos.exoplanetas)[1]) # koi_teq_err1 y koi_teq_err2 presentan todos los valores missing
datos.exoplanetas <- datos.exoplanetas[, !colnames(datos.exoplanetas) %in% c("koi_teq_err1", "koi_teq_err2")]

dim(datos.exoplanetas) # 7197 filas y 36 columnas

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

# Observaciones missing por variables
tablamis.numericas<-as.data.frame(sapply(datos.exoplanetas[,numericas],function(x) sum(is.na(x))))
names(tablamis.numericas)[1]<-"nmiss"
max(tablamis.numericas)/nrow(datos.exoplanetas) # Alrededor de un 5 % (en el peor de los casos)

tablamis.factores<-as.data.frame(sapply(datos.exoplanetas[,factores],function(x) sum(is.na(x))))
names(tablamis.factores)[1]<-"nmiss"
max(tablamis.factores)/ nrow(datos.exoplanetas) # Apenas llega al 0.01 % (en el peor de los casos)

# Observaciones missing por observaciones
datos.exoplanetas$prop_missings<-apply(datos.exoplanetas,1,function(x) sum(is.na(x)) / length(colnames(datos.exoplanetas)))
# Max = 0.75 -> Tengo alguna observacion con un 75 % de los datos perdidos
summary(datos.exoplanetas$prop_missings)
sum(boxplot(datos.exoplanetas$prop_missings, plot = FALSE)$out > 0.5) # 258 observaciones con mas del 50 % de sus variables a missing
datos.exoplanetas<-datos.exoplanetas[datos.exoplanetas$prop_missings<=0.5,]

# Max = 0.38 -> Ahora, el maximo porcentaje es de un 38 %
summary(datos.exoplanetas$prop_missings)

# La media NO es representativa
psych::describe(Filter(is.numeric, datos.exoplanetas))

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

correlacion.original <- cor(Filter(is.numeric, datos.exoplanetas), use="complete.obs", method="pearson")
datos.exoplanetas.media <- sapply(Filter(is.numeric, datos.exoplanetas),function(x) ImputacionCuant(x,"media"))
dif.correlacion.media <- abs(correlacion.original) - abs(cor(datos.exoplanetas.media, use="complete.obs", method="pearson"))

datos.exoplanetas.mediana <- sapply(Filter(is.numeric, datos.exoplanetas),function(x) ImputacionCuant(x,"mediana"))
dif.correlacion.mediana <- abs(correlacion.original) - abs(cor(datos.exoplanetas.mediana, use="complete.obs", method="pearson"))

corrplot(cor(Filter(is.numeric, datos.exoplanetas), use="complete.obs", method="pearson"), method = "ellipse",type = "upper") #No se aprecia ning?n patr?n

summary(c(dif.correlacion.media)); summary(c(dif.correlacion.mediana));

datos.exoplanetas[, numericas] <- sapply(Filter(is.numeric, datos.exoplanetas)[,-32],function(x) ImputacionCuant(x,"mediana"))

# Variables cualitativas
datos.exoplanetas[,"koi_fpflag_nt"]<-sapply(datos.exoplanetas[, "koi_fpflag_nt"],function(x) {
                                                          x[is.na(x)]<-names(sort(table(x),decreasing = T))[1]
                                                          as.factor(x)
                                                      })
summary(datos.exoplanetas)
corrplot(cor(Filter(is.numeric, datos.exoplanetas), use="complete.obs", method="pearson"), method = "ellipse",type = "upper") #No se aprecia ning?n patr?n

# Modificamos los nombres de columnas
colnames(datos.exoplanetas) <- c("is_exoplanet", "light_curve_flag", "secondary_event_flag", "nearby_star_flag", "electronic_crosstalk_flag",
                                 "orbital_period", "orbital_period_err1", "orbital_period_err2", "distance_star_object_disk", "distance_star_object_disk_err1",
                                 "distance_star_object_disk_err2", "transit_duration", "transit_duration_err1", "transi_duration_err2", "ratio_blocked_by_object",
                                 "ratio_blocked_by_object_err1", "ratio_blocked_by_object_err2", "object_radius", "object_radius_err1", "object_radius_err2",
                                 "object_temperature", "luminosity_degree", "luminosity_degree_err1", "luminosity_degree_err2", "signal_noise_relation", "star_temperature",
                                 "star_temperature_err1", "star_temperature_err2", "object_gravity", "object_gravity_err1", "object_gravity_err2", "star_radius",
                                 "star_radius_err1", "star_radius_err2", "right_ascension", "declination", "prop_missing")

create_report(datos.exoplanetas, output_file = "datos_exoplanetas_imputados")

# 5. TRANSFORMACION DE VARIABLES
datos.exoplanetas.copia <- datos.exoplanetas

library(bestNormalize)
vector.mejor.lambda <- c()
numericas.filtradas <- setdiff(numericas, c("koi_duration", "koi_duration_err1", "koi_duration_err2", "koi_period_err1", "koi_period_err2", "dec", "koi_slogg", "koi_slogg_err2", "ra"))
for(col in numericas.filtradas) {
  mejor.lambda <- yeojohnson(unlist(datos.exoplanetas.copia[, col]))$lambda
  vector.mejor.lambda <- c(vector.mejor.lambda, mejor.lambda)
  datos.exoplanetas.copia[, col] <- VGAM::yeo.johnson(unlist(datos.exoplanetas[, col]), lambda = mejor.lambda)
}
data.frame("columna" = numericas.filtradas, "lambda" = vector.mejor.lambda)

create_report(datos.exoplanetas.copia, output_file = "datos_exoplanetas_imputados_transformados")

# 5. ESTANDARIZACION DE VARIABLES CONTINUAS
numericas <- c(numericas, "prop_missings")
media <- sapply(datos.exoplanetas[, numericas], mean)
desv.tipica <- sapply(datos.exoplanetas[, numericas], sd)
datos.exoplanetas[, numericas] <- scale(datos.exoplanetas[, numericas], center = media, scale = desv.tipica)

# PARA EVITAR PROBLEMAS, MEJOR DEFINIR LA VARIABLE OUTPUT
# con valores alfanuméricos Yes, No
datos.exoplanetas$is_exoplanet <- ifelse(datos.exoplanetas$is_exoplanet=="CONFIRMED","Yes","No")
datos.exoplanetas$is_exoplanet <- as.factor(datos.exoplanetas$is_exoplanet)

# Dado que las variables cualitativas solo presentan dos valores, es mejor tratarlas como variables numericas
datos.exoplanetas$light_curve_flag <- as.numeric(as.character(datos.exoplanetas$light_curve_flag))
datos.exoplanetas$secondary_event_flag <- as.numeric(as.character(datos.exoplanetas$secondary_event_flag))
datos.exoplanetas$nearby_star_flag <- as.numeric(as.character(datos.exoplanetas$nearby_star_flag))
datos.exoplanetas$electronic_crosstalk_flag <- as.numeric(as.character(datos.exoplanetas$electronic_crosstalk_flag))

create_report(datos.exoplanetas, output_file = "datos_exoplanetas_final")
  