
# ------------- Depuracion ---------------
# Objetivo: realizar un analisis exploratorio + depuracion inicial datos
# Autor: Alberto Fernandez Hernandez

#--- Librerias
suppressPackageStartupMessages({
  library(caret)         # Data partitioning
  library(data.table)    # Lectura de ficheros mucho mas rapido que read.csv
  library(dplyr)         # Manipulacion de datos
  library(ggplotgui)     # EDA manual mediante entorno interactivo (GUI)
  library(ggplot2)       # Libreria grafica
  library(scorecard)     # Woebin + Woebin_plot + Information Value (IV)
  library(bestNormalize) # Transformacion optima variables continuas
  library(VGAM)          # Aplicacion de transformaciones sobre variables
  library(dummies)       # Creacion variables dummy
  library(psych)         # Informacion estadistica de dataframes
  library(ranger)        # Random Forest
  library(forcats)       # Tratamiento variables categoricas
  
  source("./librerias/librerias_propias.R")
})

surgical_dataset <- fread("./data/Surgical-deepnet.csv", data.table = FALSE)
dim(surgical_dataset) # Filas x columnas

# Problema - numerosas observaciones
# Por tiempo, se ha tomado la decision de elegir un subconjunto de datos
# Para ello, recurrimos a createDataPartition
set.seed(1234)
partitions <- createDataPartition(surgical_dataset$complication, p = 0.40, list = FALSE)
surgical_dataset_partitioned <- surgical_dataset[partitions, ]

#--- EDA (Exploratory Data Analysis)
surgical_dataset_partitioned$complication <- as.factor(surgical_dataset_partitioned$complication)
make_eda_report(surgical_dataset_partitioned, "complication", "First EDA Report", 
                "00_EDA_initial_report.html", "./reports")

# Por numerico de valores unicos (y por la documentacion), codificamos las siguientes variables como categoricas
cat_columns <- c("gender", "race", "asa_status", "baseline_cancer", "baseline_cvd", "baseline_dementia",
                 "baseline_diabetes", "baseline_digestive", "baseline_osteoart", "baseline_psych",
                 "baseline_pulmonary", "dow", "month", "moonphase", "mort30")
surgical_dataset_partitioned[,cat_columns] <- lapply(surgical_dataset_partitioned[, cat_columns], factor)

# Problema: complication_rsi y ccsComplicationRate se calculan a partir de la variable objetivo
# surgical_dataset_partitioned$complication_rsi    <- NULL
# surgical_dataset_partitioned$ccsComplicationRate <- NULL

# Separamos las variables en numericas, categoricas y target
cat_columns <- names(Filter(is.factor, surgical_dataset_partitioned))[-16]
num_columns <- names(Filter(is.numeric, surgical_dataset_partitioned))
target      <- "complication"

# Elaboramos un segundo informe
# ¿Se observa alguna diferencia en la proporcion de variables objetivo?
# A simple vista, llaman la atencion:
# -> asa_status (diferencia proporcion entre 0, 1 y 2)
# -> baseline_cancer (hay mas pacientes, que sufren o han sufrido cancer, que tienen complicaciones tras la operacion)
# -> baseline_cvd (hay un mayor numero de pacientes, aunque ligeramente, que tienen complicaciones cuando cvd = 0)
# -> baseline_dementia (hay un mayor numero de pacientes que sufren alguna complicacion cuando baseline_dementia = 1)
# -> baseline_digestive (hay un mayor numero de pacientes, ligeramente, que sufren un mayor numero de complicaciones cuando baseline_digestive = 1)
# -> baseline_osteoart  (hay un mayor numero de pacientes que sufren alguna complicacion cuando baseline_osteoart = 0)
# -> baseline_pulmonary (cuando es igual a 1, aunque muy por encima ligeramente)
# -> dow (los martes, miercoles y viernes son los dias de la semana donde mas pacientes sufren complicaciones)
# -> month (llama la atencion que el mes de agosto es el mes donde menos complicaciones sufren los pacientes)
# -> moonphase (cuando hay luna nueva, hay un menor numero de pacientes que sufren complicaciones)
# -> mort30 (cuando mort30 = 1, especialmente, hay un mayor numero de pacientes que sufren complicaciones)
# -> race (aunque no muy relevante, la categoria "others" es la raza que mayor numero de complicaciones sufre)
make_eda_report(surgical_dataset_partitioned, target, "Second EDA Report", 
                "00_EDA_report_with_factors.html", "./reports")

# ¿Y el criterio del Valor de la Informacion o IV?
# Categoricas
# A simple vista, las variables cuyo valor de informacion es poco o nada relevante
# asa_status (0.0063)
# baseline_cvd (0.0268)
# baseline_dementia (0)
# baseline_diabetes (9e-04)
# baseline_digestive (0.006)
# baseline_psych (0)
# baseline_pulmonary (0.0033)
# mort30 (0)
# race (1e-04)
salida.woe <- woebin(surgical_dataset_partitioned[, c(cat_columns, target)], y = target, positive = 1)
pdf("./reports/information_value_categorical.pdf")
woebin_plot(salida.woe)
dev.off()

#--- ¿Podriamos agrupar alguna categoria?
surgical_dataset_partitioned_num <- surgical_dataset_partitioned
surgical_dataset_partitioned_num$month <- as.numeric(as.character(surgical_dataset_partitioned_num$month))
surgical_dataset_partitioned_num$dow <- as.numeric(as.character(surgical_dataset_partitioned_num$dow))

# ggplot_shiny(surgical_dataset_partitioned_num)
# dow - mientras que los lunes tan solo el 14 % de los pacientes que fueron operados sufrieron alguna complicacion
#       los viernes el 35.2 % de los pacientes sufrieron algun problema post-operatorio.
graph_dow <- ggplot(surgical_dataset_partitioned_num, aes(x = dow, fill = complication)) +
  geom_histogram(position = 'identity', alpha = 0.83, binwidth = 1) +
  labs(x = 'Month', y = 'Nº patients') +
  ggtitle('Complication distribution by month') +
  theme_minimal() +
  theme(
    text = element_text(family = 'Helvetica')
  )
graph_dow

# Recodificamos las categorias
surgical_dataset_partitioned <- surgical_dataset_partitioned %>% mutate(dow = recode(
                                          dow, 
                                          `1` = "1-3",
                                          `2` = "1-3",
                                          `3` = "1-3"
                                    ))

# month - los meses de enero y septiembre (en especial con este ultimo), son los que presentan un menor
# numero de complicaciones durante el post-operatorio. Por el contrario, durante los meses de febrero-agosto y
# octubre-diciembre acumulan en conjunto un 33.2 y 31.7 % de complicaciones, respectivamente
graph_month <- ggplot(surgical_dataset_partitioned_num, aes(x = month, fill = complication)) +
  geom_histogram(position = 'identity', alpha = 0.83, binwidth = 1) +
  labs(x = 'Month', y = 'Nº patients') +
  ggtitle('Complication distribution by month') +
  theme_minimal() +
  theme(
    text = element_text(family = 'Helvetica')
  )
graph_month

surgical_dataset_partitioned <- surgical_dataset_partitioned %>% mutate(month = recode(
                                  month, 
                                  `1` = "1-7", `2` = "1-7", `3` = "1-7", `4` = "1-7", `5` = "1-7",
                                  `6` = "1-7", `7` = "1-7", `9` = "9-11", `10` = "9-11", `11` = "9-11"
                                ))

# ¿Y en relacion con las variables numericas?
# Analizando los graficos qq-plot, llama la atencion el hecho de que algunas de las variables presentan
# una separacion no lineal. Es mas, llama la atencion algunas variables con un numero limitado de valores
# ¿Podriamos pasar algunas de las variables numericas a categoricas?
# Algunas de las variables continuas como
# baseline_charlson (12)            -> relacion lineal
# ahrq_ccs (22)                     -> relacion no lineal 
# ccsComplicationRate (22)          -> relacion lineal
# ccsMort30Rate (20)                -> relacion no lineal
# No vamos a cambiar a categoricas
apply(surgical_dataset_partitioned[, num_columns], 2, function(x) {length(unique(x))})
salida.woe <- woebin(surgical_dataset_partitioned[, c(num_columns, target)], y = target, positive = 1)
pdf("./reports/information_value_numeric.pdf")
woebin_plot(salida.woe)
dev.off()

rm(surgical_dataset_partitioned_num)

#--- ¿Puede ser necesario aplicar transformaciones sobre las variables numericas?
# Creamos una copia del dataframe original
surgical_dataset_partitioned_transf <- surgical_dataset_partitioned

best_lambda_vector <- c()
for(col in num_columns) {
  print(col)
  best_lambda <- yeojohnson(unlist(surgical_dataset_partitioned_transf[, col]))$lambda
  best_lambda_vector <- c(best_lambda_vector, best_lambda)
  surgical_dataset_partitioned_transf[, col] <- yeo.johnson(unlist(surgical_dataset_partitioned_transf[, col]), 
                                                            lambda = best_lambda)
}
data.frame("column" = num_columns, "lambda transf." = best_lambda_vector)

# No nos interesa aplicar transformaciones
salida.woe.copia <- woebin(surgical_dataset_partitioned_transf, y = target, positive = 1)
sapply(salida.woe.copia[num_columns], function(x) x$total_iv[1]) - 
        sapply(salida.woe[num_columns], function(x) x$total_iv[1])

rm(surgical_dataset_partitioned_transf)

# ccsComplicationRate ¿Una variable demasiado predictiva?
# --- Estandarizacion de variables
surgical_dataset_partitioned_stnd <- surgical_dataset_partitioned

media <- sapply(surgical_dataset_partitioned_stnd[, num_columns], mean)
desv.tipica <- sapply(surgical_dataset_partitioned_stnd[, num_columns], sd)
surgical_dataset_partitioned_stnd[, num_columns] <- scale(surgical_dataset_partitioned_stnd[, num_columns], 
                                                   center = media, 
                                                   scale = desv.tipica)
describe(Filter(is.numeric, surgical_dataset_partitioned_stnd))

# --- Creacion variables dummy
columnas_dummy <- c("asa_status", "dow", "month", 
                    "moonphase", "race")
surgical_dataset_partitioned_stnd_dummy <- dummy.data.frame(surgical_dataset_partitioned_stnd[, columnas_dummy], 
                                                            sep = ".")

surgical_dataset_final <- cbind(
                                surgical_dataset_partitioned_stnd[, num_columns],
                                surgical_dataset_partitioned_stnd[, cat_columns[!cat_columns %in% columnas_dummy]],
                                surgical_dataset_partitioned_stnd_dummy,
                                surgical_dataset_partitioned_stnd[, target]
                              )
names(surgical_dataset_final)[37] <- "target"

# Renombramos las columnas para adecuarlas a formulas
names(surgical_dataset_final) <- make.names(colnames(surgical_dataset_final))

# Probamos con un random forest
my_model <- ranger( 
  target ~ . , 
  importance = 'impurity',
  data = surgical_dataset_final,
  seed = 1234
)
# **Estimacion** del error / acierto **esperado**
acierto <- 1 - my_model$prediction.error
acierto
#-- Con dataset completo: 0.880
#-- Con 5854 observaciones: 0.844

# Pintar importancia de variables
impor_df <- as.data.frame(my_model$variable.importance)
names(impor_df)[1] <- c('Importance')
impor_df$vars <- rownames(impor_df)
rownames(impor_df) <- NULL

ggplot(impor_df, aes(fct_reorder(vars, Importance), Importance)) +
  geom_col(group = 1, fill = "darkred") +
  coord_flip() + 
  labs(x = 'Variables', y = 'Importancia', title = 'Importancia Variables') +
  theme_bw()
ggsave('./charts/00_variable_importance_random_forest.png')
#-- Con dataset completo y un subconjunto es muy similar la importancia

# 1 - "Yes" ; 0 - "No"
surgical_dataset_final$target <- ifelse(
  surgical_dataset_final$target == 1,
  "Yes",
  "No"
)

surgical_dataset_final$target <- as.factor(surgical_dataset_final$target)

#--- Nos guardamos el resto de variables como conjunto test
test_surgical_dataset <- surgical_dataset[-partitions, ]

# Aplicamos los mismos cambios al conjunto test
names(test_surgical_dataset)[37] <- "target"

test_surgical_dataset$target <- ifelse(
  test_surgical_dataset$target == 1,
  "Yes",
  "No"
)

test_surgical_dataset$target <- as.factor(test_surgical_dataset$target)

test_surgical_dataset[,cat_columns] <- lapply(test_surgical_dataset[, cat_columns], factor)
test_surgical_dataset <- test_surgical_dataset %>% mutate(dow = recode(
  dow, 
  `1` = "1-3",
  `2` = "1-3",
  `3` = "1-3"
))

test_surgical_dataset <- test_surgical_dataset %>% mutate(month = recode(
  month, 
  `1` = "1-7", `2` = "1-7", `3` = "1-7", `4` = "1-7", `5` = "1-7",
  `6` = "1-7", `7` = "1-7", `9` = "9-11", `10` = "9-11", `11` = "9-11"
))

# Importante: debemos estandarizar los datos con la media/desv. tipica original del conjunto
# de entrenamiento
test_surgical_dataset[, num_columns] <- scale(test_surgical_dataset[, num_columns], 
                                                              center = media, 
                                                              scale = desv.tipica)
test_surgical_dataset_dummy <- dummy.data.frame(test_surgical_dataset[, columnas_dummy], 
                                                sep = ".")

test_surgical_dataset <- cbind(
                                test_surgical_dataset[, num_columns],
                                test_surgical_dataset[, cat_columns[!cat_columns %in% columnas_dummy]],
                                test_surgical_dataset_dummy,
                                test_surgical_dataset[, "target"]
                              )
names(test_surgical_dataset) <- make.names(colnames(test_surgical_dataset))
fwrite(test_surgical_dataset, "./data/surgical_test_data.csv")

#--- Guardamos el fichero RData
save.image(file = "./rdata/Depuracion.RData")

#--- Guardamos el dataset final
fwrite(surgical_dataset_final, "./data/surgical_dataset_final.csv")

#--- Informe final
make_eda_report(surgical_dataset_final, "target", "Final EDA Report", 
                "00_EDA_report_final.html", "./reports")


