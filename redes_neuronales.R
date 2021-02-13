# REDES NEURONALES
# 2 candidatos
source("./librerias/funcion steprepetido binaria.R")
source("./librerias/cruzadas avnnet y log binaria.R")
library(parallel)
library(doParallel)
library(caret)
variables.candidato.1 <- c("TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", 
                           "OnlineSecurity.No", "tenure.cont", "TechSupport.No", "Contract.Month.to.month")

variables.candidato.2 <- c("tenure.cont", "TotalCharges", "MonthlyCharges", "InternetService.Fiber.optic", 
                           "TechSupport.No")

vardep <- "target"

# Si quisieramos 30 observaciones por parametro
# h * (k + 1) + h + 1 = 7043 / 30 ~ 234.8
# Si k = 7, entonces 9 * h + 1 = 234.8, es decir, 26 nodos
# Si k = 5, entonces 7 * h + 1 = 234.8, es decir, 33 nodos
size.candidato.1 <- c(5, 10, 15, 20, 25, 30, 40)
decay.candidato.1 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.1 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.1, listclass=c(""),
                                          grupos=5,sinicio=1234,repe=15, size=size.candidato.1,decay=decay.candidato.1,repeticiones=15,itera=100)

# Si quisieramos 20 observaciones por parametro
# h * (k + 1) + h + 1 = 7043 / 20 ~ 352.2
# Si k = 7, entonces 9 * h + 1 = 352.2, es decir, 40 nodos
# Si k = 5, entonces 7 * h + 1 = 352.2, es decir, 50 nodos
size.candidato.2 <- c(5, 10, 15, 20, 25, 30, 40, 50)
decay.candidato.2 <- c(0.1, 0.01, 0.001)
cvnnet.candidato.2 <- cruzadaavnnetbin(data=telco.data.final,vardep=vardep,listconti=variables.candidato.2, listclass=c(""),
                                       grupos=5,sinicio=1234,repe=15, size=size.candidato.2,decay=decay.candidato.2,repeticiones=15,itera=100)

