

steprepetidobinaria<- function(data=data,vardep="vardep",
  listconti="listconti",
 sinicio=12345,sfinal=12355,porcen=0.8,criterio="BIC")
 {

library(MASS)
library(dplyr)

resultados<-data.frame(c())
data<-data[,c(listconti,vardep)]
formu1<-formula(paste("factor(",vardep,")~.",sep=""))
formu2<-formula(paste("factor(",vardep,")~1",sep=""))
listamodelos<-list()

for (semilla in sinicio:sfinal)
{
set.seed(semilla)
sample <- sample.int(n = nrow(data),
 size = floor(porcen*nrow(data)), replace = F)

train <- data[sample, ]
test  <- data[-sample, ]


full<-glm(formu1,data=train,family = binomial(link="logit"))
null<-glm(formu2,data=train,family = binomial(link="logit"))


if  (criterio=='AIC')
  {
  selec1<-stepAIC(null,scope=list(upper=full),
   direction="both",family = binomial(link="logit"),trace=FALSE)
  } 
else   if  (criterio=='BIC')
  {
 k1=log(nrow(train))
 selec1<-stepAIC(null,scope=list(upper=full),
  direction="both",family = binomial(link="logit"),k=k1,trace=FALSE)
  }

vec<-(names(selec1[[1]]))

if (length(vec)!=1)

{
# CAMBIOS

cosa<-as.data.frame(table(vec))
cosa<-as.data.frame(t(cosa))
colnames(cosa)<-vec

# 1) creo un vector con todas las variables input y ceros
# 2) voy añadiendo

cosa<-cosa[2,,drop=FALSE]
cosa<-cosa[,-(1),drop=FALSE]
cosa<- data.frame(lapply(cosa, function(x) as.numeric(as.character(x))))
cosa$id<-semilla
}

if (length(vec)==1)
{
cosa<-data.frame()
cosa[1,"id"]<-semilla
cosa$id<-as.integer(cosa$id)
}
vectormodelo<-list(names(cosa),semilla)
listamodelos<-append(listamodelos,vectormodelo)  

if (semilla==sinicio)
{
listamod<-cosa
}

else if (semilla!=sinicio)
{
 listamod<-suppressMessages(full_join(cosa,listamod,by = NULL, copy =TRUE))
}

}

listamod[is.na(listamod)] <- 0

nom<-names(listamod)
listamod$modelo<-""
for (i in 1:nrow(listamod))
{
 listamod[i,c("modelo")]<-""
 listamod[i,c("contador")]=0

  for (vari in nom)
  { 
   if (listamod[i,vari]==1)
   {
   listamod[i,c("modelo")]<-paste(listamod[i,c("modelo")],vari,collapse="",sep="+")
   listamod[i,c("contador")]=listamod[i,c("contador")]+1
   }
  
   }

}
 
listamod$modelo<-substring(listamod$modelo, 2)

tablamod<-as.data.frame(table(listamod$modelo))
names(tablamod)<-c("modelo","Freq")

tablamod<-tablamod[order(-tablamod$Freq,tablamod$modelo),]

nuevo<-listamod[,c("modelo","id","contador")]

uni<-full_join(tablamod,nuevo,by ="modelo", copy =TRUE)

uni= uni[!duplicated(uni$modelo),]
uni$semilla<-semilla

li1<-list()
# str(listamodelos)
for (i in 1:nrow(uni))
{
 for (j in 1:length(listamodelos))
 {
    if (uni[i,c("id")]==listamodelos[j][[1]])
  {
   k<-as.vector(listamodelos[j-1][[1]])
   length(k)<-length(k)-1
   li1<-c(li1,list(k))
   j=length(listamodelos)
  }
 } 

}

 uni$semilla<-NULL
 uni$id<-NULL
 return(list(uni,li1))

}


# Ejemplo steprepetidobinaria

# load("saheartbis.Rda")
# 
# listconti<-c("sbp", "tobacco", "ldl", "adiposity",
#  "obesity", "alcohol","age", "typea",
#  "famhist.Absent", "famhist.Present")
# vardep<-c("chd")
# 
# data<-saheartbis
# 
# data<-data[,c(listconti,vardep)]
# 
# lista<-steprepetidobinaria(data=data,
#  vardep=vardep,listconti=listconti,sinicio=12345,
#  sfinal=12355,porcen=0.8,criterio="BIC")
# 
# tabla<-lista[[1]]
# dput(lista[[2]][[1]])
# dput(lista[[2]][[2]])
# 
# 
