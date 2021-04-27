#Cuenta el n?mero de valores diferentes para las num?ricas
cuentaDistintos<-function(matriz){
  sapply(Filter(is.numeric, matriz),function(x) length(unique(x)))
}

# Diagrama de cajas para las variables cuantitativas y variable objetivo binaria
boxplot_targetbinaria<-function(var,target,nombreEje){
  dataaux<-data.frame(variable=var,target=target)
  ggplot(dataaux,aes(y=var))+
    geom_boxplot(aes(x="All"), notch=TRUE, fill="grey") +
    stat_summary(aes(x="All"), fun.y=mean, geom="point", shape=8) +
    geom_boxplot(aes(x=target, fill=target), notch=TRUE) +
    stat_summary(aes(x=target), fun.y=mean, geom="point", shape=8) +
    ylab(nombreEje)
}

# Histograma para las variables cuantitativas y variable objetivo binaria
hist_targetbinaria<-function(var,target,nombreEje){
  dataaux<-data.frame(variable=var,target=target)
  ggplot(dataaux, aes(x=var))+
    geom_density(aes(colour=target, fill=target), alpha=0.5) +
    geom_density(lty=1)+
    xlab(nombreEje)
}

# Gr?fico mosaico para las variables cualitativas y variable objetivo binaria
mosaico_targetbinaria<-function(var,target,nombreEje){
  ds <- table(var, target)
  ord <- order(apply(ds, 1, sum), decreasing=TRUE)
  mosaicplot(ds[ord,], color=c("darkturquoise","indianred1"), las=2, main="",xlab=nombreEje)
}

# Gr?fico de barras para las variables cualitativas y variable objetivo binaria
barras_targetbinaria<-function(var,target,nombreEje){
  dataaux<-data.frame(variable=var,target=target)
  ggplot(dataaux, aes(x="All",y = (..count..)/sum(..count..))) +
  #geom_bar(aes_string(fill=target))+
    geom_bar(aes(var,fill=target))+
    ylab("Frecuencia relativa")+xlab(nombreEje)
}

# Gr?fico correlaciones, c?digo de Rattle
graficoCorrelacion<-function(target,matriz){
  panel.hist <- function(x, ...)
  {
    usr <- par("usr"); on.exit(par(usr))
    par(usr=c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot=FALSE)
    breaks <- h$breaks; nB <- length(breaks)
    y <- h$counts; y <- y/max(y)
    rect(breaks[-nB], 0, breaks[-1], y, col="grey90", ...)
  }
  
  panel.cor <- function(x, y, digits=2, prefix="", cex.cor, ...)
  {
    usr <- par("usr"); on.exit(par(usr))
    par(usr = c(0, 1, 0, 1))
    r <- (cor(x, y, use="complete"))
    txt <- format(c(r, 0.123456789), digits=digits)[1]
    txt <- paste(prefix, txt, sep="")
    if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
    text(0.5, 0.5, txt)
  }
  
  pairs(cbind(target,Filter(is.numeric, matriz)), 
        diag.panel=panel.hist, 
        upper.panel=panel.smooth, 
        lower.panel=panel.cor)
}

# Cuenta el n?mero de at?picos y los transforma en missings
atipicosAmissing<-function(varaux){
  if (abs(skew(varaux))<1){
    criterio1<-abs((varaux-mean(varaux,na.rm=T))/sd(varaux,na.rm=T))>3
  } else {
    criterio1<-abs((varaux-median(varaux,na.rm=T))/mad(varaux,na.rm=T))>8
  }
  qnt <- quantile(varaux, probs=c(.25, .75), na.rm = T)
  H <- 3 * IQR(varaux, na.rm = T)
  criterio2<-(varaux<(qnt[1] - H))|(varaux>(qnt[2] + H))
  varaux[criterio1&criterio2]<-NA
  return(list(varaux,sum(criterio1&criterio2,na.rm=T)))
}

# Borras las observaciones completas si son at?picas
obsAtipicasBorrar<-function(varaux){
  if (abs(skew(varaux))<1){
    criterio1<-abs((varaux-mean(varaux,na.rm=T))/sd(varaux,na.rm=T))>3
  } else {
    criterio1<-abs((varaux-median(varaux,na.rm=T))/mad(varaux,na.rm=T))>8
  }
  qnt <- quantile(varaux, probs=c(.25, .75), na.rm = T)
  H <- 3 * IQR(varaux, na.rm = T)
  criterio2<-(varaux<(qnt[1] - H))|(varaux>(qnt[2] + H))
  !(criterio1&criterio2)
}

# Imputaci?n variables cuantitativas
ImputacionCuant<-function(vv,tipo){#tipo debe tomar los valores media, mediana o aleatorio
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

# Imputaci?n variables cualitativas
ImputacionCuali<-function(vv,tipo){#tipo debe tomar los valores moda o aleatorio
  if (tipo=="moda"){
    vv[is.na(vv)]<-names(sort(table(vv),decreasing = T))[1]
  } else if (tipo=="aleatorio"){
    vv[is.na(vv)]<-sample(vv[!is.na(vv)],sum(is.na(vv)),replace = T)
  }
  vv
}

# Busca la transformaci?n de variables input de intervalo que maximiza la correlaci?n con la objetivo continua
mejorTransfCorr<-function(vv,target){
  vv<-scale(vv)
  vv<-vv+abs(min(vv,na.rm=T))*1.0001
  posiblesTransf<-data.frame(x=vv,logx=log(vv),expx=exp(vv),sqrx=vv^2,sqrtx=sqrt(vv),cuartax=vv^4,raiz4=vv^(1/4))
  return(list(colnames(posiblesTransf)[which.max(abs(cor(target,posiblesTransf, use="complete.obs")))],posiblesTransf[,which.max(abs(cor(target,posiblesTransf, use="complete.obs")))]))
}

# Calcula el V de Cramer
Vcramer<-function(v,target){
  if (is.numeric(v)){
    v<-cut(v,breaks=unique(quantile(v,probs = seq(0,1,0.2))),include.lowest=T)
  }
  if (is.numeric(target)){
    target<-cut(target,breaks=unique(quantile(target,probs = seq(0,1,0.2))),include.lowest=T)
  }
  cramer.v(table(v,target))
}


# Busca la transformaci?n de variables input de intervalo que maximiza la V de Cramer con la objetivo binaria
mejorTransfVcramer<-function(vv,target){
  vv<-scale(vv)
  vv<-vv+abs(min(vv,na.rm=T))*1.0001
  posiblesTransf<-data.frame(x=vv,logx=log(vv),expx=exp(vv),sqrx=vv^2,sqrtx=sqrt(vv),cuartax=vv^4,raiz4=vv^(1/4))
  return(list(colnames(posiblesTransf)[which.max(apply(posiblesTransf,2,function(x) Vcramer(x,target)))],posiblesTransf[,which.max(apply(posiblesTransf,2,function(x) Vcramer(x,target)))]))
}

# Detecta el tipo de variable objetivo y busca la mejor transformaci?n de las variables input continuas autom?ticamente
Transf_Auto<-function(matriz,target){
    if (is.numeric(target)){
    aux<-data.frame(apply(matriz,2,function(x) mejorTransfCorr(x,target)[[2]]))
    aux2<-apply(matriz,2,function(x) mejorTransfCorr(x,target)[[1]])
    names(aux)<-paste0(aux2,names(aux2))
  } else {
    aux<-data.frame(apply(matriz,2,function(x) mejorTransfVcramer(x,target)[[2]]))
    aux2<-apply(matriz,2,function(x) mejorTransfVcramer(x,target)[[1]])   
    names(aux)<-paste0(aux2,names(aux2))
  }
  return(aux)
}

# Gr?fico con el V de cramer de todas las variables input para saber su importancia
graficoVcramer<-function(matriz, target){
  salidaVcramer<-sapply(matriz,function(x) Vcramer(x,target))
  barplot(sort(salidaVcramer,decreasing =T),las=2,ylim=c(0,1))
}

#Para evaluar el R2 en regr. lineal en cualquier conjunto de datos
Rsq<-function(modelo,varObj,datos){
  testpredicted<-predict(modelo, datos)
  testReal<-datos[,varObj]
  sse <- sum((testpredicted - testReal) ^ 2)
  sst <- sum((testReal - mean(testReal)) ^ 2)
  1 - sse/sst
}

#Para evaluar el pseudo-R2 en regr. log?stica en cualquier conjunto de datos
pseudoR2<-function(modelo,dd,nombreVar){
  pred.out.link <- predict(modelo, dd, type = "response")
  mod.out.null <- glm(as.formula(paste(nombreVar,"~1")), family = binomial, data = dd)
  pred.out.linkN <- predict(mod.out.null, dd, type = "response")
  1-sum((dd[,nombreVar]==1)*log(pred.out.link)+log(1 -pred.out.link)*(1-(dd[,nombreVar]==1)))/
    sum((dd[,nombreVar]==1)*log(pred.out.linkN)+log(1 -pred.out.linkN)*(1-(dd[,nombreVar]==1)))
}

#Gr?fico con la importancia de las variables en regr. log?stica
impVariablesLog<-function(modelo,nombreVar,dd=data_train){
  null<-glm(as.formula(paste(nombreVar,"~1")),data=dd,family=binomial)
  aux2<-capture.output(aux<-step(modelo, scope=list(lower=null, upper=modelo), direction="backward",k=0,steps=1))
  aux3<-read.table(textConnection(aux2[grep("-",aux2)]))[,c(2,5)]
  aux3$V5<-(aux3$V5-modelo$deviance)/modelo$null.deviance
  barplot(aux3$V5,names.arg = aux3$V2,las=2,horiz=T,main="Importancia de las variables (Pseudo-R2)")
}

#Calcula medidas de calidad para un punto de corte dado
sensEspCorte<-function(modelo,dd,nombreVar,ptoCorte,evento){
  probs <-predict(modelo,newdata=dd,type="prob")
  data=factor(ifelse(probs$Yes>ptoCorte,"Yes","No"))
  cm<-confusionMatrix(data=data, reference=dd$target,positive=evento)
  c(cm$overall[1],cm$byClass[1:4])
}

#Generar todas las posibles interacciones
formulaInteracciones<-function(data,posicion){
  listaFactores<-c()
  lista<-paste(names(data)[posicion],'~')
  nombres<-names(data)
  for (i in (1:length(nombres))[-posicion]){
    lista<-paste(lista,nombres[i],'+')
    if (class(data[,i])=="factor"){
      listaFactores<-c(listaFactores,i)
      for (j in ((1:length(nombres))[-c(posicion,listaFactores)])){
        lista<-paste(lista,nombres[i],':',nombres[j],'+')
      }
    }
  }
  lista<-substr(lista, 1, nchar(lista)-1)
  lista
}