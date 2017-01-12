# based on script on Croft's website - refer
# TODO:
# - clean up and write into functions
# - find out how to plot unicode in legends

library(pscl)
library(oc)
library(gdata)
library(ggplot2)

name <- ''
folder <- '.'
#
name <- args[2]
folder <- args[1]
# read folder and name from command line or define them manually
fp <- sprintf('%s/%s.csv',folder, name)

data.txt <- read.csv(fp,header=TRUE)
nCUTs <- length(2:ncol(data.txt))
tt <- data.txt [,-1]
names <- colnames(tt)
T <- (tt[,1:nCUTs])
hr <- rollcall(T, yea=1, nay=6, missing=9,notInLegis=8,desc= title,) 
   
minVOTES <- 5
# minimum number of positive terms a situation must have
# minimal percentage of situations a term may occur in positive
min_freq = 2
LOP <- (1/nrow(data.txt))* min_freq

# n Dimensional Analysis 

ndims = 2
result <- oc(hr,
               dims=ndims,
               minvotes=minVOTES,
               lop=LOP,
               polarity = 1:ndims,
               verbose = TRUE)
df = data.frame(result$legislators[,7],
  row.names = 1:nrow(result$legislators))
colnames(df) = c('dim.1')
for (i in 1:ndims-1) {
  colname = sprintf('dim.%d',i+1)
  df[[colname]] = result$legislators[,7+i]
}
write.csv(df, sprintf('%s/%s_%ddim.csv', folder, name, ndims))

coordinates = read.csv(sprintf('%s/%s_%ddim.csv', folder, name, ndims), sep = ',', header = TRUE)
labels.fp = sprintf('%s/%s_labels.csv', folder, name)
labels = read.csv(labels.fp)
gold.fp = sprintf('%s/%s_gold.csv', folder, name)
gold = read.csv(gold.fp)
index.fp = sprintf('%s/%s_situations.csv', folder, name)
situations = read.csv(index.fp)
data = cbind(coordinates, labels, gold)
data$situations = paste(situations$utterance, situations$word)

xlims = c(min(data$dim.1),max(data$dim.1))
ylims = c(min(data$dim.2),max(data$dim.2))

for (language in 5:34) {
  data.sub = data[data[,language] != '',]
  #
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > nrow(data.txt)/100 * min_freq])
  print(top)
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(jitter(dim.1,100), jitter(dim.2,100), color = data.subsub[,language], data = data.subsub)
  q = q + xlim(xlims)
  q = q + ylim(ylims)
  ggsave(sprintf('%s/%s_L%d.pdf', folder, name, language), q, height = 7, width = 7)  
}
  
centroid = aggregate(cbind(dim.1,dim.2) ~ gold, data = data, FUN = mean)
q = qplot(dim.1, dim.2, label = gold, data = centroid, geom = 'text')
ggsave(sprintf('%s/%s_centroid.pdf', folder, name), q)

q = qplot(dim.1, dim.2, label = situations, data = data, geom = 'text')
ggsave(sprintf('%s/%s_situations.pdf', folder, name), q, height = 15, width = 15)

###
#
###

data.dn = data[data$gold == 'DN',]
data.dn = data.dn[data.dn$dim.1 < -.2 & data.dn$dim.2 > 0,]
data.dn = data.dn[order(data.dn$dim.1),]
for (i in 5:34) { 
	print(i)
	print(droplevels(data.dn[,i]))
}

###
#
###

# pos -- a position specifier for the text. Values of 1, 2, 3 and 4, 
# respectively indicate positions below, to the left of, above and 
# to the right of the specified coordinates 
#
nrow <- length(result$legislators[,7])
namepos <- rep(2,nrow)
#

ws <- result$rollcalls[,8]
N1 <- result$rollcalls[,6]
N2 <- result$rollcalls[,7]

#Generates Point Data Figure [With Points Only]

data.txt[data.txt$X == 'None',]$X = ''

jpeg(paste (title,'Points_3D_12.jpeg',sep = "_"),width = 7, height = 7, units = 'in',res = 300)
plot(oc1,oc3,type="n",asp=1,
       main="",
       xlab="",
       ylab="",
       xlim=c(-1.0,1.0),ylim=c(-1.0,1.0),cex=1.2,font=2)
#
# Main title
mtext("OC Plot of Example Tabs Data\nStimuli (Row) Ideal Points",side=3,line=1.50,cex=1.2,font=2)
# x-axis title
mtext("Dimension 1",side=1,line=3.25,cex=1.2)
# y-axis title
mtext("Dimension 2",side=2,line=2.5,cex=1.2)
#
#points(oc2,oc3,pch=16,col="red",font=2)
text(oc1,oc2,labels = data.txt$X)
#text(oc1,oc2,names,pos=namepos,offset=00.20,col="blue")
dev.off()


#Generates Point Data Figure [Labels Only]

jpeg(paste (title,'Points_Fig2.jpeg',sep = "_"),width = 7, height = 7, units = 'in',res = 300)
plot(oc1,oc2,type="n",asp=1,
       main="",
       xlab="",
       ylab="",
       xlim=c(-1.0,1.0),ylim=c(-1.0,1.0),cex=1.2,font=2)

mtext("OC Plot of Example Tabs Data\nStimuli (Row) Ideal Points",side=3,line=1.50,cex=1.2,font=2)
mtext("Dimension 1",side=1,line=3.25,cex=1.2)
mtext("Dimension 2",side=2,line=2.5,cex=1.2)
#text(oc1,oc2,names,pos=namepos,col="blue")
text(oc1,oc2,names,col="blue")
dev.off()


#Generates Cutting Line Figure

plot(N1,N2,type="n",asp=1,
       main="",
       xlab="",
       ylab="",
       xlim=c(-1.0,1.0),ylim=c(-1.0,1.0),cex=1.2,font=2)


# Main title
mtext("OC Plot of Example Tabs Data\nCoombs Mesh from Cutting Lines",side=3,line=1.50,cex=1.2,font=2)
# x-axis title
mtext("Dimension 1",side=1,line=3.25,cex=1.2)
# y-axis title
mtext("Dimension 2",side=2,line=2.5,cex=1.2)
#

#
#  Set Length of Arrows off ends of Cutting Lines
#
xlarrow <- 0.1
#xlarrow <- 0.0
#
#
i <- 1
#while (i <= 4){
while (i <= length(ws)){
     if(result999[i,7]!=999){
#  Plot Cutting Line
#

#
xws <- ws[i]*N1[i]
yws <- ws[i]*N2[i]
#
#  This computes the Cutting Line
#
arrows(xws,yws,xws+N2[i],yws-N1[i],length=0.0,lwd=2,col="black")
arrows(xws,yws,xws-N2[i],yws+N1[i],length=0.0,lwd=2,col="black")
#
#
#  SET POLARITY HERE
#
polarity <- oc1*N1[i] + oc2*N2[i] - ws[i]
vote <- hr$votes[,i]
ivote <- as.integer(vote)
errors1 <- ivote==1 & polarity >= 0
errors2 <- ivote==6 & polarity <= 0
errors3 <- ivote==1 & polarity <= 0
errors4 <- ivote==6 & polarity >= 0
kerrors1 <- ifelse(is.na(errors1),9,errors1)
kerrors2 <- ifelse(is.na(errors2),9,errors2)
kerrors3 <- ifelse(is.na(errors3),9,errors3)
kerrors4 <- ifelse(is.na(errors4),9,errors4)
kerrors12 <- sum(kerrors1==1)+sum(kerrors2==1)
kerrors34 <- sum(kerrors3==1)+sum(kerrors4==1)

#
if(kerrors12 < kerrors34){
   xwslow <- (ws[i]- xlarrow)*N1[i]
   ywslow <- (ws[i]- xlarrow)*N2[i]
}
if(kerrors12 >= kerrors34){
   xwslow <- (ws[i]+ xlarrow)*N1[i]
   ywslow <- (ws[i]+ xlarrow)*N2[i]
}
#
#
arrows(xws+N2[i],yws-N1[i],xwslow+N2[i],ywslow-N1[i],length=0.1,lwd=2,col="red") 
arrows(xws-N2[i],yws+N1[i],xwslow-N2[i],ywslow+N1[i],length=0.1,lwd=2,col="red") 
#
}

i <- i + 1
}

#text(oc1,oc2,names,pos=namepos,offset=00.20,col="blue")
points(oc1,oc2,pch=16,col="red",font=2)
dev.copy(jpeg,(paste (title,'CuttingLines_Fig1.jpeg', sep = "_")),width = 7, height = 7, units = 'in',res = 300)
dev.off() #Plots Figure 1.


text(ws*N1+N2,ws*N2-N1,names2,pos=namepos,offset=00.20,col="blue")
text(ws*N1-N2,ws*N2+N1,names2,pos=namepos,offset=00.20,col="blue")

dev.copy(jpeg,(paste (title,'CuttingLines_Fig2.jpeg', sep = "_")),width = 7, height = 7, units = 'in',res = 300)
dev.off() #Plots Figure 2.

