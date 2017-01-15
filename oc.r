# based on script on Croft's website - refer
# TODO:
# - clean up and write into functions
# - find out how to plot unicode in legends

library(pscl)
library(oc)
library(gdata)
library(ggplot2)

language.labs = c('ar','bg','bs','cz','da','de','el','en','es','et','fi',
  'fr', 'he', 'hr', 'hu', 'id', 'it', 'ja', 'nl', 'no', 'pl', 'pr', 'ro',
  'ru', 'sl','sr','sv','tr','vi','zh')

name <- ''
folder <- '.'
#
name <- args[2]
folder <- args[1]
fp <- sprintf('%s/%s.csv',folder, name)
# read folder and name from command line or define them manually


#
gold.fp = sprintf('%s/%s_gold.csv', folder, name)
gold = read.csv(gold.fp)
#
selected.functions = c('DN','IN')
selected.sits = gold$gold %in% selected.functions
selected.sits = rep(TRUE,nrow(gold))

data.txt <- read.csv(fp,header=TRUE)
data.txt = data.txt[selected.sits,]
nCUTs <- length(2:ncol(data.txt))
tt <- data.txt [,-1]
names <- colnames(tt)
T <- (tt[,1:nCUTs])
hr <- rollcall(T, yea=1, nay=6, missing=9,notInLegis=8,desc= title,) 
   
minVOTES <- 5
min_freq = 5
LOP <- (1/nrow(data.txt))* min_freq
# minimum number of positive terms a situation must have
# minimal percentage of situations a term may occur in positive

##
# ACTUAL OC
##
ndims = 3
result <- oc(hr,dims=ndims, minvotes=minVOTES, lop=LOP, polarity = 1:ndims, verbose = TRUE)
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
data = cbind(coordinates, labels[selected.sits,], gold[selected.sits,])
data$situations = paste(situations[selected.sits,]$utterance,
  situations[selected.sits,]$word)

###
# PLOTTING
###

xlims = c(min(data$dim.1),max(data$dim.1))
ylims = c(min(data$dim.2),max(data$dim.2))

for (language in 5:34) {
  data.sub = data[data[,language] != '',]
  #
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > nrow(data.txt)/100 * min_freq])
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(dim.1, dim.2, color = data.subsub[,language], data = data.subsub)
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


# analyses of gradient
sorted.data = data[order(data$dim.1),]
sorted.data$indices = 1:nrow(sorted.data)
#
for (i in 4:34) {
  print(language.labs[i-3])
  print(as.numeric(sorted.data[,i]))
}
