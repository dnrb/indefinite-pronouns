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
   
minVOTES <- 1
# minimum number of positive terms a situation must have
LOP <- .001
# minimal percentage of situations a term may occur in positive
min_freq = 5
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
data = cbind(coordinates, labels, gold)

xlims = c(min(data$dim.1),max(data$dim.1))
ylims = c(min(data$dim.2),max(data$dim.2))

for (language in 5:35) {
  data.sub = data[data[,language] != '',]
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > nrow(data.txt)/100 * min_freq])
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(dim.1, dim.2, color = data.subsub[,language], data = data.subsub)
  if (language == 35) {
    q = qplot(dim.1, dim.2, color = data.subsub[,language], 
      label = data.subsub[,language], data = data.subsub, geom = 'text')
  }
  q = q + xlim(xlims)
  q = q + ylim(ylims)
  ggsave(sprintf('%s/%s_L%d.pdf', folder, name, language), q, height = 7, width = 7)  
}
  
centroid = aggregate(cbind(dim.1,dim.2) ~ gold, data = data, FUN = mean)
q = qplot(dim.1, dim.2, label = gold, data = centroid, geom = 'text')
ggsave(sprintf('%s/%s_centroid.pdf', folder, name), q)
