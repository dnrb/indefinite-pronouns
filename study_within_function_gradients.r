# based on the R scripts retrieved from
# http://www.unm.edu/~wcroft/MDS.html (January 23rd, 2017)

library(oc)
library(ggplot2)

language.labs = c('ar','bg','bs','cz','da','de','el','en','es','et','fi',
  'fr', 'he', 'hr', 'hu', 'id', 'it', 'ja', 'nl', 'no', 'pl', 'pr', 'ro',
  'ru', 'sl','sr','sv','tr','vi','zh')
name = 'oc_SPLIT'
folder = '.'
fp = sprintf('%s/%s.csv',folder, name)

##############
# parameters #
##############

onto = 'thing' # 'body'
# ontological category to consider
fun = c('SP','NS')
ndims = 2
# number of dimensions in the OC analysis
min.terms = 5
# minimum number of positive terms a situation must have
min.freq = 5
# minimum number of situations a term must be applicable to
parameters = sprintf('onto=%s_dim=%d', onto, ndims)

#############
# read data #
#############
original.data = read.csv(sprintf('%s_gold.csv', name))
selected.sits = original.data$ontological == onto & original.data$annotation %in% fun
full.data = read.csv(fp,header=TRUE)
data = full.data[selected.sits,]
nCUTs = length(2:ncol(data))
tt = data[,-1]
names = colnames(tt)
T = (tt[,1:nCUTs])
hr = rollcall(T, yea=1, nay=6, missing=9,notInLegis=8,desc=title,) 
min.sits = (1/nrow(data)) * min.freq

##########
# 1-D OC #
##########
result <- oc(hr,dims=1, minvotes=min.terms, lop=min.sits, polarity = c('Legislator 1'), verbose = TRUE)
coordinates = data.frame(result$legislators[,7], row.names = 1:nrow(result$legislators))
colnames(coordinates) = c('dim.1')
labels.fp = sprintf('%s/%s_labels.csv', folder, name)
labels = read.csv(labels.fp)
coordinates = cbind(coordinates, labels[selected.sits,], original.data[selected.sits,])
coordinates$situations = paste(coordinates$utt, coordinates$word)
coordinates = coordinates[order(coordinates$dim.1),]
coordinates$indices = 1:nrow(coordinates)
coordinates.long = reshape(coordinates, varying = c(3:32,35), v.names = 'term', timevar = 'language', 
							times = c(3:32,35), direction = 'long')	
top = sort(table(coordinates.long$term), decreasing = TRUE)
top.n = names(top[top > (nrow(data)/100) * 1])
coordinates.long.sub = droplevels(coordinates.long[coordinates.long$term %in% top.n & coordinates.long$term != '',])
cairo_pdf(sprintf('1d_OC_%s_%s.pdf', onto, paste(fun, collapse = '_')), width = 30, height = 15)
q = qplot(as.factor(language), as.factor(indices), label = substr(term,1,9), color = term, data = coordinates.long.sub, geom = 'text')
q = q + scale_color_manual(values = rep(c('darkgrey','black'), nrow(coordinates.long.sub)/2+1)) 
q = q + guides(color=FALSE) + geom_text(size = 1) 
q = q + scale_x_discrete(labels= language.labs) 
q = q + scale_y_discrete(labels = coordinates.long$situation)
print(q)
dev.off()