# based on the R scripts retrieved from
# http://www.unm.edu/~wcroft/MDS.html (January 23rd, 2017)

library(oc)
library(ggplot2)

language.labs = c('ar','bg','bs','cz','da','de','el','en','es','et','fi',
  'fr', 'he', 'hr', 'hu', 'id', 'it', 'ja', 'nl', 'no', 'pl', 'pr', 'ro',
  'ru', 'sl','sr','sv','tr','vi','zh')
name <- 'oc_SPLIT_test'
folder <- '.'
fp <- sprintf('%s/%s.csv',folder, name)

##############
# parameters #
##############
onto = 'thing' # 'body'
# ontological category to consider
ndims = 2
# number of dimensions in the OC analysis
min.terms = 1
# minimum number of positive terms a situation must have
min.freq = 1
# minimum number of situations a term must be applicable to
parameters = sprintf('onto=%s_dim=%d', onto, ndims)

#############
# read data #
#############
original.data = read.csv(sprintf('%s_gold.csv', name))
selected.sits = original.data$ontological == onto

full.data = read.csv(fp,header=TRUE)
data = full.data[selected.sits,]
nCUTs = length(2:ncol(data))
tt = data[,-1]
names = colnames(tt)
T = (tt[,1:nCUTs])
hr = rollcall(T, yea=1, nay=6, missing=9,notInLegis=8,desc=title,) 
min.sits = (1/nrow(data)) * min.freq
# minimal percentage of situations a term may occur in positive

#############
# actual OC #
#############
result = oc(hr, dims=ndims, minvotes= min.terms,
  lop= min.sits, polarity = 1:ndims, verbose = TRUE)
df = data.frame(result$legislators[,7], row.names = 1:nrow(result$legislators))
colnames(df) = c('dim.1')
for (i in 1:ndims-1) { df[[sprintf('dim.%d',i+1)]] = result$legislators[,7+i] }
write.csv(df, sprintf('%s/%s_%s.csv', folder, name, parameters))

###############
# plot resuls #
###############
coordinates.filename = sprintf('%s/%s_%s.csv', folder, name, parameters)
coordinates = read.csv(coordinates.filename, sep = ',', header = TRUE)
labels.fp = sprintf('%s/%s_labels.csv', folder, name)
labels = read.csv(labels.fp)
data.a = labels[selected.sits,]
data.a$dim.1 = coordinates$dim.1
data.a$dim.2 = coordinates$dim.2
data.a$annotation = original.data[selected.sits,]$annotation
data.a$situations = paste(original.data[selected.sits,]$utt,
  original.data[selected.sits,]$word)
data.a = droplevels(data.a[data.a$annotation != 'UF',])

flip = 1 - 2 * (onto != 'body')
xlims = c(min(-data.a$dim.1),max(-data.a$dim.1))
ylims = c(min(data.a$dim.2 * flip),max(data.a$dim.2 * flip))

library(Cairo)
for (language in 2:31) {
  data.sub = data.a[data.a[,language] != '',]
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > nrow(data.a)/100 * 2])#min_freq])
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(-dim.1, flip * dim.2, color = data.subsub[,language], label = data.subsub[,language], size = 40, data = data.subsub, geom = 'text')
  q = q + xlim(xlims)
  q = q + ylim(ylims)
  q = q + theme(axis.title.x = element_blank(), axis.title.y = element_blank())
  q = q + guides(color = (show=FALSE), size = (show= FALSE))
  CairoPNG(sprintf('%s/plots/%s_%s_%s.png', folder, parameters, name, language.labs[language-1]), height = 600, width = 600)
  print(q)
  dev.off()
}


library(RColorBrewer)
my.cols <- function(n) {
  black <- "#000000"
  if (n <= 9) {
    c(black,brewer.pal(n-1, "Dark2"))
  } else {
    c(black,hcl(h=seq(0,(n-2)/(n-1),
                  length=n-1)*360,c=100,l=20,fixup=TRUE))
  }
}
 
myColors = my.cols(8)
names(myColors) = levels(original.data$annotation)[0:8]
sb= scale_colour_manual(name = 'annotation', values = myColors)

#sb = scale_colour_brewer(name = levels(original.data$annotation), palette = "Dark2")

q = qplot(-dim.1, flip * dim.2, color = annotation, label = annotation, size = 1, data = data.a, geom = 'text') 
q = q + theme(axis.title.x = element_blank(), axis.title.y = element_blank()) 
q = q + guides(size = (show= FALSE), col = (show=FALSE))
q = q + sb
ggsave(sprintf('%s/plots/%s_%s_annotations.png', folder, parameters, name), q, height = 6, width = 6)

####
# for the DN gradient for people
####
FN = 'DN'
xlims = c(min(-data.a[data.a$annotation == FN,]$dim.1)-0.3,max(-data.a[data.a$annotation == FN,]$dim.1)+0.3)
ylims = c(min(flip * data.a[data.a$annotation == FN,]$dim.2)-0.1,max(flip * data.a[data.a$annotation == FN,]$dim.2) + 0.1)
for (language in 2:31) {
  data.sub = data.a[data.a[,language] != '' & data.a$annotation == FN,]
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > 2])#min_freq])
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(-dim.1, flip * dim.2, color = data.subsub[,language], label = data.subsub[,language], size = 20, data = data.subsub, geom = 'text')
  q = q + xlim(xlims)
  q = q + ylim(ylims)
  q = q + theme(axis.title.x = element_blank(), axis.title.y = element_blank())
  q = q + guides(color = (show=FALSE), size = (show= FALSE))
  cairo_pdf(sprintf('%s/plots/%s_%s_%s_%s.png', folder, FN, parameters, name, language.labs[language-1]), height = 6, width = 2)
  print(q)
  dev.off()
}
