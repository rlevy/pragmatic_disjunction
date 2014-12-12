library(ggplot2)

dat <- read.csv("../fig/chemla-p2-table-gb2counts.csv",1)

dat$ratio <- dat$GB2.Freq.X.or.Y / dat$GB2.Freq.X

labels = paste(dat$X, 'or', dat$Y)

xlab = "Probability of X implicating not-Y"
ylab = "P(X or Y) / P(X)"

xlab.size = 20
axis.size = 16

## For text:
sizes = rep(NA,length(labels))
## We display this subset:
fordisplay = c(1,3,4,7,8,10,14,19,21,22,26,35)
sizes[fordisplay] = 8


ggplot(dat,aes(x=p,y=ratio)) +
geom_point(color='#990000',size=5) +
geom_text(label=labels, size=sizes, angle=65) +
stat_smooth(method="lm")+ xlab(xlab) + ylab(ylab) +
coord_cartesian(ylim = c(-0.0003, 0.0015)) +
theme(axis.title.x = element_text(face="bold", colour="#990000", size=xlab.size), axis.text.x=element_text(size=axis.size)) +
theme(axis.title.y = element_text(face="bold", colour="#990000", size=xlab.size), axis.text.y=element_text(size=axis.size))


fit = lm(ratio ~ p,dat)

summary(fit)
