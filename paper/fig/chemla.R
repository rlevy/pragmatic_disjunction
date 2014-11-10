library(ggplot2)

dat <- read.csv("chemla-p2-table-gb2counts.csv",1)

dat$ratio <- dat$GB2.Freq.X.or.Y / dat$GB2.Freq.X

labels = paste(dat$X, 'or', dat$Y)

xlab = "Probability of X implicating not-Y"
ylab = "P(X or Y) / P(X)"


ggplot(dat,aes(x=p,y=ratio)) + geom_point(color='red') + geom_text(label=labels, size=5, angle=45) + stat_smooth(method="lm") + xlab(xlab) + ylab(ylab) +
    coord_cartesian(ylim = c(-0.0002, 0.0015)) 

fit = lm(ratio ~ p,dat)

summary(fit)
