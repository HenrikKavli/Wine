#programming and data analysis for business Retake
library(caret)
library(dplyr)
library(ggplot2)
library(gbm)
library(car)
library(naniar)
library(car)
library(caTools)
library(ggcorrplot)
library(ggstance)
library(stringr)
library(reshape2)
library(DescTools)
library(moments)
library(formattable)
library(tidyr)
library(dotwhisker)
library(pdp)
library(nnet)
library(ggpubr)


set.seed(321)
#importing dataset
wine <- read.csv("/Users/henrikkavli/Downloads/winequality-red.csv")

#checking basic properties, only contious variables, no 1,2,9,10 scores
str(wine)
summary(wine)

#change from 1:10 scale to 1:6 "dice" scale, given that no(1,2,9,10) quality scores are given 
old <- c(3:8)
new <- c(1:6)
wine$quality[wine$quality %in% old] <- new[match(wine$quality, old, nomatch = 0)]


#look at independent quality variable. Changing from 
funk <- function(x){length(wine$quality[wine$quality==x])}
a <- sapply(c(1:6),funk)
quality <- data.frame(value=c(1:6),count=a)

#plotting dice scores for all wine
ggplot(quality,aes(x=value,y=count,fill=count))+
  geom_bar(stat='identity')+
  labs(title="Quality Score",x=element_blank(),y=element_blank())+
  theme(plot.title=element_text(face="bold",size=14))+
  scale_x_continuous(limits=c(0.5,6.5))+
  theme(legend.position="none")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())


#Data cleaning:
length(is.na(wine)[is.na(wine)==TRUE]) #no missing values :)
length(duplicated(wine)[duplicated(wine)==TRUE]) #240 duplicates, 
#assumed to be same scores given to same wine by different people, therefore included

#plotting density and boxplot for all independent variables:
melt_wine <- melt(wine[,-12])
ggplot(data = melt_wine, aes(x = value, y = -0.3)) + 
  geom_boxploth(width=0.3) +
  stat_density(aes(x = value, y = stat(scaled)), inherit.aes = FALSE)+
  facet_wrap(~variable, scales = "free")+
  labs(title="Independent Variables",x=element_blank(),y=element_blank())+
  theme(plot.title=element_text(face="bold",size=14))+
  scale_y_continuous(breaks=NULL)+
  scale_x_continuous(breaks=NULL)
  
#we can see that there is a series of issues with the data. 
#Most variables are heavily positively skewed.
#Other variables have several severe outliers, such as residual sugar and chlorides

            
#visualize dependent and indipentent variables thorugh correlation plot
corr <- round(cor(wine),1)
ggcorrplot(corr, method="circle", hc.order = FALSE, type = "lower",
           insig="blank",outline.color = "white")+
  labs(title="Correlation plot")+
  theme(plot.title=element_text(face="bold",size=14))+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(legend.position="none")+
  scale_y_discrete(position='right')

#visualize how quality differs across all independent variables:
melt_wine2 <- data.frame(melt(wine),quality=rep(wine$quality,12))
ggplot(data = melt_wine2, aes(group=quality, y=value)) + 
  geom_boxplot() +
  facet_wrap(~variable, scales = "free")+
  labs(title="Quality Boxplot Independent Variables",x=element_blank(),y=element_blank())+
  theme(plot.title=element_text(face="bold",size=14))+
  scale_y_continuous(breaks=NULL)+
  scale_x_continuous(breaks=NULL)


#split dataset into training and testing (potentially also validation)
# use impute package later for testing:
train_split <- sample(nrow(wine), 0.8*nrow(wine)) 
train_wine <- wine[train_split,]
test_wine <- wine[-train_split,]


#make a matrix for testing linear regression matrix
train_matrix <- as.matrix(cbind(rep(c(1),1279),train_wine[,-12]))
train_y <- train_wine[,12]
beta_hat <- solve(t(train_matrix)%*%train_matrix)%*%t(train_matrix)%*%train_y 
#gives the same beta estimates as linear1 model

linear1 <- lm(quality ~., data=train_wine)
summary(linear1) #some insignificant variables, low r2
sort(vif(linear1)) #all are below 10

#looking at the residuals for the model
hist(residuals(linear1), breaks=30, freq=TRUE)
curve(dnorm(x,mean=0,sd=0.6396)) #looks fairly normal

qqline(residuals(linear1))
dwplot(linear1)
rmse1 <- sqrt(mean(residuals(linear1)^2))
lin_pred <- predict(linear1,test_wine)

rmse_lin <- sqrt(mean((test_wine$quality-lin_pred)^2))

#a second attempt at a linear model
linear2 <- lm(quality ~., data=train_wine[,-c(1,8)])
summary(linear2)
sort(vif(linear2))
qqline(residuals(linear2))
dwplot(linear2)+
  labs(title="Linear Dot Whisker")+
  theme(plot.title=element_text(face="bold",size=14))

#create predictions updated linear model
lin2_pred <- predict(linear2,test_wine)
round_lin2_pred <- as.factor(round(lin2_pred,0))
unique(round_lin2_pred) #only 3,4,5, need to add 1,2,6 factors manually
round_lin2_pred <- sapply(round_lin2_pred,factor,
                          levels=c("1","2","3","4","5","6"))

test_wine_quality <- as.factor(test_wine$quality) 
str(round_lin2_pred)

confusionMatrix(test_wine_quality, round_lin2_pred)


#beginning gradient boost specification, takes 10 sec to run
model_gbm <- gbm(train_wine$quality ~.,
                data = train_wine,
                distribution = "gaussian",
                bag.fraction = 0.5,
                shrinkage = 0.001,
                interaction.depth=32,
                n.trees = 5000)
print(model_gbm)
summary(model_gbm)
gbm.perf(model_gbm,method="OOB") #giving optimal number of iterations =2.2K

#respecifying for optimal iterations=K trees= 2277
model_gbm2 <- gbm(train_wine$quality ~.,
                 data = train_wine,
                 distribution = "gaussian",
                 bag.fraction = 0.5,
                 shrinkage = 0.001,
                 interaction.depth=32,
                 n.trees = 2277)
print(model_gbm2)
summary(model_gbm2) #alcohol, suplates and volatile acidity are key drivers


gbm_pred <- predict(model_gbm2,test_wine)
round_gbm_pred <- as.factor(round(gbm_pred,0))
unique(round_gbm_pred) 
round_gbm_pred <- sapply(round_gbm_pred,factor,
                          levels=c("1","2","3","4","5","6"))

confusionMatrix(test_wine_quality, round_gbm_pred)
rmse_gbm <- sqrt(mean((test_wine$quality-gbm_pred)^2))

(0.674-0.6411)/0.674 #gives a 5% improvement in average error over multiple linear


#creating pdp plots for variables
part_alc <- partial(model_gbm, pred.var = 'alcohol',
                    n.trees=219 ,chull = TRUE)
p1 <- autoplot(part_alc, contour = TRUE)+
labs(y="predicted quality score")

part_vol <- partial(model_gbm, pred.var = 'volatile.acidity',
                    n.trees=219 ,chull = TRUE)
p2 <- autoplot(part_vol, contour = TRUE)+
labs(y="predicted quality score")

part_sul <- partial(model_gbm, pred.var = 'sulphates',
                    n.trees=219 ,chull = TRUE)
p3 <- autoplot(part_sul, contour = TRUE)+
labs(y="predicted quality score")

ggarrange(p1,p2,p3, ncol=3, nrow=1)+
  labs(title="Partial Dependency Plots")+
  theme(plot.title.position = "plot",plot.title=element_text(face="bold",size=14))

w1. <- length(wine$quality[wine$quality==3])
w2. <-  length(wine$quality[wine$quality==4])
w3. <-   length(wine$quality)

mean(wine$quality)
(w1.+w2.)/w3.

#test model on testing data, do figures for 
# feature importance, plot pdp for most important features,
# do oos testing, create classitier at T=0.5

#try multinomial logistic regression, gradient boost at T=0.5 is clearly better
train_wine$quality<- as.factor(train_wine$quality)
train_wine$quality <- relevel(train_wine$quality, ref="1")
multilog <- multinom(quality ~ ., data=train_wine)

summary(multilog)
multilog_pred <- as.factor(predict(multilog,test_wine))
confusionMatrix(test_wine_quality,multilog_pred) #not as good as gradient boost

#creating biance variance illustration
random %>% ggplot(aes(x=n,y=verdi))+
  geom_point(size=1,color='black')+
  geom_line(size=0.5,color='darkgray',linetype='dashed')+
  geom_smooth(size=0.5,method = lm, color='deepskyblue3')+
  geom_smooth(size=0.5,method = "loess", color='deepskyblue4')+
  labs(title="Bias Variance Tradeoff",x=element_blank(),y=element_blank())+
  theme(plot.title=element_text(face="bold",size=14))+
  scale_x_continuous(breaks=NULL)+
  scale_y_continuous(breaks=NULL)

plot(wine$fixed.acidity,wine$pH)
cor(wine$fixed.acidity,wine$pH)

#creating importance plot manually
importance <- data.frame(name=c("alcohol","sulphates","volatile.acidity","total.sulfur.dioxide",
                      "chlorides","pH","citric.acid","density,free","sulfur.dioxide",
                      "fixed.acidity","residual.sugar"),
                      count=c(26.851238,15.434763,12.907959,9.273619,6.329221,5.476104,5.352128,
                      5.020723,4.663401,4.579880,4.110963))

ggplot(importance,aes(x=count,y=reorder(name,count),fill=count))+
  geom_bar(stat="identity")+
  labs(title="Importance  Plot",x=element_blank(),y=element_blank())+
  theme(plot.title=element_text(face="bold",size=14))+
  theme(legend.position="none")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())


install.packages("corrr")
library('corrr')
install.packages("factoextra")
library("factoextra")

print(wine)
str(wine)

length(is.na(wine)[is.na(wine)==FALSE])

print(corr)

wine_pca <- princomp(corr)
summary(wine_pca)

wine_pca$loadings[, 1:2]

fviz_eig(wine_pca)

eigen(corr)

fviz_pca_var(wine_pca, col.var = "black")

fviz_cos2(wine_pca, choice = "var", axes = 1:2)

barplot(eigen(corr)$value)
