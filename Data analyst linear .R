#STATS 542 Homework 1
#Question 1
install.packages('ElemStatLearn')
install.packages('kknn')
install.packages('class')
install.packages('leaps')

set.seed(5)
library(ElemStatLearn)
train = zip.train
test = zip.test
train_set = train[(train[,1]==2)|(train[,1]==3), ,drop = FALSE]
train_y = train_set[,1]
train_x = train_set[,-1]
test_set = test[(test[,1]==2)|(test[,1]==3), ,drop = FALSE]
test_y = test_set[,1]
test_x = test_set[,-1]
library('kknn')
library('class')
#10-fold Cross Validation
nfold = 10
train_size = dim(train_x)[1]
infold = sample(rep(1:nfold, length.out=train_size))
mydata = data.frame(x = train_x,y = train_y)
K = 15 #The maximun k values considered
errorMatrix = matrix(NA, K, nfold) # save the prediction error of each fold
for(l in 1:nfold){
  for (k in 1:K){
    knn.fit = knn(train_x[infold != l, ],train_x[infold == l, ],train_y[infold != l], k = k)
    errorMatrix[k, l] = mean(knn.fit != train_y[infold == l])
  }
}
which.min(apply(errorMatrix, 1, mean))
plot(rep(1:K, nfold), as.vector(errorMatrix),main = '10-fold Cross Validation',xlab = 'Value of K',ylab = 'Mean of Error', pch = 19, cex = 0.5)
points(1:K, apply(errorMatrix, 1, mean), col = "red", pch = 19, type = "l", lwd = 3)
#2)Perform kNN on the entire trainning data and validate on the testing data
error = matrix(NA, K)
train_error = matrix(NA, K)
for (k in 1:K){
  knn.fit = knn(train_x,test_x,train_y,k = k)
  knn_chun = knn(train_x,train_x,train_y,k = k)
  error[k,1] = mean((knn.fit != test_y))
  train_error[k,1] = mean((knn_chun != train_y ))
}
which.min(apply(train_error, 1, mean))
which.min(apply(error, 1, mean))
plot(rep(1:K),as.vector(error),main = 'Test Error for K',xlab = 'Value of K',ylab = 'Error',pch = 19, cex = 0.5)
plot(rep(1:K),as.vector(train_error),main = 'Best K Selection',xlab = 'Value of K',ylab = 'Error',pch = 19, cex = 0.5)
#points(1:K, apply(error, 1, mean), col = "red", pch = 19, type = "l", lwd = 3)

#Question 2 problem of degree freedom
#2.b Generate the X
set.seed(10)
n = 200
p = 4
x = matrix(rnorm(n*p), n, p )
#2.c Define the true model as 2*sin(x)
b = as.matrix(c(1, 10, 0.5, 0)) 
y1 = x %*% b + rnorm(n)
#2.d
knn.fit = kknn(y~x, train = data.frame(x = x,y = y1), test = data.frame(x=x), k = 5,kernel = "rectangular",scale = FALSE)
y_h1 = knn.fit$fitted.values
#2.e
y_i = y1
y_hi = y_h1
for (i in 1:10){
  y_temp = x %*% b + rnorm(n)
  y_i = cbind(y_i,y_temp)
  knn.fiti = kknn(y~x, train = data.frame(x = x,y = y_temp), test = data.frame(x=x), k = 5,kernel = "rectangular",scale = FALSE)
  y_hi = cbind(y_hi,knn.fiti$fitted.values)
}
addup = 0
for (i in 1:n){
  addup = addup + cov(y_i[i,],y_hi[i,])
}
addup

#Question 3 Prostate Problem
#Get the data from package
data = prostate
#standardize the data to unit variance
for (i in 1:8){
  data[,i] = (data[,i]-mean(data[,i]))/sqrt(var(data[,i])) 
}
#get the train data
data_train =  data[data$train == 1,]
#Set up data frame
p_cancer = data.frame(cbind(data_train[,1:8], "Y" = data_train[,9]))
#execute OLS regression on trainning dataset
lmfit = lm(Y~.,data = p_cancer)
summary(lmfit)

#analysis of regression
n = nrow(p_cancer)
p = 9
#Calculate AIC
extractAIC(lmfit, k = 2) # a build-in function for calculating AIC using -2log likelihood
#BIC
extractAIC(lmfit, k = log(n))

#Perform stepwise regression
step(lmfit, direction="both")#AIC
step(lmfit, direction="both", k=log(n))#BIC
step(lmfit, direction="backward")#backward direction

#best subset selection
library(leaps)
RSSleaps=regsubsets(Y~.,data = p_cancer)
sumleaps=summary(RSSleaps,matrix=T)
names(sumleaps)  # components returned by summary(RSSleaps)

sumleaps$which
msize=apply(sumleaps$which,1,sum)
Cp = sumleaps$rss/(summary(lmfit)$sigma^2) + 2*msize - n
AIC = n*log(sumleaps$rss/n) + 2*msize
BIC = n*log(sumleaps$rss/n) + msize*log(n)
cbind(Cp, sumleaps$cp)
cbind(BIC, sumleaps$bic)   
BIC-sumleaps$bic  
n*log(sum((data_train[,9] - mean(data_train[,9] ))^2/n)) 

# Rescale Cp, AIC, BIC to (0,1).
inrange <- function(x) { (x - min(x)) / (max(x) - min(x)) }

Cp = sumleaps$cp; Cp = inrange(Cp);
BIC = sumleaps$bic; BIC = inrange(BIC);
AIC = n*log(sumleaps$rss/n) + 2*msize; AIC = inrange(AIC);


plot(range(msize), c(0, 1.1), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize, Cp, col="red", type="b")
points(msize, AIC, col="blue", type="b")
points(msize, BIC, col="black", type="b")
legend("topright", lty=rep(1,3), col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))
