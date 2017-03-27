require(MASS)
require(dr)
require(CCA)

#Question 1
set.seed(1)
n = 5000
p = 10
V = 0.5^abs(outer(1:p, 1:p, "-"))  
x = mvrnorm(n, runif(p), V)
b1 = matrix(c(sqrt(0.5), 0, sqrt(0.5), 0, rep(0, p-4)))
b2 = matrix(c(0, sqrt(0.5), 0, -sqrt(0.5), rep(0, p-4)))
dir1 = x %*% b1
dir2 = x %*% b2
link = dir1 + 4*atan(2*dir2)
y = link + 0.5*rnorm(n)
# use the dr package, and perform SIR using 10 slices 
fit.sir = dr(y~., data = data.frame(x, y), method = "sir", nslices = 10)
fit.sir$evectors[, 1:2]
cc(fit.sir$evectors[, 1:2], cbind(b1, b2))$cor
##############################################################
# write your own code for SIR to get the first two direction
# calculate the smaple covariance for x first
cov_x = cov(x)
# inverse of sqaure root of matrix
mat.sqrt.inv<-function(A)
{
  ei<-eigen(A)
  d<-ei$values
  d<-(d+abs(d))/2
  d2<-1 / sqrt(d)
  d2[d == 0]<-0
  ans<-ei$vectors %*% diag(d2) %*% t(ei$vectors)
  return(ans)
}
# 1.centering the columns
standard_x = apply(x,2,function(y) y-mean(y))
R = mat.sqrt.inv(cov_x)
z = standard_x %*% R
# 2.sort z by y
mydata = data.frame("z" = z ,"y" = y)
mydata = mydata[order(mydata$y), ]
#divide the dataset to 10 pieces
H = 10
id = split(1:n, cut(seq_along(1:n), H, labels = FALSE))
# 3.for each slice compute the mean of z
means = sapply(1:H, function(h, mydata, id) colMeans(mydata[id[[h]], ]), mydata, id)
# 4.compute the covariance matrix for the slice means of z
means = means[1:p,]
cov.slice<-function(means,z){
  cov_mean = (means[,1]-colMeans(z))%*%t(means[,1]-colMeans(z))
  for (h in 2:10){
  cov_mean = cov_mean + (means[,h]-colMeans(z))%*%t(means[,h]-colMeans(z))
  }
  return(cov_mean/10)
}
cov_mean = cov.slice(means,z)
#run pca analysis find the eigen vectors
pca = eigen(cov_mean)
#get the first two directions and multiply back the R
sir = R%*%pca$vectors[,1:2]
cc(sir, cbind(b1, b2))$cor

#Question 2
library(MASS)
library(ElemStatLearn)
library(gbm)
data_spam = spam
data_spam$spam = as.numeric(factor(data_spam$spam,levels=c("email","spam"),labels = c(0,1)))-1
nrow(data_spam[data_spam[,'spam']==1,])
#1813
nrow(data_spam[data_spam[,'spam']==0,])
#2788
#cutoff of the vote for classification
cutoff = sum(data_spam$spam) / nrow(data_spam)
#perform exponential loss Adaboost
gbm.ada = gbm(spam ~ .,distribution="adaboost", data = data_spam,n.trees=50, shrinkage=1, bag.fraction=0.8,cv.folds = 5)
gbm.perf(gbm.ada)
pre_ada = predict(gbm.ada,data_spam)
ada_loss_rate = mean((pre_ada > cutoff) != data_spam$spam)
ada_loss_rate #0.07041947
#more trees iteration with smaller shrinkage 
gbm.ada.e = gbm(spam ~ .,distribution="adaboost", data = data_spam,n.trees=250, shrinkage=0.1, bag.fraction=0.8,cv.folds = 5)
gbm.perf(gbm.ada.e)
pre_ada.e = predict(gbm.ada.e,data_spam)
ada.e_loss_rate = mean((pre_ada.e > cutoff) != data_spam$spam)
ada.e_loss_rate #0.06302978
#better or not with even smaller shrinkage value
gbm.ada.b = gbm(spam ~ .,distribution="adaboost", data = data_spam,n.trees=2000, shrinkage=0.01, bag.fraction=0.8,cv.folds = 5)
pre_ada.b = predict(gbm.ada.b,data_spam)
ada.b_loss_rate = mean((pre_ada.b > cutoff) != data_spam$spam)
ada.b_loss_rate #0.0695501
#gbm.perf(gbm.ada.b)
#perform logistic likelihood loss Bernoulli
gbm.ber = gbm(spam ~ .,distribution="bernoulli", data = data_spam,n.trees=100, shrinkage=1, bag.fraction=0.8,cv.folds = 5)
gbm.perf(gbm.ber)
pre_ber = predict(gbm.ber,data_spam)
ber_loss_rate = mean((pre_ber > cutoff) != data_spam$spam)
ber_loss_rate #0.04433819
#more trees iteration with smaller shrinkage 
gbm.ber.e = gbm(spam ~ .,distribution="bernoulli", data = data_spam,n.trees=1200, shrinkage=0.1, bag.fraction=0.8,cv.folds = 5)
gbm.perf(gbm.ber.e)
pre_ber.e = predict(gbm.ber.e,data_spam)
ber.e_loss_rate = mean((pre_ber.e > cutoff) != data_spam$spam)
ber.e_loss_rate #0.03912193