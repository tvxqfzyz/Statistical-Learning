#install.packages('kernlab')
#install.packages('cvTools')
library(kernlab)
library(ElemStatLearn)# loads the data frame SAheart
library('e1071')#SVM package
data(SAheart)
heart = SAheart
heart$famhist = as.numeric(heart$famhist)-1
n = nrow(heart)
p = ncol(heart)-1
#10-fold cross validation--seperation of data
set.seed(127) #Control the randomness to compare result
nfold = 10 #could change
infold = sample(rep(1:nfold, length.out=n))
cost = c(0.1,0.2,0.3,0.4,0.5,0.6,1)#set of cost selected
error_vec = matrix(NA,10,length(cost))
for(c in 1:length(cost)){
  for (f in 1:10){
    svm.linear = svm(chd ~ ., data = heart[infold != f,], 
                      type='C-classification', kernel='linear',
                      scale=TRUE,cost = cost[c])
    predict.linear = predict(svm.linear, heart[infold == f,])
    error_vec[f,c] = mean(predict.linear != heart[infold == f,10])
  }
}
#find the minimum cost
best.linear = which.min(apply(error_vec,2,mean))
best.linear
#linear
mean(error_vec[,best.linear])

#radial
error_radial = matrix(NA,10,length(cost))
for(c in 1:length(cost)){
  for (f in 1:10){
    svm.radial = svm(chd ~ ., data = heart[infold != f,], gamma = 1/p,
                           type='C-classification', kernel='radial',
                           scale=TRUE,cost = cost[c])
    predict.radial = predict(svm.radial, heart[infold == f,])
    error_radial[f,c] = mean(predict.radial != heart[infold == f,10])
  }
}
best.radial = which.min(apply(error_radial,2,mean))
best.radial
#radial
mean(error_radial[,best.radial])

#Polynomial
error_pol = matrix(NA,10,length(cost))
for(c in 1:length(cost)){
  for (f in 1:10){
    svm.pol= svm(chd ~ ., data = heart[infold != f,],
                     type='C-classification', kernel='polynomial',degree = 2,coef0 = 1,
                     scale= TRUE,cost = cost[c])
    predict.pol = predict(svm.pol, heart[infold == f,])
    error_pol[f,c] = mean(predict.pol != heart[infold == f,10])
  }
}
best.pol = which.min(apply(error_pol,2,mean))
best.pol
#Pol
mean(error_pol[,best.pol])

library(randomForest)
#Question 3 problem of degree freedom
#3.a Generate the X
set.seed(10)
n = 200
p = 4
x = matrix(rnorm(n*p), n, p )
#3.b Define the true model as linear bx+error
b = as.matrix(c(1, 10, 0.5, 7)) 
#3.c
y = x %*% b + rnorm(n)
#3.d
rf.fit = randomForest(x, y, ntree = 1000, mtry = 1, nodesize = 55)
#knn.fit = kknn(y~x, train = data.frame(x = x,y = y1), test = data.frame(x=x), k = 5,kernel = "rectangular",scale = FALSE)
y_hat = as.matrix( predict(rf.fit, x) )
#3.d
y_i = y
y_hi = y_hat
for (i in 1:10){
  y_temp = x %*% b + rnorm(n)
  y_i = cbind(y_i,y_temp)
  rf.fiti = randomForest(x, y_temp,ntree = 1000, mtry = 1, nodesize = 55)
  y_hi = cbind(y_hi,as.matrix(predict(rf.fiti,x)))
}
addup = 0
for (i in 1:n){
  addup = addup + cov(y_i[i,],y_hi[i,])
}
addup

