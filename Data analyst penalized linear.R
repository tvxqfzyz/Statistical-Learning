#Question 1
hour = read.csv("~/Desktop/STAT 542/HW2/hour.csv")
total = length(hour$instant)
#combine the columns of factors that needed for the initial slection
#transfer the date to numeric class
hour$dteday = as.Date(hour$dteday, "%m/%d/%Y")
#log transfer
log_cnt = log(hour$cnt+1)
selected = c('dteday','hr','season','holiday','workingday','weathersit','temp','atemp','hum','windspeed','cnt','train')
train_factors = subset(hour,select  = selected )
#set the target prediction as next hour not the current
train_factors_o = train_factors[-total,]
train_y_o= log_cnt[-1]
#seperate the data into train and test parts
train_factors = train_factors_o[train_factors_o[,12] == TRUE,]
test_factors = train_factors_o[train_factors_o[,12] == FALSE,]
train_y = train_y_o[train_factors_o[,12] == TRUE]
test_y = train_y_o[train_factors_o[,12] == FALSE]
#check the correlation
cor(hour$temp,hour$atemp)
#execute OLS method including first and second order
hour_data_train = data.frame(cbind(train_factors[,2:6],train_factors[,8:11]),"Y" = train_y)
ols_fit_1 = lm(Y~.,data = hour_data_train)
summary(ols_fit_1)
hour_data_test = data.frame(cbind(test_factors[,2:6],test_factors[,8:11]))
hour_predicted = predict(ols_fit_1,hour_data_test)
#the formulat of the error
RMSE = function(y_hat, y) {
  n = length(y_hat)
  e = sqrt(sum((y_hat - y)^2) / n)
  return(e)
}
#Calculation the error
hour_ols_error = RMSE(hour_predicted,test_y)
#OLS with second order
ols_fit_2 = lm(Y~.*.-holiday:workingday,data = hour_data_train)
summary(ols_fit_2)
hour_predicted_2 = predict(ols_fit_2,hour_data_test)
hour_ols_error_2 = RMSE(hour_predicted_2,test_y)
#stepwise to choose the best subset of second order
library(leaps)
n = nrow(hour_data_train)
p = ncol(hour_data_test)
step(ols_fit_2, direction="both", k=log(n),trace = FALSE)#BIC

setp_selection = lm(formula = Y ~ hr + season + holiday + workingday + weathersit + 
                      atemp + hum + windspeed + cnt + hr:workingday + hr:atemp + 
                      hr:hum + hr:cnt + season:hum + season:cnt + holiday:cnt + 
                      workingday:hum + workingday:cnt + weathersit:windspeed + 
                      weathersit:cnt + atemp:hum + atemp:cnt + hum:cnt, data = hour_data_train)
summary(setp_selection)
hour_predicted_best = predict(setp_selection,hour_data_test)
hour_ols_error_best = RMSE(hour_predicted_best,test_y)
#since the lm.ridge is not invariant with each scale we probably
#Ridge
library(glmnet) # library for both lasso (alpha=1) and ridge (alpha=0)
library(ncvreg)
library(MASS)
library(lars)
ridge.fit = glmnet(as.matrix(hour_data_train[,-10]), hour_data_train[,10], alpha=0)
plot(ridge.fit)
dim(coef(ridge.fit))
length(ridge.fit$lambda)
#Lamda sequences
cv.ridge = cv.glmnet(as.matrix(hour_data_train[,-10]), hour_data_train[,10], alpha=0)
plot(cv.ridge)
bestlam.ridge = cv.ridge$lambda.1se
prediction_ridge=predict(ridge.fit,s=bestlam.ridge ,newx=as.matrix(hour_data_test))
RMSE(prediction_ridge,test_y)

#Lasso Fitting
lasso.fit = glmnet(as.matrix(hour_data_train[,-10]), hour_data_train[,10])
plot(lasso.fit)
cv.lasso = cv.glmnet(as.matrix(hour_data_train[,-10]), hour_data_train[,10])
plot(cv.lasso)
# use the lambda.min from the cross-validation as the tuning
bestlam.lasso = cv.lasso$lambda.min
lassofit.coef = predict(lasso.fit, s=bestlam.lasso, type="coefficients")
lassofit.coef
prediction_lasso = predict(cv.lasso, s="lambda.1se", newx=as.matrix(hour_data_test))
RMSE(prediction_lasso,test_y)
# MCP
mcp.fit = cv.ncvreg(as.matrix(hour_data_train[,-10]), hour_data_train[,10], penalty="MCP", gamma = 3)
plot(mcp.fit)
mcp.beta = mcp.fit$fit$beta[, mcp.fit$min]
mcp.beta
prediction_mcp = predict(mcp.fit, as.matrix(hour_data_test), which = mcp.fit$min)
RMSE(prediction_mcp,test_y)
#plot(ncvreg(as.matrix(hour_data_train[,-10]), hour_data_train[,10], penalty="MCP", gamma = 3), lwd = 3)
#SCAD
scad.fit = cv.ncvreg(as.matrix(hour_data_train[,-10]), hour_data_train[,10], penalty="SCAD", gamma = 3)
plot(scad.fit)
scad.beta = scad.fit$fit$beta[, scad.fit$min]
scad.beta
prediction_scad = predict(scad.fit, as.matrix(hour_data_test), which = scad.fit$min)
RMSE(prediction_scad,test_y)
#Question 2
#read-in the data with read-table
#change the directory if necessary
#setwd("~/Desktop/STAT 542/HW2")
vowel_train = read.table("vowel.train.data", header = TRUE)
vowel_test = read.table("vowel.test.data", header = TRUE)
#Part I: write the function without package
QDA = function(train_x,train_y,test_x,test_y){
  #Estimate the  QDA parameters from training data
  #1.Prior probabilities
  n = length(unique(train_y)) # number of classes
  n_k = length(train_y)/n
  pi_k = numeric(n) #estimation of pior
  #2.Centroid
  mu_k =  list() 
  #3.Sample covariance matrix for each class
  sigma_k  =  list()
  for(i in 1:n){                        
    pi_k[i] = sum(train_y == i)/length(train_y)
    mu_k[[i]] = as.matrix(colSums(train_x[train_y == i,]))/n_k
    sigma_k[[i]] = cov(train_x[train_y == i,])
  }
  decision = matrix(NA, ncol = n, nrow = nrow(test_x)) #Prepare for MAP decision
  for(i in 1:n){
    for(k in 1:nrow(test_x)){
      #MAP decision fomula on note 24
      decision[k,i] = -0.5*log(det(sigma_k[[i]])) - 0.5*as.matrix((test_x[k,]-mu_k[[i]]))%*%solve(sigma_k[[i]])%*%
          as.matrix(t(train_x[k,]-mu_k[[i]]))
        +log(pi_k[i])
    }
  }
  # argmax the decision rule
  prediction <- apply(decision,1,which.max)
  #Error Matrix
  table(test_y, prediction)
  # The rate of miss classification
  return(mean(test_y != prediction))
}
QDA(vowel_train[,-1],vowel_train[,1],vowel_test[,-1],vowel_test[,1])
#the result is 0.525974
#Part II: QDA,LDA,logistic
#QDA
num = nrow(vowel_train)
qda.fit = qda(y~.,data = vowel_train)
summary(qda.fit)
#qda test error
prediction_qda = predict(qda.fit, vowel_test)$class
qda_test_error = sum( prediction_qda != vowel_test[,1])/num
qda_test_error
#LDA
lda.fit = lda(y~., data = vowel_train)
summary(lda.fit)
#lda test error
prediction_lda = predict(lda.fit, vowel_test)$class
lda_test_error = sum( prediction_lda != vowel_test[,1])/num
lda_test_error
# Multi-class logistic
# we cannot use glm here since y has several classes
logistic.fit = multinom(y~., family = binomial, data = vowel_train)
summary(logistic.fit)
#logistic error
prediction_logistic = predict(logistic.fit, vowel_test)
logistic_test_error = sum( prediction_logistic != vowel_test[,1])/num
logistic_test_error
