#loading files
train <- read.csv("~/Desktop/STAT 542/Final Project/blogData_train.csv", header=FALSE)
test <- read.csv("~/Desktop/STAT 542/Final Project/blogData_test.csv", header=FALSE)
#analysis of data explorary analysis
train_y = train[,281]
test_y = test[,281]
stem(train_y)
#Here we find three variables are all zeros
#we cannot find any informationfrom these three variables
which(colSums(train[,-281]) == 0)
which(colSums(test[,-281]) == 0)
for(i in 1:63){
  for(j in 1:63){
    if (!is.na(cor(train[,i],train[,j])) && abs(cor(train[,i],train[,j]))>0.8 && i!=j){
      print(i)
      print(j)
    }
  }
}
#processing sparse data 
bag_words = train[ ,63:262]
bag_words = as.matrix(bag_words)
bag_words_test = as.matrix(test[ ,63:262])
#zero counts for each word
zero_count = colSums(bag_words == 0)
zero_per = colSums(bag_words == 0)/nrow(bag_words)*100
#zero counts for each observations
zero_pob = 100-rowSums(bag_words == 0)/200*100
sum_table = table(zero_pob)
hist(zero_pob[zero_pob != 100])
#PCA
#get top 5 loading from pca
pca = loadings(princomp(bag_words))[,1:5]
#SVD
library(irlba)
svd <- irlba(bag_words, nv=5, nu=0, center=colMeans(bag_words), right_only=TRUE)$v
svd_test <- irlba(bag_words_test, nv=5, nu=0, center=colMeans(bag_words_test), right_only=TRUE)$v
combine = bag_words%*%svd
combine_test = bag_words_test%*%svd_test
#just summerize
summerize = rowMeans(bag_words)
#seperation result to be 0 or 1
io_trainy = ifelse(train_y==0,0,1)
io_testy = ifelse(test_y==0,0,1)
#seperation train
strain =data.frame(x=cbind(train[,51:62],combine),y=as.factor(io_trainy))
stest = data.frame(x=cbind(test[,51:62],combine_test),y=as.factor(io_testy))
##using svm
library('e1071')
svm.radial = svm(y ~ ., data = strain, 
                                 type='C-classification', kernel='radial',
                                  scale=TRUE,cost = 1)
predict_test = predict(svm.radial , stest)
mean(predict_test == io_testy) #80.325%
##using random forest
library(randomForest)
rf.fit = randomForest(strain[,-1],strain$y, ntree = 100, mtry = 5)
pre_rf = predict(rf.fit,stest[,-1])
mean(pre_rf == io_testy)#100%
#since the result is amazing, I stop at random forest
#rf as classification is thought as the best way
#Regression prediction
