#set working directory externally
#read in data
cancer = read.csv("data.csv", header = TRUE, row.names = "id")

#examine data
View(cancer)

#load libraries
library(caret)
library(dplyr)
library(leaps)
library(caret)
library(e1071)
library(tibble)
library(Deducer)
library(ggplot2)
library(class)
library(rknn)

#set seed for data splitting
set.seed(1234)

#no missing values in this diagnostic data set
length(which(is.na(cancer)))

#find out how many observations and variables there are
ncol(cancer)
nrow(cancer)

#determine how many malignant and benign cases there are
nrow(cancer[cancer["diagnosis"] == "B",])
nrow(cancer[cancer["diagnosis"] == "M",])

#split data 

#training data set will include 300 observations
train.idx = sample(nrow(cancer), 300)
cancer.train = cancer[train.idx,]

#everything that isn't in training set is in cancer.rest,
#which will be further divided up into a validation and test set
cancer.rest = cancer[-train.idx,]
test.idx = sample(nrow(cancer.rest), 100)
cancer.valid = cancer.rest[test.idx,]
cancer.test = cancer.rest[-test.idx, ]

#check that the division worked
nrow(cancer)
nrow(cancer.train) + nrow(cancer.valid) + nrow(cancer.test)

#check that there is an even distribution of benign and malignant 
#cases in the training, validation, and test sets
summary(cancer.train$diagnosis)
summary(cancer.valid$diagnosis)
summary(cancer.test$diagnosis)

###########################
###########################
###                     ###
### Logistic Regression ###
###                     ###
###########################
###########################

#put variables in order of importance
#while testing out logistic regression
glm.mod = train(diagnosis ~ ., data = cancer.train, 
                method = "glm", family = "binomial")
import.vars = varImp(glm.mod)$importance %>% 
                  as.data.frame() %>%
                  mutate(rowname = rownames(.))%>%
                  arrange(desc(Overall))
vars = import.vars$rowname

#add each variable in the list of ranked variables
#iteratively to the model and compute the accuracy
#performs a variant of forward selection
varnum = 1:length(vars)
accuracy = rep(NA, length(vars))
for (ii in 1:length(vars)){
  #generate a formula including ii variables
  Formula <- formula(paste("diagnosis ~ ", 
                           paste(vars[1:ii], collapse=" + ")))
  
  #create the model
  partial.mod = glm(Formula, data = cancer.train, family = binomial)
  
  #predict the probability of malignancy
  cancer.probs.part = predict(partial.mod, cancer.valid, type = "response")
  
  #assign M to those observations with above 50% probability of 
  #malignancy, otherwise assign B
  cancer.pred.part = rep("B", nrow(cancer.valid))
  cancer.pred.part[cancer.probs.part >0.5] = "M"
  
  #store the accuracy of the model
  accuracy[ii] = mean(cancer.pred.part == cancer.valid$diagnosis)
}

#Plot the accuracy versus number of variables
plot(x = varnum, y = accuracy, ylab = "Accuracy", 
     xlab = "Number of Variables")
which.max(accuracy)

#formula based on the most important 10 variables
Formula <- formula(paste("diagnosis ~ ", 
                         paste(vars[1:10], collapse=" + ")))

#create the model
glm.mod.final = glm(Formula, data = cancer.train, family = binomial)

#predict the probability of malignancy
glm.probs.final = predict(glm.mod.final, cancer.valid, type = "response")

#assign M to those observations with above 50% probability of 
#malignancy, otherwise assign B
glm.pred.final = rep("B", nrow(cancer.valid))
glm.pred.final[glm.probs.final >0.5] = "M"

#find the mean accuracy rate
mean(glm.pred.final == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction = glm.pred.final)

#plot the ROc curve
rocplot(glm.mod.final) +
  ggtitle("Logistic Regression ROC Curve")

#look at the model summary
summary(glm.mod.final)


#############################
#############################
###                       ###
### SVMs w/Linear Kernels ###
###                       ###
#############################
#############################

#put variables in order of importance
#while testing out SVM with linear kernels
svm.mod = train(diagnosis ~ ., data = cancer.train,
                method = "svmLinear")

svm.var.import = varImp(svm.mod)$importance %>% 
                    as.data.frame() %>%
                    mutate(rowname = rownames(.)) %>%
                    arrange(desc(B), desc(M))

svm.vars = svm.var.import$rowname

#implement SVMs with linear kernels with different
#tuning parameter values
#use all the variables at first
tune.out1 = tune(svm, diagnosis ~ .,
                data = cancer.train, kernel = "linear",
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#allow the cross-validation algorithm to select the best model
best.svm1 = tune.out1$best.model
#the best cost tuning parameter is 1

#add each variable in the list of ranked variables
#iteratively to the model and compute the accuracy
#performs a variant of forward selection
varnum.svm = 1:length(svm.vars)
accuracy.svm = rep(NA, length(svm.vars))
for (ii in 1:length(svm.vars)){
  #generate a formula including ii variables
  Formula.svm <- formula(paste("diagnosis ~ ", 
                           paste(svm.vars[1:ii], collapse=" + ")))
  
  #create the model
  partial.mod.svm = svm(Formula.svm, data = cancer.train, kernel = "linear",
                        cost = 1)
  
  #predict malignancy
  cancer.pred.svm = predict(partial.mod.svm, cancer.valid, 
                             decision.values = TRUE)
  
  #store the accuracy of the model
  accuracy.svm[ii] = mean(cancer.pred.svm == cancer.valid$diagnosis)
}

#Plot the accuracy versus number of variables
plot(x = varnum.svm, y = accuracy.svm, ylab = "Accuracy", 
     xlab = "Number of Variables")
which.max(accuracy.svm)

#formula based on the most important 10 variables
Formula.svm.final <- formula(paste("diagnosis ~ ", 
                         paste(svm.vars[1:6], collapse=" + ")))

#create the model
svm.mod.final = svm(Formula.svm.final, data = cancer.train,
                    kernel = "linear",
                    cost = 1)

#predict malignancy
svm.pred.final = predict(svm.mod.final, cancer.valid,
                         decision.values = TRUE)

#find the accuracy of the SVM model
mean(svm.pred.final == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction =
        svm.pred.final)



###########################
###########################
###                     ###
###    kNN Modeling     ###
###                     ###
###########################
###########################

train.X = as.matrix(cancer.train[,2:ncol(cancer.train)])
valid.X = as.matrix(cancer.valid[,2:ncol(cancer.valid)])
test.X = as.matrix(cancer.test[,2:ncol(cancer.test)])
train.Y = cancer.train$diagnosis
valid.Y = cancer.valid$diagnosis
test.Y = cancer.test$diagnosis

set.seed(12)
k.optimal = 1:20
k.accuracy = rep(NA, 20)
for (ii in 1:20){
    knn.pred.init = knn(train.X, valid.X, train.Y, k=ii)
    k.accuracy[ii] = mean(knn.pred.init == valid.Y)
}
plot(k.optimal, k.accuracy, xlab = "k", ylab = "Accuracy")
which.max(k.accuracy)

knn.final.pred = knn(train.X, valid.X, train.Y, 
                     k = which.max(k.accuracy))
mean(knn.final.pred == valid.Y)
table(truth = valid.Y, prediction = knn.final.pred)