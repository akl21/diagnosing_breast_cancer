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

#put variables in order of importance
#while testing out logistic regression
glm.mod = train(diagnosis ~ ., data = cancer.train, 
                method = "glm", family = "binomial")
varImp(glm.mod)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(desc(Overall))

#put variables in order of importance
#while testing out SVM with linear kernels
svm.mod = train(diagnosis ~ ., data = cancer.train,
                method = "svmLinear")
varImp(svm.mod)$importance %>% 
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(desc(B), desc(M))

#implement SVMs with linear kernels with different
#tuning parameter values
#use the variables designated as the best through 
#best subset selection with ten variables
tune.out1 = tune(svm, diagnosis ~ radius_mean + texture_mean +
                  compactness_mean + concavity_mean + concavity_se + 
                  concave_points_se + radius_worst + area_worst +
                  symmetry_worst + fractal_dimension_worst,
                data = cancer.train, kernel = "linear",
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#allow the cross-validation algorithm to select the best model
best.svm1 = tune.out1$best.model

#predict the cancer classifications for the validation set
diag.pred1 = predict(best.svm1, cancer.valid)

#examine the results in a confusion matrix
table(predict = diag.pred1, truth = cancer.valid$diagnosis)

#find the accuracy rate
mean(cancer.valid$diagnosis == diag.pred1)

#look at the variables included in the best 14-variable model,
#selected with best subset selection
coef(best.mod, 14)

#implement SVMs with linear kernels with different
#tuning parameter values
#use the variables designated as the best through 
#best subset selection with 14 variables
tune.out2 = tune(svm, diagnosis ~ radius_mean + texture_mean +
                   area_mean + compactness_mean + concavity_mean +
                   area_se + smoothness_se + concavity_se +
                   concave_points_se + radius_worst +area_worst +
                   concavity_worst + symmetry_worst + fractal_dimension_worst,
                 data = cancer.train, kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#allow the cross-validation algorithm to select the best model
best.svm2 = tune.out2$best.model

#predict the cancer classifications for the validation set
diag.pred2 = predict(best.svm2, cancer.valid)

#examine the results in a confusion matrix
table(predict = diag.pred2, truth = cancer.valid$diagnosis)

#find the accuracy rate
mean(cancer.valid$diagnosis == diag.pred2)

#look at the variables included in the best 17-variable model,
#selected with best subset selection
coef(best.mod, 17)

#implement SVMs with linear kernels with different
#tuning parameter values
#use the variables designated as the best through 
#best subset selection with 14 variables
tune.out2 = tune(svm, diagnosis ~ radius_mean + texture_mean +
                   area_mean + compactness_mean + concavity_mean +
                   area_se + smoothness_se + concavity_se +
                   concave_points_se + radius_worst +area_worst +
                   concavity_worst + symmetry_worst + fractal_dimension_worst,
                 data = cancer.train, kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#allow the cross-validation algorithm to select the best model
best.svm2 = tune.out2$best.model

#predict the cancer classifications for the validation set
diag.pred2 = predict(best.svm2, cancer.valid)

#examine the results in a confusion matrix
table(predict = diag.pred2, truth = cancer.valid$diagnosis)

#find the accuracy rate
mean(cancer.valid$diagnosis == diag.pred2)
