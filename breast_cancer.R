#set working directory externally
#read in data
cancer = read.csv("data.csv", header = TRUE, row.names = "id")

#examine data
#View(cancer)

#load libraries
library(caret)
library(dplyr)
library(leaps)
library(caret)
library(e1071)
library(tibble)
library(ggplot2)
library(Deducer)
library(class)
library(rknn)
library(ROCR)
library(randomForest)

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

#Exploratory Data Analysis
ggplot(cancer) + 
  geom_boxplot(aes(x = as.factor(diagnosis), y = perimeter_worst), 
               fill = "blue") +
  labs(x = "Diagnosis", y = "Worst Perimeter") +
  ggtitle("Worst Perimeter Boxplots")
ggplot(cancer) + 
  geom_boxplot(aes(x = as.factor(diagnosis), y = area_worst), 
               fill = "purple") +
  labs(x = "Diagnosis", y = "Worst Area") +
  ggtitle("Worst Area Boxplots")
ggplot(cancer) + 
  geom_boxplot(aes(x = as.factor(diagnosis), y = radius_worst), 
               fill = "green") +
  labs(x = "Diagnosis", y = "Worst Radius") +
  ggtitle("Worst Radius Boxplots")
ggplot(cancer) + 
  geom_boxplot(aes(x = as.factor(diagnosis), y = concave_points_mean), 
               fill = "pink") +
  labs(x = "Diagnosis", y = "Mean of Concave Points") +
  ggtitle("Mean of Concave Points Boxplots")

##########################
##########################
###                    ###
### Data Partitioning  ###
###                    ###
##########################
##########################

#randomly sample 300 observations into training set,
#100 observations into the validation set
#and 169 observations into the test set

#randomly sample 300 of the row IDs for training
train.rows = sample(rownames(cancer), 300)

#based on the rows that weren't used in the training set,
#randomly sample 100 row IDs for the validation set
valid.rows = sample(setdiff(rownames(cancer), train.rows), 100)

#the remaining 169 rows go into the test set
test.rows = setdiff(rownames(cancer), union(train.rows, valid.rows))

#create three data frames by collecting all of the columns 
#from the appropriate rows
cancer.train = cancer[train.rows,]
cancer.valid = cancer[valid.rows,]
cancer.test = cancer[test.rows, ]

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
a.glm = data.frame(varnum, accuracy)
ggplot(a.glm)+
  geom_point(aes(x = varnum, y = accuracy))+
  labs(x = "Number of Variables", y = "Accuracy on Validation Set") +
  ggtitle("Logistic Regression Model")
which.max(accuracy)

#formula based on the most important variables,
#the number of variables to include was determined above
Formula <- formula(paste("diagnosis ~ ", 
                         paste(vars[1:which.max(accuracy)], collapse=" + ")))

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
glm.accuracy = mean(glm.pred.final == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction = glm.pred.final)

#plot the ROC curve
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
a.svm = data.frame(varnum.svm, accuracy.svm)
ggplot(a.glm)+
  geom_point(aes(x = varnum.svm, y = accuracy.svm))+
  labs(x = "Number of Variables", y = "Accuracy on Validation Set") +
  ggtitle("SVM Model")
which.max(accuracy.svm)

#formula based on the most important variables,
#the number of variables to include determined above
Formula.svm.final <- formula(paste("diagnosis ~ ", 
                         paste(svm.vars[1:which.max(accuracy.svm)], 
                               collapse=" + ")))

#create the model
svm.mod.final = svm(Formula.svm.final, data = cancer.train,
                    kernel = "linear",
                    cost = 1)

#predict malignancy
svm.pred.final = predict(svm.mod.final, cancer.valid,
                         decision.values = TRUE)

#find the accuracy of the SVM model
mean(svm.pred.final == cancer.valid$diagnosis)
svm.accuracy = mean(svm.pred.final == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction =
        svm.pred.final)

#write a function to extract performance measures from
#fitted decision values of the SVM and the actual 
#values of the diagnosis
roc.svm = function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  perf
  
}

#get the decison values from the SVM prediction
fitted = attributes(svm.pred.final)$decision.values

#find the false positive and true positive rates from the
#(hella weird) performance object
fpr = roc.svm(fitted, cancer.valid$diagnosis)@x.values
tpr = roc.svm(fitted, cancer.valid$diagnosis)@y.values

#put fpr and tpr, which are lists, into a data frame
roc.dat = data.frame(false_pos_rate = as.vector(fpr[[1]]), 
                     true_pos_rate = as.vector(tpr[[1]]))

#create a prediction object from the fitted values and the 
#actual values of the diagnosis
predobj = prediction(fitted, cancer.valid$diagnosis)

#find the area under curve for this prediction object
auc.obj = performance(predobj, measure = "auc")

#access and round the AUC value
auc.value = round(auc.obj@y.values[[1]],3)

#create a diagonal line data frame for reference
diag = data.frame(x = seq(0,1,by= 0.001), y = seq(0,1, by= 0.001))

#plot the ROC curve in ggplot2
p = ggplot(roc.dat, aes(x = false_pos_rate, y = true_pos_rate)) +
      geom_point(color = "blue") + geom_line(color = "blue") +
      geom_line(data = diag, aes(x = x,y = y), color = "red")
p + geom_point(data = diag, aes(x = x, y = y), color = "red") +
  theme(axis.text = element_text(size = 10),
        title = element_text(size = 12)) + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "SVM ROC Curve") +
  annotate("text", x = 0.85, y = 0.1, 
           label = paste("AUC: ", auc.value))
  

###########################
###########################
###                     ###
###    kNN Modeling     ###
###                     ###
###########################
###########################

#create training, validation, and testing matrices to fit the 
#knn function requirements
train.X = as.matrix(cancer.train[,2:ncol(cancer.train)])
valid.X = as.matrix(cancer.valid[,2:ncol(cancer.valid)])
test.X = as.matrix(cancer.test[,2:ncol(cancer.test)])
train.Y = cancer.train$diagnosis
valid.Y = cancer.valid$diagnosis
test.Y = cancer.test$diagnosis

#find the optimal subset of predictors when k = 10, a good 
#starting point
knn.obj = rknnBeg(train.X, train.Y, k = 1, r = 500, 
                  mtry = trunc(sqrt(ncol(train.X))),
                  fixed.partition = TRUE,
                  pk = 0.5, stopat = 4,
                  cluster = NULL, seed = NULL)
bestset(knn.obj)

#use the subset of predictors selected in the training, validation
#and test variable matrices
train.X.subset = as.matrix(cancer.train[,bestset(knn.obj)])
valid.X.subset = as.matrix(cancer.valid[,bestset(knn.obj)])
test.X.subset = as.matrix(cancer.test[,bestset(knn.obj)])

#find the optimal k based on the validation set accuracy
k.optimal.subset = 1:20
k.accuracy.subset = rep(NA, 20)
for (ii in 1:20){
  knn.pred.subset = knn(train.X.subset, valid.X.subset, train.Y, k=ii)
  k.accuracy.subset[ii] = mean(knn.pred.subset == valid.Y)
}
k.choose = data.frame(k.optimal.subset, k.accuracy.subset)
ggplot(k.choose) +
  geom_point(aes(x = k.optimal.subset, y = k.accuracy.subset))+
  labs(x = "k", y = "Accuracy") +
  ggtitle("Optimal Value of k")
which.max(k.accuracy.subset)

#use the optimal k from the loop in the final knn model
knn.final.pred1 = knn(train.X.subset, valid.X.subset, train.Y, 
                     k = which.max(k.accuracy.subset))

#look at the accuracy
mean(knn.final.pred1 == valid.Y)
knn.accuracy = mean(knn.final.pred1 == valid.Y)

#examine the confusion matrix
table(truth = valid.Y, prediction = knn.final.pred1)

#try to generate probabilities that ROCR can use

#set prob = TRUE in knn model, this gives the probability of 
#that the classifier assigns the observation to the predicted 
#category the observation is in
knn.final.pred2 = knn(train.X.subset, valid.X.subset, train.Y, 
                      k = which.max(k.accuracy.subset),
                      prob = TRUE)

#get the prob attribute
prob1 = attr(knn.final.pred2, "prob")

#convert the prob attribute into a form that ROCR likes
prob = 2*ifelse(knn.final.pred1 == "B", 1-prob1, prob1) - 1

#make a prediction object
pred_knn = prediction(prob, valid.Y)

#make a performance object
perf_knn = performance(pred_knn, "tpr", "fpr")

#find the false positive and true positive rates from the
#performance object
fpr_knn = perf_knn@x.values
tpr_knn = perf_knn@y.values

#put fpr and tpr, which are lists, into a data frame
roc.dat.knn = data.frame(false_pos_rate = as.vector(fpr_knn[[1]]), 
                     true_pos_rate = as.vector(tpr_knn[[1]]))

#find the area under curve for this prediction object
auc.obj.knn = performance(pred_knn, measure = "auc")

#access and round the AUC value
auc.value.knn = round(auc.obj.knn@y.values[[1]],3)

#plot the ROC curve in ggplot2
#use diag from the SVM section for the reference line
p.knn = ggplot(roc.dat.knn, aes(x = false_pos_rate, y = true_pos_rate)) +
  geom_point(color = "purple") + geom_line(color = "purple") +
  geom_line(data = diag, aes(x = x,y = y), color = "red")
p.knn + geom_point(data = diag, aes(x = x, y = y), color = "red") +
  theme(axis.text = element_text(size = 10),
        title = element_text(size = 12)) + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "kNN ROC Curve") +
  annotate("text", x = 0.85, y = 0.1, 
           label = paste("AUC: ", auc.value.knn))

###########################
###########################
###                     ###
###   Random Forest     ###
###                     ###
###########################
###########################

#create a random forest model with all the variables 
#to extract variable importance
rf.cancer = randomForest(diagnosis ~ ., data = cancer, subset = train.rows,
                         mtry = floor(sqrt(ncol(cancer))), importance = TRUE)

#predict the classifications
yhat.rf = predict(rf.cancer, newdata = cancer.valid)

#examine the results for all variables
table(truth = cancer.valid$diagnosis, rf.prediction = yhat.rf)
mean(yhat.rf == cancer.valid$diagnosis)

#get the variable importances
rf.var.import = importance(rf.cancer) %>% 
  as.data.frame() %>%
  mutate(rowname = rownames(.)) %>%
  arrange(desc(MeanDecreaseAccuracy))

#store the variables in order of importance
rf.vars = rf.var.import$rowname

#add each variable in the list of ranked variables
#iteratively to the model and compute the accuracy
#performs a variant of forward selection
varnum.rf = 1:length(rf.vars)
accuracy.rf = rep(NA, length(rf.vars))
for (ii in 1:length(rf.vars)){
  #generate a formula including ii variables
  Formula.rf <- formula(paste("diagnosis ~ ", 
                               paste(rf.vars[1:ii], collapse=" + ")))
  
  #create the model
  partial.mod.rf = randomForest(Formula.rf, 
                                data = cancer, subset = train.rows,
                                mtry = floor(sqrt(ii)), 
                                importance = TRUE)
  
  #predict malignancy
  cancer.pred.rf = predict(partial.mod.rf, newdata = cancer.valid)
  
  #store the accuracy of the model
  accuracy.rf[ii] = mean(cancer.pred.rf == cancer.valid$diagnosis)
}

#Plot the accuracy versus number of variables
a.rf = data.frame(varnum.rf, accuracy.rf)
ggplot(a.rf)+
  geom_point(aes(x = varnum.rf, y = accuracy.rf))+
  labs(x = "Number of Variables", y = "Accuracy on Validation Set") +
  ggtitle("Random Forest Model")
which.max(accuracy.rf)

#formula based on the most important 9 variables
Formula.rf.final <- formula(paste("diagnosis ~ ", 
                                   paste(rf.vars[1:which.max(accuracy.rf)], 
                                         collapse=" + ")))

#create the model
final.mod.rf = randomForest(Formula.rf.final, 
                              data = cancer, subset = train.rows,
                              mtry = floor(sqrt(which.max(accuracy.rf))), 
                              importance = TRUE)

#predict malignancy
rf.pred.final = predict(final.mod.rf, cancer.valid)

#find the accuracy of the random forest model
mean(rf.pred.final == cancer.valid$diagnosis)
rf.accuracy = mean(rf.pred.final == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction =
        rf.pred.final)

rf.pred.roc = predict(final.mod.rf, newdata = cancer.valid,
                      type = "prob")[,2]

rf.pred.obj = prediction(as.numeric(rf.pred.roc), cancer.valid$diagnosis)
rf.perf.obj = performance(rf.pred.obj, "tpr", "fpr")

#find the false positive and true positive rates from the
#performance object
fpr_rf = rf.perf.obj@x.values
tpr_rf = rf.perf.obj@y.values

#put fpr and tpr, which are lists, into a data frame
roc.dat.rf = data.frame(false_pos_rate = as.vector(fpr_rf[[1]]), 
                         true_pos_rate = as.vector(tpr_rf[[1]]))

#find the area under curve for this prediction object
auc.obj.rf = performance(rf.pred.obj, measure = "auc")

#access and round the AUC value
auc.value.rf = round(auc.obj.rf@y.values[[1]],3)

#plot the ROC curve in ggplot2
#use diag from the SVM section for the reference line
p.rf = ggplot(roc.dat.rf, aes(x = false_pos_rate, y = true_pos_rate)) +
  geom_point(color = "forestgreen") + geom_line(color = "forestgreen") +
  geom_line(data = diag, aes(x = x,y = y), color = "red")
p.rf + geom_point(data = diag, aes(x = x, y = y), color = "red") +
  theme(axis.text = element_text(size = 10),
        title = element_text(size = 12)) + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "Random Forest ROC Curve") +
  annotate("text", x = 0.85, y = 0.1, 
           label = paste("AUC: ", auc.value.rf))

#############################
#############################
###                       ###
### SVMs w/Radial Kernels ###
###                       ###
#############################
#############################

#put variables in order of importance
#while testing out SVM with linear kernels
svm.mod.r = train(diagnosis ~ ., data = cancer.train,
                method = "svmRadial")

svm.var.import.r = varImp(svm.mod.r)$importance %>% 
  as.data.frame() %>%
  mutate(rowname = rownames(.)) %>%
  arrange(desc(B), desc(M))

svm.vars.r = svm.var.import.r$rowname

#implement SVMs with linear kernels with different
#tuning parameter values
#use all the variables at first
tune.out1.r = tune(svm, diagnosis ~ .,
                 data = cancer.train, kernel = "radial",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

#allow the cross-validation algorithm to select the best model
best.svm1.r = tune.out1$best.model
#the best cost tuning parameter is 1

#add each variable in the list of ranked variables
#iteratively to the model and compute the accuracy
#performs a variant of forward selection
varnum.svm.r = 1:length(svm.vars.r)
accuracy.svm.r = rep(NA, length(svm.vars.r))
for (ii in 1:length(svm.vars.r)){
  #generate a formula including ii variables
  Formula.svm.r <- formula(paste("diagnosis ~ ", 
                               paste(svm.vars.r[1:ii], collapse=" + ")))
  
  #create the model
  partial.mod.svm.r = svm(Formula.svm.r, data = cancer.train, kernel = "radial",
                        cost = 1)
  
  #predict malignancy
  cancer.pred.svm.r = predict(partial.mod.svm.r, cancer.valid, 
                            decision.values = TRUE)
  
  #store the accuracy of the model
  accuracy.svm.r[ii] = mean(cancer.pred.svm.r == cancer.valid$diagnosis)
}

#Plot the accuracy versus number of variables
a.svm.r = data.frame(varnum.svm.r, accuracy.svm.r)
ggplot(a.svm.r)+
  geom_point(aes(x = varnum.svm.r, y = accuracy.svm.r))+
  labs(x = "Number of Variables", y = "Accuracy") +
  ggtitle("SVM Model (Radial)")
which.max(accuracy.svm.r)

#formula based on the most important variables,
#the number of variables to include determined above
Formula.svm.final.r <- formula(paste("diagnosis ~ ", 
                                   paste(svm.vars.r[1:which.max(accuracy.svm.r)], 
                                         collapse=" + ")))

#create the model
svm.mod.final.r = svm(Formula.svm.final.r, data = cancer.train,
                    kernel = "radial",
                    cost = 1)

#predict malignancy
svm.pred.final.r = predict(svm.mod.final.r, cancer.valid,
                         decision.values = TRUE)

#find the accuracy of the SVM model
mean(svm.pred.final.r == cancer.valid$diagnosis)
svm.accuracy.r = mean(svm.pred.final.r == cancer.valid$diagnosis)

#examine the confusion matrix
table(truth = cancer.valid$diagnosis, prediction =
        svm.pred.final.r)

#write a function to extract performance measures from
#fitted decision values of the SVM and the actual 
#values of the diagnosis
roc.svm = function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  perf
  
}

#get the decison values from the SVM prediction
fitted.r = attributes(svm.pred.final.r)$decision.values

#find the false positive and true positive rates from the
#(hella weird) performance object
fpr.r = roc.svm(fitted.r, cancer.valid$diagnosis)@x.values
tpr.r = roc.svm(fitted.r, cancer.valid$diagnosis)@y.values

#put fpr and tpr, which are lists, into a data frame
roc.dat.r = data.frame(false_pos_rate = as.vector(fpr.r[[1]]), 
                     true_pos_rate = as.vector(tpr.r[[1]]))

#create a prediction object from the fitted values and the 
#actual values of the diagnosis
predobj.r = prediction(fitted.r, cancer.valid$diagnosis)

#find the area under curve for this prediction object
auc.obj.r = performance(predobj.r, measure = "auc")

#access and round the AUC value
auc.value.r = round(auc.obj.r@y.values[[1]],3)

#create a diagonal line data frame for reference
diag = data.frame(x = seq(0,1,by= 0.001), y = seq(0,1, by= 0.001))

#plot the ROC curve in ggplot2
p.r = ggplot(roc.dat.r, aes(x = false_pos_rate, y = true_pos_rate)) +
  geom_point(color = "green") + geom_line(color = "green") +
  geom_line(data = diag, aes(x = x,y = y), color = "red")
p.r + geom_point(data = diag, aes(x = x, y = y), color = "red") +
  theme(axis.text = element_text(size = 10),
        title = element_text(size = 12)) + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC curve") +
  annotate("text", x = 0.85, y = 0.1, 
           label = paste("AUC: ", auc.value.r))

#create a data frame of the models and their accuracies
model.accuracy = data.frame(model = c("logistic", "SVM", "kNN",
                                      "random forest"),
                            accuracy.of.model = 
                              c(glm.accuracy, svm.accuracy,
                                knn.accuracy, rf.accuracy))

#plot the model accuracies
ggplot(model.accuracy) + 
  geom_bar(aes(x = as.factor(model), y = accuracy.of.model), stat = "identity",
           fill = "blue") +
  labs(x = "Model", y = "Accuracy")+
  coord_cartesian(ylim=c(0.9, 1)) +
  ggtitle("Accuracy of Models on Validation Set")

#find the accuracy of the SVM with a linear kernel
#and an SVM with a radial kernel on the test set

#predict malignancy
svm.pred.test = predict(svm.mod.final, cancer.test,
                        decision.values = TRUE)

#determine the accuracy rate
mean(svm.pred.test == cancer.test$diagnosis)

#create the confusion matrix
confusionMatrix(as.factor(svm.pred.test), as.factor(cancer.test$diagnosis))
table(prediction = svm.pred.test, truth = cancer.test$diagnosis)

#predict malignancy (radial kernels)
svm.pred.test.r = predict(svm.mod.final.r, cancer.test,
                          decision.values = TRUE)

#determine the radial kernel accuracy rate
mean(svm.pred.test.r == cancer.test$diagnosis)

#create the ROC plot for the test data
#with SVM with linear kernel

#get the decison values from the SVM prediction
fitted.test = attributes(svm.pred.test)$decision.values

#find the false positive and true positive rates from the
#(hella weird) performance object
fpr.test = roc.svm(fitted.test, cancer.test$diagnosis)@x.values
tpr.test = roc.svm(fitted.test, cancer.test$diagnosis)@y.values

#put fpr and tpr, which are lists, into a data frame
roc.dat.test = data.frame(false_pos_rate = as.vector(fpr.test[[1]]), 
                     true_pos_rate = as.vector(tpr.test[[1]]))

#create a prediction object from the fitted values and the 
#actual values of the diagnosis
predobj.test = prediction(fitted.test, cancer.test$diagnosis)

#find the area under curve for this prediction object
auc.obj.test = performance(predobj.test, measure = "auc")

#access and round the AUC value
auc.value.test = round(auc.obj.test@y.values[[1]],3)

#create a diagonal line data frame for reference
diag = data.frame(x = seq(0,1,by= 0.001), y = seq(0,1, by= 0.001))

#plot the ROC curve in ggplot2
p = ggplot(roc.dat.test, aes(x = false_pos_rate, y = true_pos_rate)) +
  geom_point(color = "blue") + geom_line(color = "blue") +
  geom_line(data = diag, aes(x = x,y = y), color = "red")
p + geom_point(data = diag, aes(x = x, y = y), color = "red") +
  theme(axis.text = element_text(size = 10),
        title = element_text(size = 12)) + 
  labs(x = "1 - Specificity", y = "Sensitivity", 
       title = "Test Set ROC Curve for SVM") +
  annotate("text", x = 0.85, y = 0.1, 
           label = paste("AUC: ", auc.value.test))
