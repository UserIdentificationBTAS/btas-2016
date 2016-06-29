# R script to implement user identification on the Smartphone-Based 
# Recognition of Human Activities and Postural Transitios data set from 
# UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
# 
# Ten classification algorithms are tested:
#   1. Random Forest
#   2. Support Vector Machine
#   3. Naive Bayes
#   4. J48
#   5. Neural Network
#   6. K Nearest Neighbors
#   7. Rpart
#   8. JRip
#   9. Bagging
#   10. AdaBoost
#
# Author: Chunxu Tang
# Email: chunxutang@gmail.com
# License: MIT


library(caret)
library(randomForest)
library(e1071)  # naive bayes
library(MASS)   # svm
library(nnet)   # neural network
library(RWeka)  # J48, JRip
library(class)  # knn
library(rpart)  # CART
library(adabag) # Bagging, Boosting
library(plyr)


load('activity_data.RData')
set.seed(100)

# Extract necessary features

label_index <- c(1:8, 43:48, 83:85, 123:128, 163:165)
data <- activity_data[, label_index]
data$subject <- as.factor(data$subject)

# Obtain data for every activity

walking <- subset(data, label == 1)
walking$label <- NULL

upstairs <- subset(data, label == 2)
upstairs$label <- NULL

downstairs <- subset(data, label == 3)
downstairs$label <- NULL

sitting <- subset(data, label == 4)
sitting$label <- NULL

standing <- subset(data, label == 5)
standing$label <- NULL

laying <- subset(data, label == 6)
laying$label <- NULL


partition_dataset(walking, "walking")
partition_dataset(upstairs, "upstairs")
partition_dataset(downstairs, "downstairs")
partition_dataset(sitting, "sitting")
partition_dataset(standing, "standing")
partition_dataset(laying, "laying")


# The function partitions the data set into 70% training set,
# and 30% testing. Then it saves the data into local files.
partition_dataset <- function(data, name) {
    inTrain <- createDataPartition(y=data$subject, p=0.7, list=FALSE)
    train <- data[inTrain, ]
    test <- data[-inTrain, ]
    train_file <- paste('./CleanData/', name, '_train.RData', sep="")
    test_file <- paste('./CleanData/', name, '_test.RData', sep="")
    save(train, file=train_file)
    save(test, file=test_file)
}

# The function partition the aggregated data set (all activities mixed
# together) into 70% training set and 30% testing set. And then, it saves
# data sets into local files.
construct_all <- function() {
    inTrain <- createDataPartition(y=data$subject, p=0.7, list=FALSE)
    train <- data[inTrain, ]
    test <- data[-inTrain, ]
    save(train, file="./CleanData/aggregate_train.RData")
    save(test, file="./CleanData/aggregate_test.RData")
}

# The lists to store classification performance of the classifiers.
rf_ret <- NULL    # random forest
svm_ret <- NULL   # support vector machine
nb_ret <- NULL    # naive bayes
j_ret <- NULL     # C4.5 (J48)
nn_ret <- NULL    # neural network
knn_ret <- NULL   # k nearest neighbor
cart_ret <- NULL  # CART
jr_ret <- NULL    # JRip
bag_ret <- NULL   # Bagging
boost_ret <- NULL # Adaboost

ret_index <- c(3, 8, 13, 18, 23, 28, 33, 38, 43)
ret_index <- c(3,4, 8,9, 13,14, 18,19, 23,24, 28,29, 33,34, 38,39, 43,44) # 18

# Load .RData file into environment.
loadRData <- function(fileName){
    load(fileName)
    get(ls()[ls() != "fileName"])
}

train <- NULL
test <- NULL

# Execute the user identification.
run <- function() {
    analysis("walking")
    analysis("upstairs")
    analysis("downstairs")
    analysis("sitting")
    analysis("standing")
    analysis("laying")
    
    analysis_aggregate()
    additional_analysis()
}

# Execute classifications on a target data set.
analysis <- function(name) {
    train_file <- paste('./CleanData/', name, '_train.RData', sep="")
    test_file <- paste('./CleanData/', name, '_test.RData', sep="")
    
    train <<- loadRData(train_file)
    test <<- loadRData(test_file)
    
    random_forest()
    support_vector_machine()
    naive_bayes()
    j48()
    neural_network()
    k_nearest_neighbor()
    r_part()
    j_rip()
    bagg_ing()
    ada_boost()
}

# Execute classifications on the aggregated data set.
# There are two tests:
#   1. Classifications on the aggregated data set with activity labels.
#   2. Classifications on the aggregated data set without labels.
analysis_aggregate <- function() {
    train <<- loadRData("./CleanData/aggregate_train.RData")
    test <<- loadRData("./CleanData/aggregate_test.RData")
    
    random_forest()
    support_vector_machine()
    naive_bayes()
    j48()
    neural_network()
    k_nearest_neighbor()
    r_part()
    j_rip()
    bagg_ing()
    ada_boost()
    
    train <<- loadRData("./CleanData/aggregate_train.RData")
    test <<- loadRData("./CleanData/aggregate_test.RData")
    
    # Labels of the activities are removed.
    train$label <<- NULL
    test$label <<- NULL
    
    random_forest()
    support_vector_machine()
    naive_bayes()
    j48()
    neural_network()
    k_nearest_neighbor()
    r_part()
    j_rip()
    bagg_ing()
    ada_boost()
}

# Partition the data set mixed with activity data and postural transitions
# into 70% training set and 30% testing data set. And then, it saves the
# data to local files.
additional_partition <- function() {
    load("postural_data.RData")
    load("activity_data.RData")
    data <- activity_data[, label_index]
    data <- rbind(data, postural_data[, label_index])
    
    data$label <- NULL  # Activity labels are removed.
    data$subject <- as.factor(data$subject)
    
    inTrain <- createDataPartition(y=data$subject, p=0.7, list=FALSE)
    train <- data[inTrain, ]
    test <- data[-inTrain, ]
    save(train, file="./CleanData/all_train.RData")
    save(test, file="./CleanData/all_test.RData")
}

# Execute analysis on the total data set.
additional_analysis <- function() {
    train <<- loadRData("./CleanData/all_train.RData")
    test <<- loadRData("./CleanData/all_test.RData")
    
    random_forest()
    support_vector_machine()
    naive_bayes()
    j48()
    neural_network()
    k_nearest_neighbor()
    r_part()
    j_rip()
    bagg_ing()
    ada_boost()
}

random_forest <- function() {
    model <- randomForest(subject ~ ., data=train)
    cross_val <- predict(model, test)
    rf_ret <<- c(rf_ret, confusionMatrix(cross_val, test$subject))
}

support_vector_machine <- function() {
    model <- svm(subject ~ ., data=train)
    cross_val <- predict(model, test)
    svm_ret <<- c(svm_ret, confusionMatrix(cross_val, test$subject))
}

naive_bayes <- function() {
    model <- naiveBayes(subject ~ ., data=train)
    cross_val <- predict(model, test)
    nb_ret <<- c(nb_ret, confusionMatrix(cross_val, test$subject))
}

j48 <- function() {
    model <- J48(subject ~ ., data=train)
    cross_val <- predict(model, test)
    j_ret <<- c(j_ret, confusionMatrix(cross_val, test$subject))
}

neural_network <- function() {
    model <- nnet(subject ~ ., data=train, size=9, rang=0.1, decay=5e-4, maxit=1000, trace=FALSE)
    cross_val <- predict(model, test, type="class")
    cross_val <- factor(cross_val)
    nn_ret <<- c(nn_ret, confusionMatrix(cross_val, test$subject))
}

k_nearest_neighbor <- function() {
    cross_val <- knn(train, test, train$subject, k = 5)
    knn_ret <<- c(knn_ret, confusionMatrix(cross_val, test$subject))
}

r_part <- function() {
    model <- rpart(subject ~ ., data=train)
    cross_val <- predict(model, test, type="class")
    cart_ret <<- c(cart_ret, confusionMatrix(cross_val, test$subject))
}

j_rip <- function() {
    model <- JRip(subject ~ ., data=train)
    cross_val <- predict(model, test, type="class")
    jr_ret <<- c(jr_ret, confusionMatrix(cross_val, test$subject))
}

bagg_ing <- function() {
    model <- bagging(subject ~ ., data=train)
    cross_val <- predict(model, test, type="class")
    cross_val <- cross_val$class
    bag_ret <<- c(bag_ret, confusionMatrix(cross_val, test$subject))
}

ada_boost <- function() {
    model <- boosting(subject~ ., data=train)
    cross_val <- predict(model, test, type="class")
    cross_val <- cross_val$class
    boost_ret <<- c(boost_ret, confusionMatrix(cross_val, test$subject))
}

# Save classification results into local files.
save_all <- function() {
    save(rf_ret,    file="./CleanData/ret/rf_ret.RData")
    save(svm_ret,   file="./CleanData/ret/svm_ret.RData")
    save(nb_ret,    file="./CleanData/ret/nb_ret.RData")
    save(j_ret,     file="./CleanData/ret/j_ret.RData")
    save(nn_ret,    file="./CleanData/ret/nn_ret.RData")
    save(knn_ret,   file="./CleanData/ret/knn_ret.RData")
    save(cart_ret,  file="./CleanData/ret/cart_ret.RData")
    save(jr_ret,    file="./CleanData/ret/jr_ret.RData")
    save(bag_ret,   file="./CleanData/ret/bag_ret.RData")
    save(boost_ret, file="./CleanData/ret/boost_ret.RData")
}

# Load classification results from local files. It is helpful for
# analysis of the results.
load_all <- function() {
    load("./CleanData/ret/rf_ret.RData")
    load("./CleanData/ret/svm_ret.RData")
    load("./CleanData/ret/nb_ret.RData")
    load("./CleanData/ret/j_ret.RData")
    load("./CleanData/ret/nn_ret.RData")
    load("./CleanData/ret/knn_ret.RData")
    load("./CleanData/ret/cart_ret.RData")
    load("./CleanData/ret/jr_ret.RData")
    load("./CleanData/ret/bag_ret.RData")
    load("./CleanData/ret/boost_ret.RData")
}
