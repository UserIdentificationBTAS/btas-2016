# R script to implement user identification on the User Identification
# From Walking Activity Data Set from UCI Machine Learning Repository.
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


set.seed(100)

walking_data <- read.csv('total.csv')
names(walking_data) <- c('subject', 'mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'mad_x', 'mad_y', 'mad_z')
walking_data$subject <- as.factor(walking_data$subject)

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

# Load .RData file into environment.
loadRData <- function(fileName){
    load(fileName)
    get(ls()[ls() != "fileName"])
}

train <- NULL
test <- NULL

# Execute the user identification.
run <- function() {
    train <<- loadRData("./CleanData2/train.RData")
    test <<- loadRData("./CleanData2/test.RData")
    
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
    start <- Sys.time()
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
    save(rf_ret,    file="./CleanData2/ret/rf_ret.RData")
    save(svm_ret,   file="./CleanData2/ret/svm_ret.RData")
    save(nb_ret,    file="./CleanData2/ret/nb_ret.RData")
    save(j_ret,     file="./CleanData2/ret/j_ret.RData")
    save(nn_ret,    file="./CleanData2/ret/nn_ret.RData")
    save(knn_ret,   file="./CleanData2/ret/knn_ret.RData")
    save(cart_ret,  file="./CleanData2/ret/cart_ret.RData")
    save(jr_ret,    file="./CleanData2/ret/jr_ret.RData")
    save(bag_ret,   file="./CleanData2/ret/bag_ret.RData")
    save(boost_ret, file="./CleanData2/ret/boost_ret.RData")
}

# Load classification results from local files. It is helpful for
# analysis of the results.
load_all <- function() {
    load("./CleanData2/ret/rf_ret.RData")
    load("./CleanData2/ret/svm_ret.RData")
    load("./CleanData2/ret/nb_ret.RData")
    load("./CleanData2/ret/j_ret.RData")
    load("./CleanData2/ret/nn_ret.RData")
    load("./CleanData2/ret/knn_ret.RData")
    load("./CleanData2/ret/cart_ret.RData")
    load("./CleanData2/ret/jr_ret.RData")
    load("./CleanData2/ret/bag_ret.RData")
    load("./CleanData2/ret/boost_ret.RData")
}
