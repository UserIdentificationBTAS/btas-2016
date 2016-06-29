# Preprocess the Smartphone-Based Recognition of Human Activities and
# Postural Transitios data set from UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
# 
# Author: Chunxu Tang
# Email: chunxutang@gmail.com
# License: MIT


# The function merges the original training dataset and testing dataset together, 
# and partition the data into two parts:
#     1. data with activity 1-6
#     2. data with activity 7-16 (postural transition data)
preprocess <- function() {
    activity_labels <- read.table('./activity_labels.txt')
    activity_labels <- activity_labels$V2
    features <- read.table('./features.txt')
    features <- features$V1
    
    x_test <- read.table('./Test/X_test.txt')
    y_test <- read.table('./Test/y_test.txt')
    subject_test <- read.table('./Test/subject_id_test.txt')
    names(x_test) <- features
    names(y_test) <- "label"
    names(subject_test) <- "subject"
    
    x_train <- read.table('./Train/X_train.txt')
    y_train <- read.table('./Train/y_train.txt')
    subject_train <- read.table('./Train/subject_id_train.txt')
    names(x_train) <- features
    names(y_train) <- "label"
    names(subject_train) <- "subject"
    
    main_data <- rbind(x_test, x_train)
    data <- rbind(data.frame(y_test, subject_test), 
                  data.frame(y_train, subject_train))
    data <- cbind(data, main_data)
    
    activity_data <- subset(data, label < 7)
    postural_data <- subset(data, label > 6)
    
    names(activity_data) <- gsub('-', '', names(activity_data))
    names(postural_data) <- gsub('-', '', names(postural_data))
    
    save(activity_data, file='activity_data.RData')
    save(postural_data, file='postural_data.RData')
}