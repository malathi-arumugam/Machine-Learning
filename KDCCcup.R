
#library RWeka is used to read arff(attribute related file format) 
library(RWeka)
#pROC is used to draw the receiver operating characteristic
library(pROC)
# to include the random Forest model
library(randomForest)
# The data set consists of 60839 rows and 43 columns

trial<-read.arff("D:/literature1/literature/KDDCup99/KDDCup99_norm_idf.arff")
dim(trial)
str(trial)
# returns the data type of all variables including Id.
# note all the variables of type num.The last one is class variable of type factor.
# the last variable called outlier is dependent variable with two levels whether there is an attack has occured(yes) or not(no)
#Let us see how many outliers are of level "yes" and how many are of level "no".
table(trial$outlier)
# returns 60593 rows have no outliers present and 246 rows have the presence of outliers.
# we don't need id column because outlier does not depend on ID field.Let us remove it. And save the dataframe in another variable trial1.
trial1<-trial[,-1]
set.seed(1234)
# divide the data into training set which is 70% of data and the rest is validation set. 
random_splits<-runif(nrow(trial1))
train_df<-trial1[random_splits < .7,]
validate_df <- trial1[random_splits >=.7,]
# returns the number of rows in the training and validation set.
dim(validate_df)
dim(train_df)
outcome_name<-'outlier'
# select all the columns except outcome_name which is class variable
feature_names<-setdiff(names(trial1),outcome_name)

##----------------
set.seed(1234)
#train the data with random forest model . The number of trees choosen are 20. 
rf_model <- randomForest(x=train_df[,feature_names],y=train_df[,outcome_name],
                                                   importance=TRUE, ntree=20, mtry = 3)
# the model is tested with unseen data with no labels.
validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
# get the confusion matrix
rf_model$confusion
#  These commands return the parameters of the rf_model and all the important variables.
#str(rf_model)
#rf_model$importanceSD
#rf_model$importance
#validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
str(validate_predictions)
#plot roc which is a curve between Sensitivity and specificity
#Get the AUC score 
auc_rf = roc(response=as.numeric(validate_df[,outcome_name])-1,predictor=validate_predictions[,2])
plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

# let us try deep learning and use autoencoder model
# The library h2o is used for the application of deep learning models.
library(h2o)
 h2o.init()
trial1.hex<-as.h2o(train_df, destination_frame="train.hex")
# use a simple autoencoder model.
# The response variable is not included.

# the data set is iterated 50 times.You can change this parameter.
# There are 4 layers in this model. Two hidden layers of size 30 nodes is used. You try 
# with increasing or reducing this number.Default values are used for other parameters.

trial1.dl = h2o.deeplearning(x = feature_names, training_frame = trial1.hex,autoencoder = TRUE,
                                                                                          reproducible = T,
                                                                                         seed = 1234,
                                                                                          hidden = c(30,30), epochs = 50)
#returns the model summary
summary(trial1.dl)
#anomaly function returns the reconstruction Mean Squared Error
trial1.anon = h2o.anomaly(trial1.dl, trial1.hex, per_feature=FALSE)
head(trial1.anon)
dim(trial1.anon)

err <- as.data.frame(trial1.anon)
head(err)
class(err)
class(trial1.anon)
dim(err)
# gives the summary of MSE
summary(err)
# the error column is sorted and plot the sorted values

plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
# can fix a threshold and have a count of rows.
sum(err$Reconstruction.MSE>1)
sum(err$Reconstruction.MSE<.7)

train_df_auto <- train_df[err$Reconstruction.MSE < 1,]


set.seed(1234)
#outlier should be int
#fit the random forest model on the data which have rsconstuction error less than 1  
rf_model <- randomForest(x=train_df_auto[,feature_names],
                                                 y=as.factor(train_df_auto[,outcome_name]),
                                                   importance=TRUE, ntree=20, mtry = 3)
 validate_predictions_known <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
 auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,predictor=validate_predictions_known[,2])
 # get the plot and find AUC
 plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
 abline(h=1,col='blue')
 
# fit the random forest model on the data which have reconstruction error more than 1. 
 
 train_df_auto <- train_df[err$Reconstruction.MSE >= 1,]
 
 set.seed(1234)
 rf_model <- randomForest(x=train_df_auto[,feature_names],
                          y=train_df_auto[,outcome_name],
                          importance=TRUE, ntree=20, mtry = 3)
 
 validate_predictions_unknown <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
 # get AUC for this data
 auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,predictor=validate_predictions_unknown[,2])
 
 plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
 abline(h=1,col='blue')
 # combine the models
 
 
 valid_all <- (validate_predictions_known[,2] + validate_predictions_unknown[,2]) / 2
 # get the AUC for this combined models
 auc_rf = roc(response=as.numeric(validate_df[,outcome_name])-1,predictor=valid_all)
 plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
 abline(h=1,col='blue')
 abline(h=0,col='green')
 
 plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
  abline(h=1,col='blue')
  abline(h=0,col='green')
 
  
 
 
