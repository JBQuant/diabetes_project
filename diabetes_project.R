rm(list=ls())
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}
needed <- c('glmnet', 'randomForest', 'pROC', "verification", "ada", "leaps", 'dplyr')  
installIfAbsentAndLoad(needed)

results <- read.csv("FinalData.OneHourRandom.csv", header = T)
summary(results)
str(results)
names(results)

results <- results[-which(results$Avg_Glucose_Value < 40), ]  # Drop rows where Avg_Glucose_Value < 40 (bad data).
results$TimeOfDay <- as.factor(results$TimeOfDay)
results$LowOccurance <- as.factor(results$LowOccurance)
results$HighOccurance <- as.factor(results$HighOccurance)
results$LowOccurance_Lag1 <- as.factor(results$LowOccurance_Lag1)
results$HighOccurance_Lag1 <- as.factor(results$HighOccurance_Lag1)
par(mfrow = c(5,4), mai = c(0.25, 0.3, 0.25, 0.1))
k <- 1
for(i in c(3:9, c(12:17), c(28:32), c(37,38))) { 
  if (k == 13) {
    plot(results[,i], results$Avg_Glucose_Value, xlab= "", ylab = "", main=names(results)[i])
    title(ylab = "Avg Glucose Val", cex.lab=1.5)
    k <- k+1
  } else {
    plot(results[,i], results$Avg_Glucose_Value, xlab= "", ylab = "", main=names(results)[i]) 
    k <- k+1
  }
}
par(mfrow=c(1,1))
#############################################################
################# Mulit Lin Reg  ##########################
#############################################################
lm.data <-  select(.data = results, c("TimeOfDay","Sum_Steps","Avg_BPM","Min_BPM",
                                      "Max_BPM","StdDev_BPM","Sum_Calories_Burnt",
                                      "Sum_Distance_Traveled","Std_Dev_Lag1","Avg_Glucose_Value_Lag_1",
                                      "Calories_Burnt_Lag1", "Avg_BPM_Lag1","LowOccurance_Lag1",
                                      "HighOccurance_Lag1","BG_rate_of_change_Lag1","Max_BPM_Lag1", 
                                      "Avg_Glucose_Value"))
set.seed(1)
nobs <- nrow(lm.data) # Determine the number of rows in the dataset.
train <- sample(nobs, 0.7 * nobs)  
lm.model <- glm(formula = Avg_Glucose_Value ~ ., data = lm.data[train, ])
###Evaluate by scoring the test set
test <- setdiff(1:nobs, train)  # Collect test data
pr.actuals <- lm.data$Avg_Glucose_Value[test]
preds <- predict(lm.model, newdata = na.omit(lm.data[test, ]))
plot(preds, pr.actuals)
abline(lm(pr.actuals ~ preds), col = "red", lwd = 3)
coef(lm(pr.actuals ~ preds))
(lm.std <- sqrt(mean((preds - pr.actuals) ^ 2))) # Calculate RMSE
plot(lm.model)
#############################################################
################# Lasso Reg  ##########################
#############################################################
lasso.data <-  select(.data = results, c("TimeOfDay","Sum_Steps","Avg_BPM","Min_BPM",
                                      "Max_BPM","StdDev_BPM","Sum_Calories_Burnt",
                                      "Sum_Distance_Traveled","Std_Dev_Lag1","Avg_Glucose_Value_Lag_1",
                                      "Calories_Burnt_Lag1", "Avg_BPM_Lag1","LowOccurance_Lag1",
                                      "HighOccurance_Lag1","BG_rate_of_change_Lag1","Max_BPM_Lag1", 
                                      "Avg_Glucose_Value"))

set.seed(1)
nobs <- nrow(lasso.data) # Determine the number of rows in the dataset.
train <- sample(nobs, 0.7 * nobs)  
x <- model.matrix(Avg_Glucose_Value ~ ., lasso.data)[, -1]  # Collect the predictors and omit the intercept.
y <- lasso.data$Avg_Glucose_Value  # Create a vector of the dependent variable values.
grid <- 10 ^ seq(10, -2, length = 100)  # Set up the grid of values for the lambdas.
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)  # Use the glm function to generate the model.
plot(lasso.mod, xvar = 'lambda', label = T) 
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)  # We now perform cross-validation and compute the associated test error.
plot(cv.out)
(bestlam <- cv.out$lambda.min)
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[-train, ])  
plot(lasso.pred, lasso.data$Avg_Glucose_Value[-train], main = "Predicted Values vs. Actual Values")
abline(lm(lasso.data$Avg_Glucose_Value[-train]~lasso.pred),col="red", lwd = 2)
(lasso.data.std <- sqrt(mean((lasso.pred - y[-train]) ^ 2))) # Calculate RMSE
head(data.frame("Predicted" = lasso.pred, "Actual" = y[-train]), 10)
out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s = bestlam)
names(lasso.coef[which(lasso.coef == 0),])
lasso.coef # Display Lasso Coefficients
#############################################################
################# Random Forest  ##########################
#############################################################
rf.data <-  select(.data = results, c("TimeOfDay","Sum_Steps","Avg_BPM","Min_BPM",
                          "Max_BPM","StdDev_BPM","Sum_Calories_Burnt",
                          "Sum_Distance_Traveled","Std_Dev_Lag1","Avg_Glucose_Value_Lag_1",
                          "Calories_Burnt_Lag1", "Avg_BPM_Lag1","LowOccurance_Lag1",
                          "HighOccurance_Lag1","BG_rate_of_change_Lag1","Max_BPM_Lag1", "Avg_Glucose_Value"))
set.seed(1)
nobs <- nrow(rf.data) # Determine the number of rows in the dataset.
train <- sample(nobs, 0.7 * nobs)  
rf <- randomForest(formula = Avg_Glucose_Value ~ ., data = rf.data[train, ], ntree = 500, mtry = 5, 
                   importance = TRUE, localImp = TRUE, na.action = na.roughfix, replace = FALSE)
head(rf$predicted, 25)
importance(rf)[order(importance(rf)[, "%IncMSE"], decreasing = T), ]  #Display Variable Importance   
###Examine Error Rates for the Trees
head(rf$mse)
plot(rf, main = "Error Rates for Random Forest")  # Plot the error rate against the number of trees.
legend("topright", c("OOB", "No", "Yes"), text.col = 1:6, lty = 1:3, col = 1:3)
min.err <- min(rf$mse)  # Determine the minimum error rate for Out Of Bag.
min.err.idx <- which(rf$mse == min.err)  # Determine the index corresponding to minimum error rate for Out Of Bag.
min.err.idx
rf$mse[min.err.idx[1]]  # Return the error rates for each of OOB, 0 and 1 corresponding to minimum error rate for Out Of Bag.
###Rebuild the forest with the number of trees that minimizes the OOB error rate - use the first one if there are more than one
rf <- randomForest(formula = Avg_Glucose_Value ~ ., data = rf.data[train,], ntree = min.err.idx[1], mtry = 5, 
                   importance = TRUE, localImp = TRUE, na.action = na.roughfix, replace = FALSE)
plot(rf, main = "Error Rates for Random Forest")  # Plot the error rate against the number of trees.
legend("topright", c("OOB", "No", "Yes"), text.col = 1:6, lty = 1:3, col = 1:3)
###Evaluate by scoring the test set
test <- setdiff(1:nobs, train)  # Collect test data
pr.actuals <- rf.data$Avg_Glucose_Value[test]
preds <- predict(rf, newdata = na.omit(rf.data[test, ]))
plot(preds, pr.actuals)
abline(lm(pr.actuals ~ preds), col = "red", lwd = 3)
coef(lm(pr.actuals ~ preds))
(rf.std <- sqrt(mean((preds - pr.actuals) ^ 2))) # Calculate RMSE

#############################################################
################# Random Forest Classification ##############
#############################################################
rf.data2 <-  select(.data = results, c("TimeOfDay","Sum_Steps","Avg_BPM","Min_BPM",
                                       "Max_BPM","StdDev_BPM","Sum_Calories_Burnt",
                                       "Sum_Distance_Traveled","Std_Dev_Lag1","Avg_Glucose_Value_Lag_1",
                                       "Calories_Burnt_Lag1", "Avg_BPM_Lag1","LowOccurance_Lag1",
                                       "HighOccurance_Lag1","BG_rate_of_change_Lag1","Max_BPM_Lag1", 
                                       "LowOccurance"))
set.seed(1)
nobs <- nrow(rf.data2) # Determine the number of rows in the dataset.
train <- sample(nobs, 0.7 * nobs)  
rf2 <- randomForest(formula = LowOccurance ~ ., data = rf.data2[train, ], ntree = 500, mtry = 4, 
                   importance = TRUE, localImp = TRUE, na.action = na.roughfix, replace = FALSE)
head(rf2$predicted, 25)
importance(rf2)[order(importance(rf2)[, "MeanDecreaseAccuracy"], decreasing = T), ]  #Display Variable Importance   
###Examine Error Rates for the Trees
head(rf2$err.rate)
plot(rf2, main = "Error Rates for Random Forest")  # Plot the error rate against the number of trees.
legend("topright", c("OOB", "No", "Yes"), text.col = 1:6, lty = 1:3, col = 1:3)
min.err <- min(rf2$err.rate[,"OOB"])  # Determine the minimum error rate for Out Of Bag.
min.err.idx <- which(rf2$err.rate[, "OOB"] == min.err)  # Determine the index corresponding to minimum error rate for Out Of Bag.
min.err.idx
rf2$err.rate[min.err.idx[1],]  # Return the error rates for each of OOB, 0 and 1 corresponding to minimum error rate for Out Of Bag.
###Rebuild the forest with the number of trees that minimizes the OOB error rate - use the first one if there are more than one
rf2 <- randomForest(formula = LowOccurance ~ ., data = rf.data2[train,], ntree = min.err.idx[1], mtry = 4, 
                   importance = TRUE, localImp = TRUE, na.action = na.roughfix, replace = FALSE)
plot(rf2, main = "Random Forest Error Rates")
legend("topright", c("OOB", "No", "Yes"), text.col = 1:6, lty = 1:3, col = 1:3)
head(rf2$votes)  # Review voting info for each observation
###Plot the OOB ROC curve and calculate AUC. 
aucc <- roc.area(as.integer(as.factor(rf.data2[train, "LowOccurance"])) - 1, rf2$votes[, 2])
aucc$A
aucc$p.value  # Review p-value and compare with null hypothesis: aucc = 0.5 
roc.plot(as.integer(as.factor(rf.data2[train, "LowOccurance"])) - 1, rf2$votes[, 2], main = "ROC Plot for RandomForest Classifier")
###Evaluate by scoring the test set
test <- setdiff(1:nobs, train)  # Collect test data
prtest <- predict(rf2, newdata = na.omit(rf.data2[test, ]))
table(rf.data2[test, "LowOccurance"], prtest, dnn = c("Actual", "Predicted"))
round(100 * table(rf.data2[test, "LowOccurance"], prtest, dnn = c("% Actual", "% Predicted")) / length(prtest), 1)
table(rf.data2$LowOccurance)[2] / sum(table(rf.data2$LowOccurance))

#############################################################
##################### AdaBoost ##############################
#############################################################
# Predicting if there is at least one blood sugar in a given hour
# which is indicated by the binary variable LowOccurance which is number 31
ada.data <-  select(.data = results, c("TimeOfDay","Sum_Steps","Avg_BPM","Min_BPM",
                                      "Max_BPM","StdDev_BPM","Sum_Calories_Burnt",
                                      "Sum_Distance_Traveled","Std_Dev_Lag1","Avg_Glucose_Value_Lag_1",
                                      "Calories_Burnt_Lag1", "Avg_BPM_Lag1","LowOccurance_Lag1",
                                      "HighOccurance_Lag1","BG_rate_of_change_Lag1","Max_BPM_Lag1", 
                                      "LowOccurance"))
nobs <- nrow(ada.data)
train <- sample(nobs, 0.7*nobs)
bm<- ada(formula=LowOccurance ~ .,data=ada.data[train,],
         iter=50,
         bag.frac=0.5,
         control=rpart.control(maxdepth=30,cp=0.01,
                               minsplit=20,xval=10))
print(bm) 
# Evaluate by scoring the training set
prtrain <- predict(bm, newdata=ada.data[train,])
prtrainProbs <- predict(bm, newdata=ada.data[train,], type = "prob")
table(ada.data[train,"LowOccurance"], prtrain,dnn=c("Actual", "Predicted"))
round(100* table(ada.data[train,"LowOccurance"], prtrain,dnn=c("% Actual", "% Predicted"))/length(prtrain),1)
#ROC Curve
aucc <- roc.area(as.integer(as.factor(ada.data[train, "LowOccurance"]))-1,prtrainProbs[,2])
aucc$A
aucc$p.value                #null hypothesis: aucc=0.5 
roc.plot(as.integer(as.factor(ada.data[train,"LowOccurance"]))-1,prtrainProbs[,2], main="ROC Plot for Adaptive Boosting Classifier")
# Evaluate by scoring the test set
prtest <- predict(bm, newdata=ada.data[-train,])
table(ada.data[-train,"LowOccurance"], prtest,dnn=c("Actual", "Predicted"))
round(100* table(ada.data[-train,"LowOccurance"], prtest,dnn=c("% Actual", "% Predicted"))/length(prtest),1)
#ada.data[1:100,]
probs <- predict(bm, newdata = ada.data[-train,], "prob")
prtest2 <- rep(0, nrow(ada.data[-train,]))
# for (i in c(0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.20, 0.15, 0.1, 0.05)) {
#   print(i)
#   prtest2[which(predict(bm, newdata = ada.data[-train,], "prob")[,2] > i)] <- 1
#   #print(table(ada.data[-train,"LowOccurance"], prtest2,dnn=c("Actual", "Predicted")))
#   table <- round(100* table(ada.data[-train,"LowOccurance"], prtest2,dnn=c("% Actual", "% Predicted"))/length(prtest),1)
#   print(table)
#   cat("error rate:", sum(table) - (table[1,1] + table[2,2]/ sum(table)))
# }
prtest2[which(predict(bm, newdata = ada.data[-train,], "prob")[,2] > 0.3)] <- 1
round(100* table(ada.data[-train,"LowOccurance"], prtest2,dnn=c("% Actual", "% Predicted"))/length(prtest),1)
