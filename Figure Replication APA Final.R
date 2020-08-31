######################################### Figure Replication #########################################

# Student number 601631 & 534751
# Humboldt University APA 2020 Prof. Stefan Lessmann

# Load relevant libraries
if(!require("randomForest")) install.packages("randomForest"); library(randomForest)
if(!require("mlr")) install.packages("mlr"); library(mlr)
if(!require("pdp")) install.packages("pdp"); library(pdp)
if(!require("ranger")) install.packages("ranger"); library(ranger)
if(!require("DescTools")) install.packages("DescTools"); library(DescTools)
if(!require("ggplot2")) install.packages("ggplot2"); library(ggplot2)
if(!require("clusterGeneration")) install.packages("clusterGeneration"); library("clusterGeneration")
if(!require("mvtnorm")) install.packages("mvtnorm"); library("mvtnorm")
if(!require("xgboost")) install.packages("xgboost"); library(xgboost)
if(!require("ICEbox")) install.packages("ICEbox"); library(ICEbox)
if(!require("parallelMap")) install.packages("parallelMap"); library(parallelMap)

set.seed(123)





######################################### REQUIRED FUNCTIONS #########################################

#### Data Generation Function ####
# By Daniel Jakob, adapted for this project

#Data Set Generation Function
datagen <- function(N=5000,k=10,random_d=T,theta_num=1, var=1) {
  
  #Set seed for reproducible results
  set.seed(123)
  
  b = 1 / (1:k) #decreases influence of feature on CATE, found only in CATE [theta] formulae
  
  # = Generate covariance matrix of z = #
  sigma <- genPositiveDefMat(k, "unifcorrmat")$Sigma
  sigma <- cov2cor(sigma)
  
  z_fix <- rmvnorm(N, sigma = sigma) # = Generate z
  z_fix[,9] <- sample(c(0,1), replace=TRUE, size=N) #replace 9th var with dummy switch
  
  # Options for theta
  theta1 <- as.vector(sin(z_fix[,-10] %*% b[1:9]) + ifelse(z_fix[,10]>0, z_fix[,10]*1, 0)+ rnorm(N,0,0.5)) #v10 linear after 0
  theta2 <- as.vector(sin(z_fix[,-10] %*% b[1:9])+z_fix[,10]+ rnorm(N,0,0.5)) #v10 linear
  theta3 <- as.vector(sin(z_fix[,-10] %*% b[1:9])-z_fix[,10]+ rnorm(N,0,2)) #v10 negative linear with lots of noise
  theta4 <- as.vector(sin(z_fix[,1:8] %*% b[1:8])+z_fix[,10]*(-sin(z_fix[,8]))+ rnorm(N,0,0.5)) #v10 interacts with v8
  theta5 <- as.vector(sin(z_fix[,1:8] %*% b[1:8])+z_fix[,10]*z_fix[,9]+ rnorm(N,0,0.5)) #v10 interacts with v9
  theta6 <- as.vector(sin(z_fix[,-10] %*% b[1:9]) + ifelse(z_fix[,10]>0, z_fix[,10]*3, -z_fix[,10]*3)+ rnorm(N,0,.5)) #v10 v shaped
  theta7 <- as.vector(sin(z_fix[,-10] %*% b[1:9]) + ifelse(z_fix[,10]>0, z_fix[,10]*3, -z_fix[,10]*3)+ rnorm(N,0,5)) #v10 v shaped more noise
  theta8 <- as.vector(sin(z_fix[,1:8] %*% b[1:8]) + ifelse(z_fix[,9]==1, z_fix[,10]*3, -z_fix[,10]*3)+ rnorm(N,0,5)) #v10 shape depends on v9
  
  #create theta df
  thetadf <- data.frame(theta1, theta2, theta3, theta4, theta5, theta6,theta7, theta8)
  
  z <- z_fix
  
  ### Options for D (m_(X))
  if (random_d == T) {
    d <- rep(c(0, 1), length.out = N)
  } else 
    if(random_d =="imbalanced"){
      d <-  as.numeric(rbinom(N,prob=0.2,size=1))
    }
  else
    if(random_d == "linear"){
      d_prop <- pnorm( z[,k/2] + z[,2] + z[,k/4] - z[,10]) # D is dependent on Za
      d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
    }
  else 
    if(random_d == "interaction"){
      d_prop <- pnorm((z %*% b) + z[,k/2] + z[,2] + z[,k/4]*z[,10]) # D is dependent on Za
      d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
    }
  else{
    d_prop <- pnorm((z %*% b) + sin(z[,k/2]) + z[,2] + cos(z[,k/4]*z[,10])) # D is dependent on Za
    d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
  }
  
  theta <- thetadf[theta_num]
  
  g <- as.vector(z[,k/10] + z[,k/2] + z[,k/4]*z[,k/10])
  
  y <- theta * d + g + rnorm(N,0,var)
  
  data <- as.data.frame(y)
  data <- cbind(data, theta, d, z)
  colnames(data) <- c("y", "theta", "d", c(paste0("V", 1:k)))
  return(data)
}





#### CATE Estimation Function ####

#Assign data set from random data generation

simulateCATEestimation <- function(dataset_from_datagenfnc, test_split=0.8, predict_on_all = F) {
  
  df <- dataset_from_datagenfnc
  df$theta <- NULL
  
  #Set seed for reproducible results
  set.seed(123)
  
  #Train-test split
  train_index <- sample(1:nrow(df), test_split * nrow(df))
  test_index <- setdiff(1:nrow(df), train_index)
  
  #Get the generated data's CATE for MSE of prediction later
  actualtheta <- dataset[test_index,"theta"]
  actualtheta_all <- dataset[,"theta"]
  
  #Set X_train using index
  X_train <- df[train_index,]
  
  #Split into treated X_train
  X_train_treated <- X_train[X_train$d == 1,]
  X_train_treated$d <- NULL
  
  #Split untreated X_train
  X_train_untreated <- X_train[X_train$d == 0,]
  X_train_untreated$d <- NULL
  
  #Set up test
  X_test <- df[test_index, -1]
  y_test <- df[test_index, "y"]
  X_test$d <- NULL
  
  #train model to predict treated and untreated values (T-leaner)
  ranger_treated <- ranger(y ~ ., data = X_train_treated, probability= F)
  
  ranger_untreated <- ranger(y ~ ., data = X_train_untreated, probability= F)
  
  if (predict_on_all == T) {
    #Predict model treated with test
    pred.treated <-predict(ranger_treated, data = df, type = "response")
    
    #Predict model untreated with test
    pred.untreated <-predict(ranger_untreated, data = df, type = "response")
    
    #Estimate CATE, find difference beteen treated and untreated in y
    predtheta = pred.treated$predictions - pred.untreated$predictions
    
    #Check with actual theta from data generated data set
    residuals <- actualtheta_all - predtheta
    res2 <- residuals^2
    MSE = (1/(nrow(df)))*(sum(res2))
    
    #add predictions to dataset
    df$theta_hat <- predtheta
    X_test <- df
    
  } else {
    
    #Predict model treated with test
    pred.treated <-predict(ranger_treated, data = X_test, type = "response")
    
    #Predict model untreated with test
    pred.untreated <-predict(ranger_untreated, data = X_test, type = "response")
    
    #Estimate CATE, find difference beteen treated and untreated in y
    predtheta = pred.treated$predictions - pred.untreated$predictions
    
    #Check with actual theta from data generated data set
    residuals <- actualtheta - predtheta
    res2 <- residuals^2
    MSE = (1/(nrow(X_test)))*(sum(res2))
    
    #add predictions to dataset
    X_test$theta_hat <- predtheta
  }
  
  X_test$y <- NULL
  X_test$d <- NULL
  
  results <- list("Estimated_CATE_df" = X_test, "MSE" = MSE, "original_theta" = df[test_index,]$V10, "estimated_theta" = predtheta)
  
  return(results)
}

#### CATE Estimation Function 2 ####

xgboost_cate <- function(df) {
  
  
  train_index <- sample(1:nrow(df), 0.8 * nrow(df))
  test_index <- setdiff(1:nrow(df), train_index)
  
  X_train <- df[train_index,]
  
  X_train_treated <- X_train[X_train$d == 1,]
  X_train_treated$d <- NULL
  X_train_untreated <- X_train[X_train$d == 0,]
  X_train_untreated$d <- NULL
  
  X_test <- df[test_index, -1]
  y_test <- df[test_index, "y"]
  X_test$d <- NULL
  dim(X_test)
  
  ml_task <- makeRegrTask(data = X_train_treated, target = "y")
  cv_folds <- makeResampleDesc("RepCV",reps=2,folds=3) 
  
  model <- makeLearner("regr.xgboost") # Regression XgBoost model
  model <- setHyperPars(learner = model,
                        par.vals = list(
                          nrounds = 45,
                          max_depth = 6,
                          lambda = 0.5817046,
                          eta = 0.1237508,
                          subsample = 0.5878101,
                          min_child_weight = 2.823564,
                          colsample_bytree = 0.9325822
                        )
  )
  
  
  
  resample(model,ml_task,cv_folds,measures = list(rsq,mse))
  xg_treated <- train(learner = model,task = ml_task)
  
  ml_task <- makeRegrTask(data = X_train_untreated, target = "y")
  cv_folds <- makeResampleDesc("RepCV",reps=2,folds=3) # repeated CV
  model <- makeLearner("regr.xgboost")
  model <- setHyperPars(learner = model,
                        par.vals = list(
                          nrounds = 53,
                          max_depth = 5,
                          lambda = 0.7103812,
                          eta = 0.106243,
                          subsample = 0.7068414,
                          min_child_weight = 1.683222,
                          colsample_bytree = 0.9078216
                        )
  )
  
  resample(model,ml_task,cv_folds,measures = list(rsq,mse))
  xg_untreated <- train(learner = model,task = ml_task)
  
  pred.treated <-predict(xg_treated, newdata = X_test, type = "response")
  pred.untreated <-predict(xg_untreated, newdata = X_test, type = "response")
  
  #Estimate CATE
  predtheta = pred.treated$data - pred.untreated$data 
  predtheta <- unlist(predtheta)
  X_test$theta <- predtheta
  head(X_test)
  return(X_test)
}


######################################### END OF REQUIRED FUNCTIONS #########################################


# Figure 4.1.1
dataset  <- datagen(theta_num = 3)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.2.1

#first row
dataset  <- datagen(theta_num = 1)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#first row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="only test set used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

CATEresults <- simulateCATEestimation(dataset, predict_on_all=T)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="all observations used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)


#second row
dataset  <- datagen(theta_num = 6)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#second row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#second row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="only test set used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

CATEresults <- simulateCATEestimation(dataset, predict_on_all=T)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#second row right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="all observations used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)


#third row
dataset  <- datagen(theta_num = 7)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#third row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#third row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="only test set used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

CATEresults <- simulateCATEestimation(dataset, predict_on_all=T)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#third row right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="all observations used", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.2.2

dataset  <- datagen(theta_num = 7)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="80-20 test split", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

CATEresults <- simulateCATEestimation(dataset, test_split=.5)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="50-50 test split", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.2.3

#first row
dataset  <- datagen(N=500, theta_num = 2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship, N=500",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#first row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=500",
     sub="all relevant variables included", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#first row right
rf_onevar <- ranger(theta_hat ~ V10, data = tr)

pdp_one <- partial(rf_onevar, pred.var = "V10")
plot(pdp_one, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=500",
     sub="only variable of interest used for PDP generation", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#second row
dataset  <- datagen(N=5000, theta_num = 2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#second row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship, N=5000",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#second row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=5000",
     sub="all relevant variables included", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#second row right
rf_onevar <- ranger(theta_hat ~ V10, data = tr)

pdp_one <- partial(rf_onevar, pred.var = "V10")
plot(pdp_one, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=5000",
     sub="only variable of interest used for PDP generation", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)


#third row
dataset  <- datagen(N=50000, theta_num = 2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#third row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship, N=50000",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#third row middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=50000",
     sub="all relevant variables included", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#third row right
rf_onevar <- ranger(theta_hat ~ V10, data = tr)

pdp_one <- partial(rf_onevar, pred.var = "V10")
plot(pdp_one, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP, N=50000",
     sub="only variable of interest used for PDP generation", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.2.4
dataset <- datagen(N=5000, theta_num =2, random_d = T, var=2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#top left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#top right
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="balanced", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#bottom left
dataset <- datagen(N=5000, random_d ="imbalanced", theta_num =2, var=2) 
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="imbalanced", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#bottom right
dataset <- datagen(N=5000, random_d ="linear", theta_num=2, var=2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     sub="treatment selection dependent on features", cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.3.1

#first row
dataset  <- datagen(theta_num = 3)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#first row right
#create wrapper for partial function to develop standard deviations
pred_wrapper <- function(object, newdata) {
  p <- predict(object, data = newdata)$predictions
  p <- Winsorize(p, probs = c(.05, 0.95)) #trim 10% of outliers, can also use other methods mentioned in paper
  c("avg" = mean(p), "avg-1sd" = mean(p) - sd(p), "avg+1sd" = mean(p) + sd(p))
}

#generate PDP with pred_wrapper
pdp_sd <- partial(rf_allvar, pred.var = "V10", pred.fun = pred_wrapper)

#display must be done with ggplot2
pdp_sd_display <- autoplot(pdp_sd) + 
  theme_light() +
  labs(x = "Feature of Interest", y = "CATE") +
  theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank())
grid.arrange(pdp_sd_display, nrow = 1)

#second row
dataset  <- datagen(theta_num = 6)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#second row right
#create wrapper for partial function to develop standard deviations
pred_wrapper <- function(object, newdata) {
  p <- predict(object, data = newdata)$predictions
  p <- Winsorize(p, probs = c(.05, 0.95)) #trim 10% of outliers, can also use other methods mentioned in paper
  c("avg" = mean(p), "avg-1sd" = mean(p) - sd(p), "avg+1sd" = mean(p) + sd(p))
}

#generate PDP with pred_wrapper
pdp_sd <- partial(rf_allvar, pred.var = "V10", pred.fun = pred_wrapper)

#display must be done with ggplot2
pdp_sd_display <- autoplot(pdp_sd) + 
  theme_light() +
  labs(x = "Feature of Interest", y = "CATE") +
  theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank())
grid.arrange(pdp_sd_display, nrow = 1)

#third row
dataset  <- datagen(theta_num = 1)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#third row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#third row right
#create wrapper for partial function to develop standard deviations
pred_wrapper <- function(object, newdata) {
  p <- predict(object, data = newdata)$predictions
  p <- Winsorize(p, probs = c(.05, 0.95)) #trim 10% of outliers, can also use other methods mentioned in paper
  c("avg" = mean(p), "avg-1sd" = mean(p) - sd(p), "avg+1sd" = mean(p) + sd(p))
}

#generate PDP with pred_wrapper
pdp_sd <- partial(rf_allvar, pred.var = "V10", pred.fun = pred_wrapper)

#display must be done with ggplot2
pdp_sd_display <- autoplot(pdp_sd) + 
  theme_light() +
  labs(x = "Feature of Interest", y = "CATE") +
  theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank())
grid.arrange(pdp_sd_display, nrow = 1)




# Figure 4.4.1
dataset  <- datagen(theta_num = 2)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#right
tr_quantile <- tr

#split V10 into quantiles
interval <- c(min(tr_quantile$V10), quantile(tr_quantile$V10, probs=c(0.20, .4, .6, .8)), max(tr_quantile$V10)+1)
tr_quantile$V10quantile <- factor(findInterval(tr_quantile$V10, interval))

#isolate rows with extreme quantile values
tr_quantile_extreme <- tr_quantile[tr_quantile$V10quantile == 1 | tr_quantile$V10quantile == 5,]

#remove quantiles for final PDP generation
tr_quantile_extreme$V10quantile <- NULL

#model for PDP
rf_quantile <- ranger(theta_hat ~ ., data = tr_quantile_extreme)

#PDPs with quantiles
pdp_quantile <- partial(rf_quantile, pred.var = "V10")
plot(pdp_quantile, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP Quantiles with 5K Obs & Random Assignment",
     cex.main=1.5, cex.lab=1.5, cex.axis=1.5, lwd=2)





# Figure 4.4.2
dataset  <- datagen(theta_num = 7)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#middle
plot(pdp, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP",
     cex.main=2, cex.lab=1.5, cex.axis=1.5, lwd=2)

#right
tr_quantile <- tr

#split V10 into quantiles
interval <- c(min(tr_quantile$V10), quantile(tr_quantile$V10, probs=c(0.20, .4, .6, .8)), max(tr_quantile$V10)+1)
tr_quantile$V10quantile <- factor(findInterval(tr_quantile$V10, interval))

#isolate rows with extreme quantile values
tr_quantile_extreme <- tr_quantile[tr_quantile$V10quantile == 1 | tr_quantile$V10quantile == 5,]

#remove quantiles for final PDP generation
tr_quantile_extreme$V10quantile <- NULL

#model for PDP
rf_quantile <- ranger(theta_hat ~ ., data = tr_quantile_extreme)

#PDPs with quantiles
pdp_quantile <- partial(rf_quantile, pred.var = "V10")
plot(pdp_quantile, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP Quantiles",
     cex.main=1.5, cex.lab=1.5, cex.axis=1.5, lwd=2)




# Figure 4.4.3
#first row
dataset  <- datagen(theta_num = 7)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#first row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#first row middle
tr_quantile <- tr

#split V10 into quantiles
interval <- c(min(tr_quantile$V10), quantile(tr_quantile$V10, probs=c(0.20, .4, .6, .8)), max(tr_quantile$V10)+1)
tr_quantile$V10quantile <- factor(findInterval(tr_quantile$V10, interval))

#isolate rows with extreme quantile values
tr_quantile_extreme <- tr_quantile[tr_quantile$V10quantile == 1 | tr_quantile$V10quantile == 5,]

#remove quantiles for final PDP generation
tr_quantile_extreme$V10quantile <- NULL

#model for PDP
rf_quantile <- ranger(theta_hat ~ ., data = tr_quantile_extreme)

#PDPs with quantiles
pdp_quantile <- partial(rf_quantile, pred.var = "V10")
plot(pdp_quantile, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP Quantiles with 5K Obs & Random Assignment",
     cex.main=1.5, cex.lab=1.5, cex.axis=1.5)

#first row right
#create wrapper for partial function to develop standard deviations
pred_wrapper <- function(object, newdata) {
  p <- predict(object, data = newdata)$predictions
  p <- Winsorize(p, probs = c(.05, 0.95)) #trim 10% of outliers, can also use other methods mentioned in paper
  c("avg" = mean(p), "avg-1sd" = mean(p) - sd(p), "avg+1sd" = mean(p) + sd(p))
}

#generate PDP with pred_wrapper
pdp_sd_q <- partial(rf_quantile, pred.var = "V10", pred.fun = pred_wrapper)

#display must be done with ggplot2
pdp_sd_display <- autoplot(pdp_sd_q) + 
  theme_light() +
  labs(x = "Feature of Interest", y = "CATE") +
  theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank())
grid.arrange(pdp_sd_display, nrow = 1)

#second row
dataset  <- datagen(theta_num = 3)
CATEresults <- simulateCATEestimation(dataset)
tr <- CATEresults$Estimated_CATE_df

rf_allvar <- ranger(theta_hat ~ ., data = tr)

pdp <- partial(rf_allvar, pred.var = "V10")

#second row left
plot(dataset$V10,dataset$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE Relationship",
     cex.main=2, cex.lab=1.5, cex.axis=1.5)

#second row middle
tr_quantile <- tr

#split V10 into quantiles
interval <- c(min(tr_quantile$V10), quantile(tr_quantile$V10, probs=c(0.20, .4, .6, .8)), max(tr_quantile$V10)+1)
tr_quantile$V10quantile <- factor(findInterval(tr_quantile$V10, interval))

#isolate rows with extreme quantile values
tr_quantile_extreme <- tr_quantile[tr_quantile$V10quantile == 1 | tr_quantile$V10quantile == 5,]

#remove quantiles for final PDP generation
tr_quantile_extreme$V10quantile <- NULL

#model for PDP
rf_quantile <- ranger(theta_hat ~ ., data = tr_quantile_extreme)

#PDPs with quantiles
pdp_quantile <- partial(rf_quantile, pred.var = "V10")
plot(pdp_quantile, type="l", xlab="Feature of Interest", ylab = "CATE", main="PDP Quantiles with 5K Obs & Random Assignment",
     cex.main=1.5, cex.lab=1.5, cex.axis=1.5)

#second row right
#create wrapper for partial function to develop standard deviations
pred_wrapper <- function(object, newdata) {
  p <- predict(object, data = newdata)$predictions
  p <- Winsorize(p, probs = c(.05, 0.95)) #trim 10% of outliers, can also use other methods mentioned in paper
  c("avg" = mean(p), "avg-1sd" = mean(p) - sd(p), "avg+1sd" = mean(p) + sd(p))
}

#generate PDP with pred_wrapper
pdp_sd_q <- partial(rf_quantile, pred.var = "V10", pred.fun = pred_wrapper)

#display must be done with ggplot2
pdp_sd_display <- autoplot(pdp_sd_q) + 
  theme_light() +
  labs(x = "Feature of Interest", y = "CATE") +
  theme(legend.position = "none", panel.grid.major = element_blank(), panel.grid.minor = element_blank())
grid.arrange(pdp_sd_display, nrow = 1)

############################# Plots for ICE options
set.seed(123)

parallelStartSocket(8)

df <- datagen(N = 50000, k=10,random_d=T,theta_num=8, var=1)
df$theta <- NULL

data <- xgboost_cate(df)

smp_size <- floor(0.75 * nrow(data))

train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]


task = makeRegrTask(data = train, target = "theta")

rf.mlr <- makeLearner("regr.randomForest", predict.type = "response", 
                      par.vals = list("replace" = TRUE, 
                                      "importance" = TRUE, 
                                      "mtry" = 4, 
                                      "sampsize" = 846, 
                                      "ntree" = 989))
rf <- mlr::train(rf.mlr, task = task)

# Figure 5.1

plot(test$V10,test$theta, ylab="CATE", xlab="Variable of Interest", main="True CATE relationship", cex.main=2, cex.lab=1.5, cex.axis=1.5)

pdps <- partial(object = rf$learner.model, train = test, pred.var = "V10")
plot(pdps, type = "l", ylab="CATE", xlab="Variable of Interest", main="PDP", cex.main=2, cex.lab=1.5, cex.axis=1.5)

ice <- partial(object = rf$learner.model , # the model
               train=test,
               pred.var = "V10", # ICE variable to plot
               prob = FALSE, ice=TRUE, center = FALSE, frac_to_build = 0.1)

plotPartial(ice, contour = TRUE, col.regions = NULL, palette = c("viridis","magma", "inferno", "plasma", "cividis"), xlab="Variable of Interest", ylab = "CATE", main="ICE", cex.main= 5, cex.lab= 4, cex.axis= 2)

# Figure 5.1.1 centered ice

ice <- partial(object = rf$learner.model , # the model
               train=test,
               pred.var = "V10", # ICE variable to plot
               prob = FALSE, ice=TRUE, center = TRUE, frac_to_build = 0.1)

plotPartial(ice, contour = TRUE, col.regions = NULL, palette = c("viridis","magma", "inferno", "plasma", "cividis"), ylab="CATE", xlab="Variable of Interest", main="centered ICE", cex.main= 5, cex.lab= 4, cex.axis= 2)

# Figure 5.2.1 colored ice
bh.ice <- ice(object = rf$learner.model , # the model
              X = test, y = test$theta,
              predictor = "V10", frac_to_build = 0.1)

plot(bh.ice,  plot_orig_pts_preds = T, color_by = "V9", ylab="CATE", xlab="Variable of Interest colored by V9", main="colored ICE", cex.main=1.2, cex.lab=1, cex.axis=1)

# Figure 5.3.1 clustered ice
clusterICE(bh.ice, nClusters = 2, plot_legend = TRUE, plot_pdp = FALSE)


# 5.4.1 d-ICE
bhd.dice = dice(bh.ice)
plot(bhd.dice, plot_sd = FALSE, color_by = "V9", ylab="CATE", main="d-ICE", cex.main=3, cex.lab=2, cex.axis=2)

# 5.4.2 d-ICE

df <- datagen(N = 50000, k=10, random_d=T, theta_num=2, var=1)
df$theta <- NULL

data <- xgboost_cate(df)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

task = makeRegrTask(data = train, target = "theta")

rf.mlr <- makeLearner("regr.randomForest", predict.type = "response", 
                      par.vals = list( "replace" = TRUE, 
                                       "importance" = TRUE, 
                                       "mtry" = 4, 
                                       "sampsize" = 846, 
                                       "ntree" = 989))
rf <- mlr::train(rf.mlr, task = task)


bh.ice <- ice(object = rf$learner.model , # the model
              X = test, y = test$theta,
              predictor = "V10", frac_to_build = 0.1)

bhd.dice = dice(bh.ice)
plot(bhd.dice, plot_sd = FALSE, color_by = "V9", ylab="CATE", main="d-ICE", cex.main=3, cex.lab=2, cex.axis=2)
