if(!require("xgboost")) install.packages("xgboost"); library(xgboost)
if(!require("ICEbox")) install.packages("ICEbox"); library(ICEbox)
if(!require("parallelMap")) install.packages("parallelMap"); library(parallelMap)


set.seed(123)

parallelStartSocket(8)

#1. Generate a dataset

df <- datagen(N = 50000, k=10,random_d=T,theta_num=8, var=1)
df$theta <- NULL

#2. Apply xgboost and estimate CATE


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


# 3. Plot PDP & ICE

data <- X_test

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
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

plot(test$V10,test$theta, ylab="CATE", xlab="Variable of Interest", main="PDP", cex.main=2, cex.lab=1.5, cex.axis=1.5)

pdps <- partial(object = rf$learner.model, train = test, pred.var = "V10")
plot(pdps, type = "l", ylab="CATE", xlab="Variable of Interest", main="PDP", cex.main=2, cex.lab=1.5, cex.axis=1.5)

ice <- partial(object = rf$learner.model , # the model
               train=test,
               pred.var = "V10", # ICE variable to plot
               prob = FALSE, ice=TRUE, center = FALSE, frac_to_build = 0.1)

plotPartial(ice, contour = TRUE, col.regions = NULL, palette = c("viridis","magma", "inferno", "plasma", "cividis"), ylab="", xlab="Variable of Interest", main="ICE", cex.main= 5, cex.lab= 4, cex.axis= 2)

# Figure 5.1.1 centered ice

ice <- partial(object = rf$learner.model , # the model
               train=test,
               pred.var = "V10", # ICE variable to plot
               prob = FALSE, ice=TRUE, center = TRUE, frac_to_build = 0.1)

plotPartial(ice, contour = TRUE, col.regions = NULL, palette = c("viridis","magma", "inferno", "plasma", "cividis"), ylab="", xlab="Variable of Interest", main="ICE", cex.main= 5, cex.lab= 4, cex.axis= 2)

# Figure 5.2.1 colored ice
bh.ice <- ice(object = rf$learner.model , # the model
              X = test, y = test$theta,
              predictor = "V10", frac_to_build = 0.1)

plot(bh.ice,  plot_orig_pts_preds = T, color_by = "V9", ylab="CATE", xlab="Variable of Interest colored by V9", main="colored ICE", cex.main=1.2, cex.lab=1, cex.axis=1)

# Figure 5.3.1 clustered ice
clusterICE(bh.ice, nClusters = 2, plot_legend = TRUE, plot_pdp = FALSE)

# 5.4.1 d-ICE
bhd.dice = dice(bh.ice)
plot(bhd.dice, plot_sd = FALSE, color_by = "V9", ylab="CATE", main="d-ICE", cex.main=1.5, cex.lab=1.5, cex.axis=1.5)

# 5.4.2 d-ICE

df <- datagen(N = 50000, k=10, random_d=T, theta_num=2, var=1)
df$theta <- NULL

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


data <- X_test

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
plot(bhd.dice, plot_sd = FALSE, color_by = "V9", ylab="CATE", main="d-ICE", cex.main=1.5, cex.lab=1.5, cex.axis=1.5)
