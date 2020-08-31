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
