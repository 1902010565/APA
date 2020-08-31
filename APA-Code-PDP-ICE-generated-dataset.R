if(!require("xgboost")) install.packages("xgboost"); library(xgboost)
if(!require("ICEbox")) install.packages("ICEbox"); library(ICEbox)
if(!require("parallelMap")) install.packages("parallelMap"); library(parallelMap)


set.seed(123)

parallelStartSocket(8)

#1. Generate a dataset

df <- datagen(N = 50000, k=10,random_d=T,theta_num=8, var=1)
df$theta <- NULL

#2. Estimate CATE
data <- xgboost_cate(df)

# 3. Plot PDP & ICE

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
plot(bhd.dice, plot_sd = FALSE, color_by = "V9", ylab="CATE", main="d-ICE", cex.main=1.5, cex.lab=1.5, cex.axis=1.5)
