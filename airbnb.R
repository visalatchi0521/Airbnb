install.packages("e1071")
install.packages("SparseM")
install.packages("neuralnet")
install.packages("CRAN")
library(dplyr)
library(reshape2)
library(tidyverse)
library(ggplot2)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(magrittr)
library(Matrix)
library(neuralnet)
library('CRAN')
library("e1071")
library('SparseM')
set.seed(1234)

df=read_csv("C:/Users/saluv/OneDrive/Desktop/DA/Kaggle/Airbnb/airbnb_cd.csv")
head(df)
str(df)
anyNA(df)
df$id=as.character(df$id)
df$date_first_booking=as.Date(df$date_first_booking)
df$gender=as.factor(df$gender)
df$age=as.integer(df$age)
df$date_first_booking_month=as.integer(df$date_first_booking_month)
df$date_first_booking_year=as.integer(df$date_first_booking_year)
df$country_destination=as.factor(df$country_destination)
df$signup_flow=as.factor(df$signup_flow)
df$signup_method=as.factor(df$signup_method)
df$language=as.factor(df$language)
df$affiliate_channel=as.factor(df$affiliate_channel)
df$language=as.factor(df$language)
df$first_affiliate_tracked=as.factor(df$first_affiliate_tracked)
df$affiliate_provider=as.factor(df$affiliate_provider)
df$first_browser=as.factor(df$first_browser)
df$signup_app=as.factor(df$signup_app)
df$first_device_type=as.factor(df$first_device_type)
df$first_browser=as.factor(df$first_browser)



index=createDataPartition(y=df$country_destination,p=0.7,list=FALSE)
df_train=df[index,]
df_test=df[-index,]
str(df_train)
head(df_train[-c(1:3)])
x <- as.matrix(df_train[,-16])

# put the labels in a separate vector
y <- df_train[,16]

# convert to compressed sparse row format
xs <- as.matrix.csr(x)

# write the output libsvm format file 
write.matrix.csr(xs, y=y, file="out.txt")


write.csv(df_train,'df_train.csv')
write.csv(df_test,'df_test.csv')


  
  
trainm=sparse.model.matrix(country_destination~.-1,-16,data=df_train)
train_label=df_train[,"country_destination"]
train_matrix=xgb.DMatrix(data=as.matrix(trainm),label=train_label)

testm=sparse.model.matrix(country_destination~.-1,-16,data=df_test)
head(testm)
test_label=df_test[,"country_destination"]
test_matrix=xgb.DMatrix(data=as.matrix(testm),label=test_label)

# Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000,
                       watchlist = watchlist,
                       eta = 0.001,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss == 0.625217,]

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = nc, ncol = length(p)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)


#nn
m <- model.matrix(~date_account_created+timestamp_first_active+date_first_booking+gender+age+signup_method+signup_flow+language+affiliate_channel+affiliate_provider+first_affiliate_tracked+signup_app+first_device_type+first_browser+country_destination+date_first_booking_month+date_first_booking_year,data = df_train)
data=data.frame(m)
dim(data)
head(data[,122:131])


nn=neuralnet(country_destinationUS+country_destinationCA+country_destinationFR+country_destinationIT+country_destinationNL+country_destinationES+country_destinationother+country_destinationDE+country_destinationGB+country_destinationNL+country_destinationPT~.,data=data,hidden=2,err.fct="ce",linear.output=FALSE)
plot(nn)
output=compute(nn,data[-c(122:131)])
head=output$net.result
head