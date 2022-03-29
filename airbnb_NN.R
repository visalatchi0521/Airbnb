install.packages("neuralnet")
library(neuralnet)
set.seed(333)
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
