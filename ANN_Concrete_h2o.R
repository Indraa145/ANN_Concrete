library(h2o)
localH2O = h2o.init(ip="127.0.0.1", port = 50001, 
                    startH2O = TRUE, nthreads=-1)

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

data <- h2o.importFile("data/Concrete_Data.csv")

#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]

y = names(train)[9]
x = names(train)[1:8]

train[,y] = as.factor(train[,y])
test[,y] = as.factor(train[,y])

model = h2o.deeplearning(x=x, 
                         y=y, 
                         training_frame=train, 
                         validation_frame=test, 
                         distribution = "multinomial",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)

print(model)