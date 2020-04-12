# Deep Learning - ANN on Concrete Dataset
## Deep Learning Homework 3 No. 7 | Indra Imanuel Gunawan - 20195118
This is R implementation on the Concrete Dataset. There are 5 R packages that is used in this experiments, which are:
1. Rneuralnet
2. h20
3. mxnet
4. TensorFlow & KerasR
5. TensorFlow & Keras
I will explain each of the code in this report.

## Rneuralnet
First, load the neuralnet libary
```R
library(neuralnet)
```
Load the data, and normalize it
```R
concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
Split it into training set and test set
```R
#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]
```

Build the neural network model
```R
concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = concrete_train, hidden = 5 )
```
Build the predictor and see its performance
```R
model_results2 <- compute(concrete_model2, concrete_test[1:8])

#storing the results
predicted_strength2 <- model_results2$net.result

cor(predicted_strength2, concrete_test$strength)
```

## H2O
Load the h2o library
```R
library(h2o)
```
Initialize the h2o
```R
localH2O = h2o.init(ip="127.0.0.1", port = 50001, 
                    startH2O = TRUE, nthreads=-1)
```
Load and normalize the data
```R
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- h2o.importFile("data/Concrete_Data.csv")
concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
Split it into training and test dataset
```R
#training set
train <- concrete_norm[1:773, ]

#test set
test <- concrete_norm[774:1030, ]
```
Set the x and y
```R
y = names(train)[9]
x = names(train)[1:8]

train[,y] = as.factor(train[,y])
test[,y] = as.factor(train[,y])
```
Build and run the model
```R
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
```

## MXNET
Load the mxnet library
```R
library(mxnet)
```
Load the data and normalize it
```R
concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
Set the X and Y, and set the training and test dataset
```R
y = as.matrix(concrete_norm[,9])
y = as.numeric(y)
x = as.numeric(as.matrix(concrete_norm[,1:8]))
x = matrix(as.numeric(x),ncol=9)

train.x = x
train.y = y
test.x = x
test.y = y
```
Build and run the model, and the predictor as well, to see the model performance
```R
mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(5,5), out_node=2, out_activation="softmax", num.round=20, array.batch.size=32, learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
## Auto detect layout of input matrix, use rowmajor..
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
```

## TensorFlow & KerasR
Load TensorFlow and KerasR library
```R
library(tensorflow)
library(kerasR)
```
Load the data and normalize it
```R
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))
concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
Split the data into training and test dataset, also set the x and y
```R
#training set
tf_train <- concrete_norm[1:773, ]

#test set
tf_test <- concrete_norm[774:1030, ]

X_train = as.matrix(tf_train[,1:8])
X_test = as.matrix(tf_test[,1:8])
y_train = as.matrix(tf_train[,9])
y_test = as.matrix(tf_test[,9])
```
Build the neural network
```R
n_units = 512 

mod <- Sequential()
mod$add(Dense(units = n_units, input_shape = dim(X_train)[2]))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(units = n_units))
mod$add(LeakyReLU())
mod$add(Dropout(0.25))

mod$add(Dense(2))
mod$add(Activation("softmax"))
```
Compile the model and fit the data into the model
```R
keras_compile(mod, loss = 'categorical_crossentropy', optimizer = RMSprop())

keras_fit(mod, X_train, Y_train, batch_size = 32, epochs = 15, verbose = 2, validation_split = 1.0)
```
See how well does the model perform on the dataset (its accuracy)
```R
Y_test_hat <- keras_predict_classes(mod, X_test)
table(y_test, Y_test_hat)
print(c("Mean validation accuracy = ",mean(y_test == Y_test_hat)))
```

## TensorFlow and Keras
Load all of the necessary libraries
```R
library(magrittr)
library(tensorflow)
library(keras)
```
Load and normalize the dataset
```R
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))
concrete_norm <- as.data.frame(lapply(concrete, normalize))
```
Split the data into training and test dataset, also set the x and y
```R
#training set
tf_train <- concrete_norm[1:773, ]

#test set
tf_test <- concrete_norm[774:1030, ]

X_train = as.matrix(tf_train[,1:8])
X_test = as.matrix(tf_test[,1:8])
y_train = as.matrix(tf_train[,9])
y_test = as.matrix(tf_test[,9])
```
Build the neural network
```R
model <- keras_model_sequential() 

n_units = 100
model %>% 
  layer_dense(units = n_units, 
              activation = 'relu', 
              input_shape = dim(X_train)[2]) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')
```
Compile the model and fit the data into the model
```R
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model %>% fit(
  X_train, Y_train, 
  epochs = 5, batch_size = 32, verbose = 1, 
  validation_split = 0.1
)
```
