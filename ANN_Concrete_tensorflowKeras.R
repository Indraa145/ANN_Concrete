library(magrittr)
library(tensorflow)
library(keras)

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))
concrete_norm <- as.data.frame(lapply(concrete, normalize))

#training set
tf_train <- concrete_norm[1:773, ]

#test set
tf_test <- concrete_norm[774:1030, ]

X_train = as.matrix(tf_train[,1:8])
X_test = as.matrix(tf_test[,1:8])
y_train = as.matrix(tf_train[,9])
y_test = as.matrix(tf_test[,9])

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