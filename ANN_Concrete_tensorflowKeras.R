library(magrittr)
library(keras)

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))
concrete_norm <- as.data.frame(lapply(concrete, normalize))

#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]

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