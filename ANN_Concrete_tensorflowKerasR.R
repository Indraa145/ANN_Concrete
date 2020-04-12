library(tensorflow)
library(kerasR)

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))
concrete_norm <- as.data.frame(lapply(concrete, normalize))

#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]

X_train = as.matrix(tf_train[,1:8])
X_test = as.matrix(tf_test[,1:8])
y_train = as.matrix(tf_train[,9])
y_test = as.matrix(tf_test[,9])

idx = which(y_train=="benign"); y_train[idx]=0; y_train[-idx]=1; y_train=as.integer(y_train)
idx = which(y_test=="benign"); y_test[idx]=0; y_test[-idx]=1; y_test=as.integer(y_test)

Y_train <- to_categorical(y_train,2)

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

keras_compile(mod, loss = 'categorical_crossentropy', optimizer = RMSprop())

keras_fit(mod, X_train, Y_train, batch_size = 32, epochs = 15, verbose = 2, validation_split = 1.0)

#Validation
Y_test_hat <- keras_predict_classes(mod, X_test)
table(y_test, Y_test_hat)
print(c("Mean validation accuracy = ",mean(y_test == Y_test_hat)))