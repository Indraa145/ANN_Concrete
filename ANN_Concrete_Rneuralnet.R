library(neuralnet) #artifical neural network 

concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))

#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]

#building the new model
concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = concrete_train, hidden = 5 )

#nuilding the new predictor
model_results2 <- compute(concrete_model2, concrete_test[1:8])

#storing the results
predicted_strength2 <- model_results2$net.result

cor(predicted_strength2, concrete_test$strength)