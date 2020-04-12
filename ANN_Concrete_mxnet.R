library(mxnet)
concrete <- read.csv(file = file.path("data", "Concrete_Data.csv"))

normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))

y = as.matrix(concrete_norm[,9])
y = as.numeric(y)
x = as.numeric(as.matrix(concrete_norm[,1:8]))
x = matrix(as.numeric(x),ncol=9)

train.x = x
train.y = y
test.x = x
test.y = y

mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=c(5,5), out_node=2, out_activation="softmax", num.round=20, array.batch.size=32, learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
## Auto detect layout of input matrix, use rowmajor..
pred.label = max.col(t(preds))-1
table(pred.label, test.y)