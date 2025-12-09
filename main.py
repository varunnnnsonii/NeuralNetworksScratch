import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

 
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights =0.1*np.random.randn(n_inputs,n_neurons) # randomn weights
        self.biases=np.zeros((1,n_neurons)) #biases to zero
        pass
    def forward(self,inputs):
        self.output =np.dot(inputs,self.weights)+self.biases #(inputs * weights) + bias
        pass

class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs) #(activation function) Rectified linear activation fnc ,0 for -ve ,x for x where x -> +ve



# we use exponential bcz the relu directly cuts the negative value to 0 ,but the difference between 2 values decreases loosing dept ,
#eg diff between 2 and -100 ,and 2 and -2 theres a difference ,but after relu we know that difference is just 2 and we see it as same 
#due to the use of e the values tend to be too long after the decimal hence ,before exponentiating we subtract my the max,
# (bigger the number longer the exponential value)
# so the highest values goes to 0 and the rest values also reduce eg 100,95,70 goes to 0,5,30


class Activation_Softmax: 
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1 , keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities

class Loss:
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self , y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods= -np.log(correct_confidences)
        return negative_log_likelihoods

X,y =spiral_data(samples=100,classes=3) #logic in dataset.py 

dense1 =Layer_Dense(2,3) # input is 2 here indicating x and y coordinate of a particular point ,3 neurons (number of neurons defines no of outputs of that layer)
activation1=Activation_ReLU()
dense2=Layer_Dense(3,3)
activation2=Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function=Loss_CategoricalCrossentropy()
loss=loss_function.calculate(activation2.output,y)

print("loss:",loss)





# layer1.forward(X)
# activation1.forward(layer1.output)
# print(activation1.output)















