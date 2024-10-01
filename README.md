## <b>Neural Networks to recognize hand written digits</b> 
\- Implemented neural networks to predict the hand written digits
  - Load the data
  - Scale the data to get better accuracy
  - Build a neural network model using keras API with activation function - sigmoid function, optimizer - adam, error metric - sparse_categorical_crossentropy, performance metric - accuracy
  - Evaluate and predict the model
  - Evaluate the performance
  - Build the neural network with hidden layers for better accurate results

<pre>
<h3><b>Arguments</b></h3>
> units: Positive integer, dimensionality of the output space.
> activation: Activation function to use. If you don't specify anything,
> no activation is applied (ie. "linear" activation: a(x) = x).
> use_bias: Boolean, whether the layer uses a bias vector.
> kernel_initializer: Initializer for the kernel weights matrix.
> bias_initializer: Initializer for the bias vector. 
> kernel_regularizer:Regularizer function applied to the kernel weights matrix.
> bias_regularizer: Regularizer function applied to the bias vector.
> activity_regularizer: Regularizer function applied to the output of the layer (its "activation").. 
> kernel_constraint: Constraint function applied to the kernel weights matrix. 
> bias_constraint: Constraint function applied to the bias vector.
> lora_rank: Optional integer. If set, the layer's forward pass will implement LoRA (Low-Rank Adaptation) with the provided rank.
</pre>

## <b>Activation functions</b> 
\- Activation function in neural networks is a mathematical function that determines the output of a neuron based on its input. As the name suggests, it is some kind of function that should "activate" the neuron. Three mathematical functions that are commonly used as activation functions are sigmoid, tanh, and ReLU.
 - The sigmoid function (discussed above) performs the following transform on input, producing an output value between 0 and 1
 - The tanh (short for "hyperbolic tangent") function transforms input <i>x</i> to produce an output value between –1 and 1
 - The rectified linear unit activation function (or ReLU, for short) transforms output using the following algorithm: <br>
  &emsp;&emsp;\- If the input value is less than 0, return 0. <br>
  &emsp;&emsp;\- If the input value is greater than or equal to 0, return the input value. <br>
  ReLU can be represented mathematically using the max() function 

## <b>Matrix Operations</b>
\- Dot product multiplicaion of matrices

## <b>Loss functions</b>
\- Loss functions in deep learning are used to measure how well a neural network model performs. <br>
### Tensorflow loss value examples
- sparse_categorical_crossentropy
- binary_crossentropy
- categorical_crossentropy
- mean_absolute_error
- mean_squared_error
Log loss is used for Logistic regression because it is a non linear function and gradient descent optimization is not applicable because of its non convex curve

## Gradient Descent 
\- Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks. It trains machine learning models by minimizing errors between predicted and actual results.
Implementing gradient descent using keras and in python
- Importing libraries
- Loading the data
- Splitting the data in training and testing data
- Scaling the data
- Building a dense neural network
- Evaluating and predicting the model
Implementing different types of Gradient Descent through a Machine Learning Model
- Batch Gradient Descent, Stochastic Gradient Descent, Mini Batch Gradient Descent
