# Digit_Recognizer_Ann
 Digit Recognizer Model (Sing language img  data)

When we generate backward forward propagation model in machine learning, normally we get 1 prediction after multiply(weights and inputs) and activation function process.
But there is difference between deep learning and machine learning. In machine learning neurons are isolated from each other.Data normalization is somewhat part of it, but in a good way. Of course we dont want some of features dominate entire model. In case of deep learning, expectation is neurons to interact with each other.


Forward backward propagation in machine learning:
![Adsız](https://user-images.githubusercontent.com/65484711/161892956-bcecd3c6-d41d-4877-ac49-d98d24adff28.png)



in deep learning:

![Adsız1](https://user-images.githubusercontent.com/65484711/161893150-f4991714-207a-4b85-b2b1-8a96250cde5b.png)

As you see, first step exactly same with machine learning, except output(prediction). The neurons in the first layer interact with each other through the nodes in the second layer.

For this network, as many inputs as the number of nodes in the next layer are required from previous layer. 

In this case 3.
Therefore, we must define our parameters in accordance with the layers.

In code:

```
parameters = { "layer1_weights": np.random.randn(3,4096) * 0.1,
               "bias1": np.zeros((3,1)),
               "layer2_weights": np.random.randn(1,3) * 0.1,
               "bias2": np.zeros((Y_train.shape[0],1))
                  }
                  
```

![adsız22](https://user-images.githubusercontent.com/65484711/161894979-0812b8f9-d9e3-4f05-abb3-8a861c66e5a9.png)

Now, first layer's predictions are suitable for second layer.

![Adsız3](https://user-images.githubusercontent.com/65484711/161895013-9927d646-fca1-400a-9fc2-c6e92bdf37a6.png)

