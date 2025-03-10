Study Reference : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#working-with-data

# Working with data
## Dataset & DataLoader
* 'torch.utils.data.Dataset' stores the samples and their corresponding labels
* 'torch.utils.data.DataLoader' wraps an iterable around the 'Dataset'

## Domain Specific Libraries
* ex) TorchText, TorchVision, TorchAudio

### TorchVision
* 'torchvision.datasets' module contains 'Dataset' objects. 
* Every TorchVision 'Dataset' includes two arguments. To modify the samples and labels repectively.
	* transform 
	* target_transform

* Pass 'Dataset' as an argument to 'DataLoader'. 
	* Wraps an iterable over dataset
	* Supports automatic batching, sampling, shuffling and multiprocess data loading.

```
training_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()
)		
test_data = datasets.FashionMNIST(
	root="data",
	train=False,
	download=True,
	transform=ToTensor()
)


batch_size = 64

# Create data loaders

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
	print(f"Shape of X [N, C, H, W]: {X.shape}")
	print(f"Shape of y : {y.shape} {y.dtype}")
	break
```

# Creating Models
* To define a neural network in Pytorch, create a class that inherits from 'nn.Module'
* Define the layers of the network in the '__init__' function
* Specify how data will pass through the network in the 'forward' funciton 
* To accelerate operations in the neural network, move it to the accelerator. such as CUDA, MPS, MTIA, or XPU

```
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define Model 
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

model = NeuralNetwork().to(device)
```

# Optimizing the Model Parameters
* Loss function & Optimizer are neeed.
```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters, lr=1e-3)
```

* Model makes predictions on the training dataset & backpropagates the prediction error to adjust the model's parameter

```
pred = model(X)
loss = loss_fn(pred, y)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

* Check the model's performance against the test dataset.

```
pred = model(X)
test_loss += loss_fn(pred, y).item()
correct += (pred.argmax(1) == y).type(torch.float).sum().item()
```

# Saving Models
* Serializer the internal state dictionary (Containing the model parameters)

```
torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")
```

# Loading Models
* Re-creating the model structure
* Loading the state dictionary into it

```
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```