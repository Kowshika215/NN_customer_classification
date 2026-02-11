<img width="1034" height="187" alt="image" src="https://github.com/user-attachments/assets/0c7cf2a0-2884-4a04-91a5-bfeeecf8b979" /># Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1058" height="531" alt="image" src="https://github.com/user-attachments/assets/393c5390-fd0a-4f48-97fc-4507a1f26e8c" />

## DESIGN STEPS

### STEP 1:
Loading the dataset

### STEP 2:
Split the dataset into training and testing

### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:
Build the Neural Network Model and compile the model.

### STEP 5:
Train the model with the training data.

### STEP 6:
Plot the performance plot

### STEP 7:
Evaluate the model with the testing data.

## PROGRAM

### Name: Kowshika R
### Register Number: 212224220049
```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

## Dataset Information

<img width="1034" height="187" alt="image" src="https://github.com/user-attachments/assets/c1112f0c-34bd-4ca9-8e95-c208c6ca30a6" />


## OUTPUT

### Confusion Matrix

<img width="918" height="685" alt="image" src="https://github.com/user-attachments/assets/366da1e9-c085-4ed1-b6e3-81f82e770a18" />


### Classification Report

<img width="917" height="538" alt="image" src="https://github.com/user-attachments/assets/883034d7-01ea-43c6-a003-9d42b855ab95" />


### New Sample Data Prediction

<img width="1034" height="101" alt="image" src="https://github.com/user-attachments/assets/d2f895d4-55a7-4aa2-902c-c09be12f4ffe" />

## RESULT
Thus neural network classification model is developded for the given dataset.
