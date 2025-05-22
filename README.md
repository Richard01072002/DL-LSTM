# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## THEORY
Named Entity Recognition (NER) is a task in natural language processing (NLP) that identifies and classifies named entities (like person names, organizations, locations, etc.) in text. A BiLSTM model is commonly used for this task as it can capture dependencies from both directions in a sequence, enhancing context understanding.


## Neural Network Model
neural network model diagram.

[Input Embeddings] --> [BiLSTM Layer] --> [Linear Layer (Tag Scores)] --> [Softmax Layer] --> [NER Tags]

## DESIGN STEPS
### STEP 1: 

Prepare the dataset for NER with token-level annotations (e.g., using CoNLL 2003 format).

### STEP 2: 

Preprocess the text: tokenize, build vocabulary, convert tokens and tags to IDs.

### STEP 3: 

Define the BiLSTM model class using PyTorch.

### STEP 4: 

 Initialize loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 5: 

Train the model over multiple epochs using training and validation sets.


### STEP 6: 

 Evaluate the model and plot the loss vs. epoch curve. Test with sample input.



## PROGRAM
```

```
### Name: RICHARDSON A

### Register Number: 212222233005

```python
class BiLSTMTagger(nn.Module):
    # Include your code here







    def forward(self, input_ids):
        # Include your code here
        


model = 
loss_fn = 
optimizer = 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions



    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="698" alt="Screenshot 2025-05-22 at 1 53 50 PM" src="https://github.com/user-attachments/assets/452a2c03-21ad-41ac-bda4-b079dbc8de4a" />


### Sample Text Prediction


## RESULT

An LSTM-based model was successfully developed for Named Entity Recognition. The model was trained and evaluated using token-labeled data. The results showed decreasing loss over epochs, indicating effective learning. The model was also able to predict named entities on a sample input sentence.
