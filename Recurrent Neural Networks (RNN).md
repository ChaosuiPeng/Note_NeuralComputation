# Recurrent Neural Networks (RNN)
In this exercise, you will learn the following:
* data generation
* What a one-hot encoding is
* How to define an RNN model
* How to train and test an RNN model

## Example One: Generate the next word in a sentence using a RNN
We aim to predict next word in a sentence when given the first word or the first few words. 
The model will be fed with sentences and will predict what the next character in the sentence will be. 
This process will repeat itself until we generate the next word. 
To keep this short and simple, we won't be using any large or external datasets. 
Instead, we'll just be defining a few sentences to see how the model learns from these sentences. 

We will first need to import some necessary libraries
* **numpy** provides a high-performance multidimensional array object, and tools for working with these arrays. 
* **torch library**, an open source machine learning framework.
* **torch.nn** is a modular interface specially designed for neural networks

```python
import torch
from torch import nn
import numpy as np
```

### Data Generation

First, we'll define the sentences that we want our model to output when fed with the first word or the first few characters.

Then we'll create a dictionary out of all the characters that we have in the sentences and map them to an integer. 

* `text`: *text* contains four sentences we define.   

* `int2char`: converting our input integers to their respective characters. 

* `char2int`: converting our input characters to their respective integers.

```python
text = ['hey we are teaching deep learning','hey how are you', 'have a nice day', 'nice to meet you']

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}
```

The char2int dictionary will look like this: It holds all the letters/symbols that were present in our sentences and maps each of them to a unique integer.

```python
char2int
```

Next, we'll be padding our input sentences to ensure that all the sentences are of standard length. While RNNs are typically able to take in variably sized inputs. We will usually want to feed training data in batches to speed up the training process. In order to use batches to train our network, we'll need to ensure that each sequence within the input data is of equal size.
 * `maxlen = len(max(text, key=len))`: finding the length of the longest string in our data.
 * `text[i] += ' '`: padding the rest of the sentences with blank spaces to match that length.

```python
# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))

# Padding
# maxlen+=1 means adding a ' ' at each sentence, which helps to predict last word of sentences
maxlen+=1
# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of
# the sentence matches the length of the longest sentence
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += ' '
```

As we're going to predict the next character in the sequence at each time step, we'll have to divide each sentence into:

* `input_seq`:The last input character should be excluded as it does not need to be fed into the model (we have no data to tell us what the character after this)
* `target_seq`:One time-step ahead of the Input data as this will be the "correct answer" for the model at each time step corresponding to the input data

```python
# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])
    
    # Remove first character for target sequence
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
```

For example, our input sequence and target sequence will look like this:

* Input Sequence: hey how are yo  
* Target Sequence: ey how are you  

The target sequence will always be one-time step ahead of the input sequence.

* `input_seq[i]`: converting our input sequences to **sequences of integers**.  
* `target_seq[i]`: converting our target sequences to **sequences of integers**. 

This will allow us to one-hot-encode our input sequence subsequently.

```python
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]
```


### One-Hot Encoding
One-hot encoding is a method of representing some some categories, in a way which avoids suggesting that some categories are more signifcant than others. 
If we were to just use integers to represent characters the character represented by 8 is not neccessarily more important than the character represented by 1, 
however it may lead to a larger impulse. 
To prevent this we use the integer representation explained before to dictate the index where a 1 will be present in the one-hot encoding, 
whilst all other index positions will be 0. 

#### One-hot encoding example:
If we have dictionary containing 5 characters `{'a':0,'b':1,'c':2,'d':3,'e':4}` then the representation of the letter a would be $(1,0,0,0,0)$ and the representation of the letter d would be $(0,0,0,1,0)$.  
  
Whilst one-hot encoding in our case is used for characters, it could also be used to represent words in a vocabulary or other un-ordered categorical data. 
Before encoding our input sequence into one-hot vectors, we'll define 3 key variables:

* `dict_size`: Dictionary size - The number of unique characters that we have in our text.   
This will determine the length of each one-hot vector, as each character will correspond to a unique index in the one-hot encoded vectors.
* `seq_len`: The length of the sequences that we're feeding into the model.  
As we standardized the length of all our sentences to be equal to the longest sentences, this value will be the `maxlen - 1` as we removed the last character input as well
* `batch_size`: The number of sentences that we defined and are going to feed into the model as a batch


The function `one_hot_encode()` is defined that creates arrays of zeros for each character and replaces the corresponding character index with a 1. 
It is applied to the whole batch. 

```python
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features
```

Now we can take our variables into the function. 

```python
# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print(input_seq.shape)
```

Since we're done with all the data pre-processing, we can now move the data from NumPy arrays to PyTorch's very own data structure - Torch Tensors.
* `input_seq = torch.from_numpy(input_seq)`: Creating a Tensor from a numpy.ndarray (`input_seq`).
* `target_seq = torch.Tensor(target_seq)`: Constructing a tensor with data.


```python
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
```

### Defining RNN Model



Now we've reached the fun part of this project! We'll be defining the model using the Torch library, 
and this is where you can add or remove layers, including fully connected layers, convolutional layers, 
vanilla RNN layers, LSTM layers, and many more! In this tutorial, 
we'll be using the basic `nn.rnn` module to demonstrate a simple example of how RNNs can be used.


<img src="fig/rnn.png" width=70%>

Classes provide a means of bundling data and functionality together. All classes have a function called `__init__()`, 
which is always executed when the class is being initialised. 
We have seen in earlier tutorials how this function is used to define the layers of the network (and therefore the parameters). 

We define a class that inherits PyTorch’s base class(nn.module) for all neural network modules to start building our own RNN model. 

* `__init__()`: Defining some variables and also the layers for our model under the constructor.  
    **self.rnn = nn.RNN()**: Defining 1 layer of RNN. This module uses a standard RNN layer with hidden to hidden connections (as opposed to output to hidden connections). The non-linearity (which is tanh unless specified otherwise) is built into this module. We should also note that this module will act as a sequence to sequence RNN so we will need some code later to extract the next letter as the output.
    
    **self.fc = nn.Linear()**: This is The fully connected layer, which will be in charge of converting the RNN output to our desired output shape.
   
    

* `forward()`: Defining the forward pass function. It is executed sequentially, therefore we’ll have to pass the inputs and the zero-initialized hidden state through the RNN layer first, before passing the RNN outputs to the fully-connected layer.  
    **out = out.contiguous().view()**: Reshaping the outputs such that it can be fit into the fully connected layer. contiguous() returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.
    

* `init_hidden()`:Initializing the hidden state. This basically creates a tensor of zeros in the shape of our hidden states.

You will notice in this case that the `__init()__` function takes some inputs. 
This means when we initialise the class we will have to input some arguments. 
In the case of the network below, this allows us to define the dimension of the hidden layer and the number of layers of the network when we intialise it. 

```python
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
```

After defining the model above, we'll have to instantiate the model with the relevant parameters and define our hyper-parameters as well. The hyper-parameters we're defining below are:

* `n_epochs`: Number of Epochs --> Number of times our model will go through the entire training dataset.    
* `lr`: Learning Rate --> Rate at which our model updates the weights in the cells each time back-propagation is done.
* `criterion = nn.CrossEntropyLoss()`: Using CrossEntropyLoss as loss function. Note that this uses the integer class labels as targets, so there is no need to convert the target sequences to to one-hot encoded vectors.  
* `optimizer = torch.optim.Adam()`: Using the common Adam optimizer as optimizer.


```python
# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)


# Define hyperparameters
n_epochs = 200
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```


### Experiments
#### Training RNN
Now we can begin our training! As we only have a few sentences, this training process is very fast. 
However, as we progress, larger datasets and deeper models mean that the input data is much larger and the number of parameters within the model that we have to compute is much more.
You will recognise the commands used for training this RNN from previous tutorials.
* `optimizer.zero_grad()`: Clearing existing gradients from previous epoch.  
* `output, hidden = model(input_seq)`: Feeding input data into the defined model.  
* `loss = criterion()`: Calculating the loss values for each epoch.  
* `loss.backward()`: Doing backpropagation and calculating gradients.  
* `optimizer.step()`: Updating the weights accordingly.


```python
# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
```

#### Testing RNN

Let’s test our model now and see what kind of output we will get. 
* `predict()`: This is a function to predict next characters with input characters and trained model.  
    **prob = nn.functional.softmax()**: It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
    **char_ind = torch.max()**: Taking the class with the highest probability score from the output.
    

```python
# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden
```

* `sample()` :This function takes the model and the input first word or first a few words as arguments, 
* returning the produced next word. The function does this by repeatedly applying the `predict()` function to predict the next character, 
* and appending this to the end of the string. The loop is broken when the next predicted character is a space, as this indicates the word is complete.


```python
def sample(model, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = maxlen
    # Now pass in the previous characters and get a new one
    c=0
    for ii in range(size):
        char, h = predict(model, chars)
        c+=1
        if char==' ' and c>1:
            break
        chars.append(char)

    return ''.join(chars)
```

Let's run the function with our model and the starting words 'hey'.

```python
sample(model, 'hey we are teaching deep')
```

As we can see, the model is able to come up with the word ‘learning‘ if we feed it with the words ‘hey we are teaching deep’. 
When we feed with the first word or the first few characters in a sentence, the model will predict next word in this sentence.

```python
sample(model, 'hey how are')
```

```python
sample(model, 'nice to meet')
```

```python
sample(model, 'have a nice')
```


## Example Two: Sentiment analysis with an RNN

In this example, you'll implement a recurrent neural network that performs sentiment analysis. 
RNNs have an advantage over feed-forward neural networks since we can include information about the *sequence* of words. 

Here we'll use a dataset of movie reviews, accompanied by sentiment labels: positive or negative.

<img src="fig/reviews_ex.png" width=40%>

We will first need to import some necessary libraries
* **string.punctuation** is a pre-initialized string used as string constant. In python, string.punctuation will give the all sets of punctuation. 
* **collections.Counter** is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
* **torch.utils.data.TensorDataset** takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
* **torch.utils.data.DataLoader** creates DataLoaders and batch our training, validation, and test Tensor datasets.


```python
import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
```

### Load in and visualize the data

We read the reviews and labels from their respective text files. 
Each of the files are read into a string object which can then be indexed. 
It should be noted that at this point the indexing the string is in terms of characters rather than words.

```python
# read data from text files
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()
```

```python
print(reviews[:100])
print()
print(labels[:20])
```

#### Data pre-processing

The first step when building a neural network model is getting your data into the proper form to feed into the network. 
Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.

You can see an example of the reviews data above. Here are the processing steps, we'll want to take:
* We'll want to get rid of periods and irrelevant punctuation.
* Also, you might notice that the reviews are delimited with newline characters `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter. 
* Then I can combined all the reviews back together into one big string.

First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.
* `reviews.lower()`:The lower() method returns a string where all characters are lower case.  Symbols and Numbers are ignored.


```python
# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])


# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()
```

```python
all_text[:40]
```

```python
words[:30]
```

### Encoding the words

The embedding lookup requires that we pass in integers to our network. 
The easiest way to do this is to create dictionaries that map the words in the vocabulary to integers. 
Then we can convert each of our reviews into integers so they can be passed into the network.

```python
## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
```

```python
# stats about vocabulary
print('Unique words: ', len((vocab_to_int))) 
print('Original review: ', reviews_split[1])
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])
```

### Encoding the labels

Our labels are "positive" or "negative". To use these labels in our network, we need to convert them to 0 and 1.

* `encoded_labels`: Converting the labels of "positive" or "negative" to 0 and 1.

```python
# 1=positive, 0=negative label conversion
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
```

### Generating Reviews with the Same Length

As an additional pre-processing step, we want to make sure that our reviews are in good shape for standard processing. 
That is, our network will expect a standard input text size, and so, we'll want to shape our reviews into a specific length. 
We'll approach this task in two main steps:

1. Getting rid of extremely long or short reviews; the outliers
2. Padding/truncating the remaining data so that we have reviews of the same length.

* `review_lens = Counter()`: Calculating the length of each review.

Before we pad our review text, we should check for reviews of extremely short or long lengths; outliers that may mess with our training.

```python
# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
```

```python
print('Number of reviews before removing outliers: ', len(reviews_ints))

## remove any reviews/labels with zero length from the reviews_ints list.

# get indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# remove 0-length reviews and their labels
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints))
```

To deal with both short and very long reviews, we'll pad or truncate all our reviews to a specific length. 
For reviews shorter than some `seq_length`, we'll pad with 0s. For reviews longer than `seq_length`, we can truncate them to the first `seq_length` words. A good `seq_length`, in this case, is 200.

* `features[i, -len(row):] = np.array(row)[:seq_length]`: For reviews shorter than `seq_length` words,
*  **left pad** with *0*s. For reviews longer than `seq_length`, use only the first `seq_length` words as the feature vector.

```python
seq_length = 200

# getting the correct rows x cols shape
features = np.zeros((len(reviews_ints), seq_length), dtype=int)

# for each review, I grab that review and 
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_length]


## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])
```

### Training, Validation, Test DataLoaders and Batching

With our data in nice shape, we'll split it into training, validation, and test sets.
 
* `x` indicates reviews while `y` indicates the corresponding labels. 
* `split_frac` as the fraction of data to **keep** in the training set. 
Usually this is set to 0.8 or 0.9. Whatever data is left will be split in half to create the validation and *testing* data.

```python
split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
```

After creating training, test, and validation data, we can create DataLoaders for this data by following two steps:
1. Create a known format for accessing our data, using [TensorDataset](https://pytorch.org/docs/stable/data.html#) which takes in an input set of data and a target set of data with the same first dimension, and creates a dataset.
2. Create DataLoaders and batch our training, validation, and test Tensor datasets.

```
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, batch_size=batch_size)
```

This is an alternative to creating a generator function for batching our data into full batches.

```python
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 50

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
```

```python
# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)
```

### Sentiment Network with An RNN

#### Network Architecture

The architecture for this network is shown below.

<img src="fig/network_diagram.png" width=40%>

>**First, we'll pass in words to an embedding layer.** We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation for our input data than one-hot encoded vectors. 
However, it's good enough to just have an embedding layer and let the network learn a different embedding table on its own. 
*In this case, the embedding layer is for dimensionality reduction, rather than for learning semantic representations.*

>**After input words are passed to an embedding layer, the new embeddings will be passed to LSTM cells.** 
The LSTM cells will add *recurrent* connections to the network and give us the ability to include information about the *sequence* of words in the movie review data. 

>**Finally, the LSTM outputs will go to a sigmoid output layer.** We're using a sigmoid function because positive and negative = 1 and 0, 
respectively, and a sigmoid will output predicted, sentiment values between 0-1. 

We don't care about the sigmoid outputs except for the **very last one**; we can ignore the rest. We'll calculate the loss by comparing the output at the last time step and the training label (pos or neg).


#### The Embedding Layer

We need to add an [embedding layer](https://pytorch.org/docs/stable/nn.html#embedding) because there are 74000+ words in our vocabulary. It is massively inefficient to one-hot encode that many classes. 
So, instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table. 


#### The LSTM Layer(s)

We'll create an [LSTM](https://pytorch.org/docs/stable/nn.html#lstm) to use in our recurrent network, which takes in an input_size, a hidden_dim, a number of layers, and a batch_first parameter.

Most of the time, you're network will have better performance with more layers; between 2-3. Adding more layers allows the network to learn really complex relationships. 
We define a class that inherits PyTorch’s base class(nn.module) for all neural network modules to start building our own LSTM model. 

* `__init__()`: Defining some variables and also the layers for our model under the constructor.  
    **self.embedding = nn.Embedding()**: Defining 1 embedding layer, which converts our word tokens (integers) into embeddings of a specific size.  
    **self.lstm = nn.LSTM()**: Defining 1 LSTM layer, which is defined by a hidden_state size and number of layers.  
    **self.fc = nn.Linear()**: A fully-connected output layer that maps the LSTM layer outputs to a desired output_size.  
    **self.sig = nn.Sigmoid()**: A sigmoid activation layer which turns all outputs into a value 0-1; return **only the last sigmoid output** as the output of this network.  
      

* `forward()`: Defining the forward pass function. It is executed sequentially.
    

* `init_hidden()`:Initializing the hidden state and cell state of LSTM. This basically creates a tensor of zeros in the shape of our hidden states and cell state of LSTM.

```python
# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
```

```python

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            batch_first=True)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out[:, -1, :] # getting the last time step output
        
        # fully-connected layer
        out = self.fc(lstm_out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        
```


### Instantiate the network

Here, we'll instantiate the network. First up, defining the hyperparameters.

* `vocab_size`: Size of our vocabulary or the range of values for our input, word tokens.
* `output_size`: Size of our desired output; the number of class scores we want to output (pos/neg).
* `embedding_dim`: Number of columns in the embedding lookup table; size of our embeddings.
* `hidden_dim`: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
* `n_layers`: Number of LSTM layers in the network. Typically between 1-3

```python
# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 1

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
```

### Training

Below is the typical training code. If you want to do this yourself, feel free to delete all this code and implement it yourself. 
You can also add code to save a model by name. It should be noted that we include gradient clipping in the below training process by using the `nn.utils.clip_grad_norm_()`. 
This shows how we can adapt the same training commands we introduced in previous tutorials to contain more complicated concepts. 

* `criterion = nn.BCELoss()`: a new kind of cross entropy loss, which is designed to work with a single Sigmoid output. 
* [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1.

We also have some data and training hyparameters:

* `lr`: Learning rate for our optimizer.
* `epochs`: Number of times to iterate through the training dataset.
* `clip`: The maximum gradient value to clip at (to prevent exploding gradients).

```python
# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
```

```python
# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):

    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        
        # initialize hidden state
        h = net.init_hidden(inputs.size(0))
        

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            
            # Get validation loss
            
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = net.init_hidden(inputs.size(0))

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
```

### Testing

There are a few ways to test your network.

* **Test data performance:** First, we'll see how our trained model performs on all of our defined test_data, above. 
* We'll calculate the average loss and accuracy over the test data.

* **Inference on user-generated data:** Second, 
* we'll see if we can input just one example review at a time (without a label), and see what the trained model predicts. 
* Looking at new, user input data like this, and predicting an output label, is called **inference**.

```python
# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0



net.eval()
# iterate over test data

for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    # init hidden state
    h = net.init_hidden(inputs.size(0))
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
```

### Inference on a test review

You can change this test_review to any text that you want. Read it and think: is it pos or neg? Then see if your model predicts correctly!
 
* `tokenize_review()`: Encoding the test words, which is same with the above.

```python
# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
```

```python

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words])

    return test_ints

# test code and generate tokenized review
test_ints = tokenize_review(test_review_neg)
print(test_ints)
```

test sequence padding, just like the above.

```python
# test sequence padding
seq_length=200
features = np.zeros((len(test_ints), seq_length), dtype=int)

#For reviews shorter than seq_length words, left pad with 0s. For reviews longer than seq_length, use only the first seq_length words as the feature vector.
for i, row in enumerate(test_ints):
    features[i, -len(row):] = np.array(row)[:seq_length]

print(features)
```

```python
#test conversion to tensor and pass into your model
feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())
```

```python
def predict(net, test_review, sequence_length=200):
    
    net.eval()
    
    # tokenize review
    test_ints = tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length=sequence_length
    
    features = np.zeros((len(test_ints), seq_length), dtype=int)
    # For reviews shorter than seq_length words, left pad with 0s. For reviews longer than seq_length, use only the first seq_length words as the feature vector.
    for i, row in enumerate(test_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    
    # get the output from the model
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
        
```

```python
# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'
```

```python
# call function
seq_length=200 # good to use the length that was trained on

predict(net, test_review_pos, seq_length)
```

Now that you have a trained model and a predict function, you can pass in any kind of text and this model will predict whether the text has a positive or negative sentiment. 
Push this model to its limits and try to find what words it associates with positive or negative.
