# Assignment 3: Dependency Parsing

## Part 2

B) Steps required for parsing a sentence containing N words = 2N <br>
The SHIFT transition takes N steps for shifting N words from the buffer to the stack. <br>
Establishing (head,dependent) relations between the N words takes N steps for N words. <br>

F) To - do

## Model Performance

After training: <br>
Best Dev UAS: 88.79 <br>
Test UAS: 89.02 <br>

## Learning Outcomes

- Extracting embeddings from embeddings matrix using indices.
- Adam Optimizer (momentum and adaptive learning rate)
- tqdm
- PyTorch

```python3
# commands for training and testing
model.train()       # training - apply dropout layer
model.eval()        # testing - doesn't apply dropout layer
```
