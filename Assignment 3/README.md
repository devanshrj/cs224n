# Assignment 3: Dependency Parsing

## Part 1

A)
i) Momentum prevents oscillations (caused by variance due to mini batch gradient descent) by considering past gradients to **smooth** out the update. <br>
ii) The update by v ^ 1/2 is used for normalisation. It provides larger updates to the weight parameters as compared to bias parameters.
As a result, weight parameters are updated faster as compared to bias parameters, leading to faster learning and convergence.
[Explanation](https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop) <br>

B)
i) gamma = 1/(1-p) <br>
ii) Dropout is used for regularisation to “spread” out weights so that they have similar influence on the output of that layer.
During evaluation, dropout can lead to noisy predictions due to randomness, so it is not ideal.

## Part 2

B) Steps required for parsing a sentence containing N words = 2N <br>
The SHIFT transition takes N steps for shifting N words from the buffer to the stack. <br>
Establishing (head,dependent) relations between the N words takes N steps for N words. <br>

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
