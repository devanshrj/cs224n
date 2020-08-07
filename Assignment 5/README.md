## Assignment 5

<!-- ### Section 1
(a) Convolutional architectures can only work on fixed inputs as they produce fixed outputs. The fully connected layers in a CNN can only work with weights and inputs of fixed size.
<br>
(b) The smallest possible size that m_word can have is 1. However, since we are adding a start and end token to each word, the smallest size is 3. In order for a window of kernel size 5 to exist, the padding required will be of size 1.
Thus, padding_size = 1. (1 on each side)
<br>
(c) 
<br>
(d)
 -->

## Model Performance
Local Test -> Corpus BLEU: 83.1101499133603
<br>

## Learning Outcomes
- Read docs completely!!
- Pytorch
```python
# cuda
device = torch.device("cuda:0")
model = model.to(device)
# or
model = model.cuda()

# required to call nn.Module's init
super(Highway, self).__init__()

# to manipulate shape of tensor
reshape(), view(), transpose(), permute()
```
