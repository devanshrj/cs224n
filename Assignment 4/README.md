# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

## Model Evaluation
Epoch = 14
Iterations = 90000
Cum. loss = 25.61
Cum ppl = 3.47 
Cum. examples = 63977
**Validation**: iter 90000, dev. ppl 7.184368
<br>
### Corpus BLEU: 35.752705837873755

## Learning Outcomes
- PyTorch docs are beautiful
```python3
# simple functions can be called using the syntax <input>.<function>() instead of torch.<function>(<input>)

# to split a tensor
torch.split()

# add and remove singleton dimensions
torch.unsqueeze()
torch.squeeze()

# concatenation of tensors
torch.cat()

# batch matrix multiplication
# (b, n, m) * (b, m, p) -> (b, n, p)
torch.bmm(mat1, mat2)

# simple layers can be applied directly instead of explicitly defining layer
layer = nn.Softmax()
output = layer(input)
# can be replaced by
output = nn.Softmax()(input)

# alternately, a lot of such layers have nn.functional functions defined and those can be used as well
output = F.softmax(input) 