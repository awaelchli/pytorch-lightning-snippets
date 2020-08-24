
# Model Verification

## Normalization Layers and Biases

It is not a good idea to combine network layers with biases (e.g. Conv) and normalization layers (e.g. BatchNorm). 
Here is a tool that let's you check that this does not happen while you develop your networks.

```python 
from verification.batch_norm import BatchNormVerification

model = YourPyTorchModel()
verification = BatchNormVerification(model)
valid = verification.check(input_array=torch.rand(2, 3, 4))
```

This will run a forward pass using the provided example input to determine in which order the submodules get executed. 
It will create a list of all layers with bias term directly followed by a normalization layer such as BatchNorm, InstanceNorm, GroupNorm, etc.

There is also a Callback version of this check for PyTorch Lightning:

```python 
from pytorch_lightning import Trainer
from verification.batch_norm import BatchNormVerificationCallback

model = YourLightningModule()
verification = BatchNormVerificationCallback()
trainer = Trainer(callbacks=[verification])
trainer.fit(model)
```

It will print a warning if it detects a bad combination of bias and normalization in your model:

```bash 
Detected a layer 'model.conv1' with bias followed by a normalization layer 'model.bn1'.
This makes the normalization ineffective and can lead to unstable training.
Either remove the normalization or turn off the bias.
```

## Mixing Data Across the Batch Dimension

Gradient descent over a batch of samples can not only benefit the optimization but also leverages data parallelism.
However, you have to be careful not to mix data across the batch. 
Only a small error in a reshape or permutation operation and your optimization will get stuck and you won't even get an error. 
How can you tell if the model mixes data in the batch? 
A simple trick is to do the following: 
1. run the model on an example batch (can be random data)
2. get the output batch and select the n-th sample (choose n)
3. compute a dummy loss value of only that sample and compute the gradient w.r.t the entire input batch
4. observe that only the i-th sample in the input batch has non-zero gradient

If the gradient is non-zero for the other samples in the batch, it means you are mixing data and you need to fix your model!
Here is a simple tool that does all of that for you:

```python 
from verification.batch_gradient import BatchGradientVerification

model = YourPyTorchModel()
verification = BatchGradientVerification(model)
valid = verification.check(input_array=torch.rand(2, 3, 4), sample_idx=1)
```

In this example we run the test on a batch size 2 by inspecting gradients on the second sample. 
The same is available as a callback for your PyTorch Lightning models:

```python 
from pytorch_lightning import Trainer
from verification.batch_gradient import BatchGradientVerificationCallback

model = YourLightningModule()
verification = BatchGradientVerificationCallback()
trainer = Trainer(callbacks=[verification])
trainer.fit(model)
```

It will warn you if batch data mixing is detected:

```bash 
Your model is mixing data across the batch dimension.
This can lead to wrong gradient updates in the optimizer.
Check the operations that reshape and permute tensor dimensions in your model.
```