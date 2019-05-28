# Time Gated LSTM for irregular time series

This repository contain a PyTorch implementation of a variant of Vanilla LSTM in order to take into account a irregular time between time samples. The new LSTM structure (Time Gated LSTM) is based on the paper [Nonuniformly Sampled Data Processing Using LSTM Networks](https://ieeexplore.ieee.org/document/8478179/)  by *Safa Onur Sahin* and *Suleyman Serdar Kozat*. 

## Use
This implementation supports more layers and bidirectionality.

    TGLSTM(input_size, hidden_size, num_layers, bias=True,
		   batch_first=False, dropout=False, bidirectional=False)

## Notes
TGLSTM runs on GPUs but the performance are worse than PyTorch LSTM.
For parameters details see [PyTorch Documentation] (https://pytorch.org/docs/stable/nn.html#LSTM)
