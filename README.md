# Time Gated LSTM for irregular time series

This repository contain a PyTorch implementation of a variant of Vanilla LSTM in order to take into account a irregular time between time samples. The new LSTM structure (Time Gated LSTM) is based on the paper [Nonuniformly Sampled Data Processing Using LSTM Networks](https://ieeexplore.ieee.org/document/8478179/)  by *Safa Onur Sahin* and *Suleyman Serdar Kozat*. 

## Use
This implementation supports more layers and bidirectionality.

	class TimeGatedLSTM(nn.Module):
	    def __init__(self, in_size=7, h_size=800, n_layers=7,
			 out_size=4):
		super(TimeGatedLSTM, self).__init__()

		self.in_size = in_size
		self.h_size = h_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.tglstm = TGLSTM(input_size, hidden_size, num_layers, bias=True,
		   			batch_first=False, dropout=False, bidirectional=False)
		self.fc = nn.Linear(in_features=self.h_size,
				    out_features=self.out_size)

	    def forward(self, X, time):
		# X.shape: [batch_size, seq_len, features]
		# time.shape: [batch_size, seq_len, features]
		# swap axis to get batch_first=False
		X = X.permute(1, 0, 2)
		time = time.permute(1, 0, 2)
		output_rnn, _ = self.tglstm(inp, time)
		fc_output = self.fc(output_rnn.permute(1, 0, 2))
		# fc_output will be batch_size*seq_len*num_classes
		return fc_output

## Notes
TGLSTM runs on GPUs but the performance are worse than PyTorch LSTM.
For parameters details see [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html#LSTM)
