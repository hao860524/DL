from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    def forward(self, seq_in):
        # x shape (batch_size, time_step(sequence length), input_size)
        lstm_out, _ = self.lstm(seq_in)
        # output shape (batch_size, time_step, hidden_dim)
        ht = lstm_out[:, -1, :]
        output = self.fc(ht)
        # output = self.fc(self.sigmoid(ht))
        # final output (batch_size, output_dim)
        # return self.sigmoid(output)
        return output