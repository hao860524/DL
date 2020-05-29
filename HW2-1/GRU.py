from torch import nn


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    def forward(self, seq_in):
        # x shape (batch_size, time_step(sequence length), input_size)
        gru_out, _ = self.gru(seq_in)
        # output shape (batch_size, time_step, hidden_size)
        ht = gru_out[:, -1, :]
        output = self.fc(ht)
        # output = self.fc(self.sigmoid(ht))
        # final output (batch_size, output_dim)
        # return self.sigmoid(output)
        return output