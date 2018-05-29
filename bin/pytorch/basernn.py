import torch
import torch.nn as nn

"""
Based upon this implementation
https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb
"""


class baseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1):
        # Call superclass constructor, assign fields
        super(baseRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Assign layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))  # flatten/reshape
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.Tensor(torch.zeros(self.n_layers, 1, self.hidden_size))

# TODO
# except KeyboardInterrupt:
#     print("Saving before quit...")
# save()