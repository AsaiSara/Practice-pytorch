import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# hyper_parameter
n_word = 10
hidden_size = 2
num_layers = 1

# modules
embedding = nn.Embedding(n_word, hidden_size,padding_idx = 0)
gru = nn.GRU(hidden_size, hidden_size, num_layers)

# EmotionEncDec
sr1 = torch.tensor([[1, 2, 0],[2, 3, 8],[4, 0, 0],[5, 4, 0]])
print('**** source tensor (batch_size = 4, max_length = 3)')
print(sr1)
sr1_len = torch.tensor([2,3,1,2])
print('**** sr1_len : ',sr1_len)
sr1_len, sr1_index = sr1_len.sort(0,descending=True)
sr1 = sr1[sr1_index]
print('**** sorted_len : ',sr1_len)
print('**** sorted_index : ',sr1_index)
print('**** sorted_source : ', sr1)
embed_sr1 = embedding(sr1)
print('**** embedded source : ', embed_sr1)

# EncoderRNN
embed_sr1_pad = pack_padded_sequence(embed_sr1,sr1_len, batch_first=True)
print('**** pack padded source')
print(embed_sr1_pad)
output, hidden = gru(embed_sr1_pad)
print('**** output of gru (2,2,num_layer = 1)')
print(output)
print('**** hidden')
print(hidden)
output, output_len = pad_packed_sequence(output, batch_first=True)
print('**** pad packed output')
print(output)
print('**** output_len')
print(output_len)

gru = nn.GRU(hidden_size,hidden_size,2, batch_first=True)
output, hidden = gru(embed_sr1_pad)
print('**** output of gru (2,2,num_layer = 2)')
print(output)
print('**** hidden')
print(hidden)
output = pad_packed_sequence(output, batch_first=True)
print('**** pad packed output')
print(output)

gru = nn.GRU(hidden_size, hidden_size, 1, bidirectional = True, batch_first=True)
output, hidden = gru(embed_sr1_pad)
print('**** output of gru(2,2,num_layer = 1, bidirectional = True')
print(output)
print('**** hidden')
print(hidden)
output = pad_packed_sequence(output, batch_first=True)
print('**** pad packed output')
print(output)


gru = nn.GRU(hidden_size, hidden_size, 2, bidirectional = True, batch_first=True)
output, hidden = gru(embed_sr1_pad)
print('**** output of gru(2,2,num_layer = 2, bidirectional = True')
print(output)
print('**** hidden')
print(hidden)
output = pad_packed_sequence(output, batch_first=True)
print('**** pad packed output')
print(output)







