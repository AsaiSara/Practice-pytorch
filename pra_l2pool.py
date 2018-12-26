import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# hyper_parameter
n_word = 10
hidden_size = 2
num_layers = 1
uttr_len = 3
batch_size = 4
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

gru = nn.GRU(hidden_size, hidden_size, 2, bidirectional = True, batch_first=True)
output, hidden = gru(embed_sr1_pad)
print('**** output of gru(2,2,num_layer = 2, bidirectional = True')
print(output)
print('**** hidden')
print(hidden)
output, _ = pad_packed_sequence(output, batch_first=True)
print('**** pad packed output')
print(output)

#l2_pooling
output_bi = output.view(batch_size, uttr_len, 2, -1)
print('**** forward_output_size = batch, uttr_len, hidden')
output_for = output_bi[:,:,0]
output_back = output_bi[:,:,1]

print('**** try l2pooling about batch_1, forward')
print('*** forward_output')
print(output_for)
output_for_1 = torch.sum(torch.pow(output_for[0][:sr1_len[0]],2), dim=0)
output_for_0 = torch.pow(output_for[0][:sr1_len[0]],2)
print('** output_l2step0: per uttr ** 2 \n', output_for_0)
print('** output_l2step1: sum of all uttr \n', output_for_1)
output_for_2 = output_for_1/sr1_len[0].type(torch.FloatTensor)
print('** output_l2step2: average of all uttr ** 2\n',output_for_2)
output_for_3 = torch.sqrt(output_for_2).type(torch.FloatTensor)
print('** output_l2step3: sqrt average\n', output_for_3)

def l2_pooling(hiddens, src_len):
    return torch.stack(
        [torch.sqrt(torch.sum(torch.pow(hiddens[b][:src_len[b]],2),dim=0)/src_len[b].type(torch.FloatTensor)) for b in range(hiddens.size(0))])

forward = l2_pooling(output_for, sr1_len)
backward = l2_pooling(output_back, sr1_len)
print('** output_l2step3 backward\n', backward)
output_l2_bi = torch.cat((forward, backward), dim=1)
print('*** concat for and back\n', output_l2_bi)
 
hidden_state = (output_l2_bi[:,:hidden_size] + output_l2_bi[:,hidden_size:])
print('**** forward + backward\n', hidden_state)


