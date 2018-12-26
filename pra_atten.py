import torch
import torch.nn.functional as F
import torch.nn as nn

class AttnDec(nn.Module):
    def __init__(self,embedding, emotion_embedding, hidden_size, output_size,num_layers=1, dropout=0.1): 
        super().__init__()
        self.embedding = embedding
        self.emotion_embedding = emotion_embedding
        self.num_layers = num_layers
        self.grus = nn.ModuleList([nn.GRUCell(hidden_size+5, hidden_size)] + [nn.GRUCell(hidden_size,hidden_size) for _ in range(num_layers-1)])
        self.gru = nn.GRU(hidden_size+5, hidden_size+5,num_layers=1,dropout=(0 if num_layers == 1 else dropout), batch_first = True)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size) 
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = dropout
        self.training = True

    def forward(self, input_step, hiddens, encoder_output,emotion,criterion=None):
        # emb_input_seq_size (batch, 1, hidden) ->(batch, hidden)
        emb_input = self.embedding(input_step).squeeze(1)
        # emb_emotion_size (batch, 5)
        emb_emotion = self.emotion_embedding(emotion).squeeze(1)
        # Concatinate embedded_input_step and embedded_emotion
        # embedded_size (batch, 1, hidden+5)
        #import pdb; pdb.set_trace()
        embedded = torch.cat((emb_input, emb_emotion), dim=1)
        embedded = F.dropout(embedded, p=self.dropout)
        # rnn_output_size (batch, hidden)
        # hiddens_size [(batch, hidden) * num_layers]
        rnn_output = embedded
        for i in range(self.num_layers):
            rnn_output = self.grus[i](rnn_output, hiddens[i])
            if i < self.num_layers-1:
                rnn_output = F.dropout(rnn_output, self.dropout)
            hiddens[i] = rnn_output

        print("# encoder_output_size (batch, seq_len, hidden")
        print(encoder_output)
        #import pdb; pdb.set_trace()
        print("# score_size (batch, seq_len)") 
        score = torch.sum(rnn_output.unsqueeze(1) * encoder_output, dim=2)
        print(score)
        #import pdb; pdb.set_trace()
        print("# attention weights per uttrances size (batch,1, seq_len)")
        at = F.softmax(score, dim=1).unsqueeze(1)
        print(at)
        print("# context_vector_size (batch,hidden) ")
        ct = torch.sum(at*encoder_output.transpose(1,2),dim=2)
        concat_input = torch.cat((ct, rnn_output),1)
        print(ct)
        attn_output = torch.tanh(self.attn_combine(concat_input))
        #import pdb; pdb.set_trace()
        print("# output \n", self.out(attn_output))
        return self.out(attn_output), hiddens

batch_size = 2
seq_len = 3
hidden_size = 2
n_word = 3 
num_layers = 2

input_seq = torch.tensor([[1],[0]])
enc_output = torch.tensor([[[1,2],[1,2],[1,2]],[[2,3],[2,3],[0,0]]]).type(torch.FloatTensor)
emotion = torch.tensor([[1],[1]])


embedding = torch.nn.Embedding(n_word, hidden_size, padding_idx=0)
emotion_embedding = torch.nn.Embedding(5,5,padding_idx=0)
attn = AttnDec(embedding, emotion_embedding, hidden_size, n_word, num_layers=2)
# decoder_hidden ... batch_size, hidden_size
decoder_hidden = torch.tensor([[0.3,0.2],[1.7,0.5]])
# decoder_hidden ... [(batch_size, hidden_size) * num_layers]
decoder_hidden = [decoder_hidden for _ in range(num_layers)]
output, hidden = attn(input_seq,decoder_hidden, enc_output, emotion)
print(output)

