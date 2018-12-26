import torch
import torch.nn as nn
from collections import Counter
from pra_atten_1214 import AttnDec
import torch.nn.functional as F

def inf_uttr(decoder, encoder_output, encoder_hidden, emotion):
    n_words = 3
    EOS_id = 2
    batch_size = 1
    beam_width = 2
    num_layers = 2
    len_alpha = 0.6
    eos_gamma = 0.05
    suppress_lmd = 1.0
    MAX_UTTR_LEN = 5
    
    # hiddens ... [(batch=1, hidden) * num_layers] 
    hiddens = [encoder_hidden for _ in range(num_layers)]
    
    decoder_input = torch.tensor([[1]]) #<s>
    # decoder_output ... batch=1, n_words
    decoder_output, hiddens = decoder(decoder_input, hiddens, encoder_output, emotion)
    # topv ... batch=1, beam_width
    topv, topi = F.log_softmax(decoder_output, dim=1).topk(beam_width)

    # adapt to beam_width
    # encoder_output ... batch=1, seq_len, hidden_size -> beam_width, seq_len, hidden_size
    encoder_output = encoder_output.expand(beam_width, encoder_output.size(1), encoder_output.size(2))
    # decoder_output ... beam_width, seq_len, n_words
    decoder_output = decoder_output.expand(beam_width, -1)
    # emotion ... beam_width, 1
    emotion = emotion.expand(beam_width, -1)

    # beam width 1d tensor
    # topv ... beam_width
    topv = topv.flatten()
    # decoder_input ... beam_width, batch_size=1
    decoder_input = topi.t()
    # hiddens ... [(beam_width, hidden_size) * num_layers]
    hiddens = [hidden.expand(beam_width, hidden_size) for hidden in hiddens]
    #decoder_hiddens = decoder_hiddens.expand(num_layers, beam_width, hidden_size).contiguous()
    # inf_uttrs = [[id] * beam_width]
    inf_uttrs = [[id.item()] for id in decoder_input]
    # repet_counts ... beam_width, n_words 
    repet_counts = torch.Tensor([1]).expand(beam_width, n_words)
    
    # beam search
    for _ in range(MAX_UTTR_LEN-1):
        repet_counts = torch.tensor([[repet_counts[b][w]+1 if inf_uttrs[b][-1] == w else repet_counts[b][w] for w in range(n_words)] for b in range(beam_width)])
        eos_idx = [idx for idx, words in enumerate(inf_uttrs) if words[-1] == EOS_id]
        prev_output, prev_hiddens = decoder_output, hiddens
        #import pdb; pdb.set_trace()
        decoder_output, hiddens = decoder(
            decoder_input, hiddens, encoder_output, emotion
            )

        # suppression of repetitive generation
        # suppressor ... beam_width, n_words
        suppressor = torch.Tensor([1]).expand(beam_width, n_words) / repet_counts.pow(suppress_lmd)
        decoder_output = topv.unsqueeze(1) + F.log_softmax(decoder_output * suppressor, dim=1)
        # Don't update ouptut, hiddens if the last word is <s>
        if len(eos_idx) > 0:
            decoder_output[eos_idx] = float('-inf')
            decoder_output[eos_idx, EOS_id] = prev_output[eos_idx,EOS_id]
        hiddens = [torch.stack([prev_hiddens[l][b] if b in eos_idx else hiddens[l][b]
                               for b in range(beam_width)])
                   for l in range(num_layers)]
        lp = torch.tensor([(5+len(uttr)+1)**len_alpha / (5+1)**len_alpha for uttr in inf_uttrs])
        normalized_output = decoder_output / lp.unsqueeze(1)
        normalized_output[:,EOS_id] -= eos_gamma * (MAX_UTTR_LEN / torch.tensor([len(uttr) for uttr in inf_uttrs]).float())

        topv, topi = normalized_output.topk(beam_width)
        topv, topi = topv.flatten(), topi.flatten()
        topv, perm_index = topv.sort(0, descending=True)

        topv = topv[:beam_width]
        decoder_input = topi[perm_index[:beam_width]].view(-1,1)
        former_index = perm_index[:beam_width] // beam_width

        decoder_output = decoder_output[former_index]
        hiddens = [hiddens[l][former_index] for l in range(num_layers)]
        inf_uttrs = [inf_uttrs[former] + [decoder_input[i].item()]
                    if inf_uttrs[former][-1] != EOS_id
                        else inf_uttrs[former]
                    for i, former in enumerate(former_index)]
        repet_counts = repet_counts[former_index]

        # If all last words are </s>, break
        if sum([words[-1] == EOS_id for words in inf_uttrs]) == beam_width:
            break

    return inf_uttrs, topv
    import pdb; pdb.set_trace()

batch_size = 1
num_layer = 2
hidden_size = 2
encoder_seq_len = 3
n_words = 3
# batch=1, seq_len, hidden
enc_output = torch.tensor([[[1,2],[1,2],[1,2]]]).float()
# batch=1, 1
emotion = torch.tensor([[1]])
# batch=1, hidden
enc_hidden = torch.tensor([[0.3,0.2]])
embedding = nn.Embedding(n_words,hidden_size,padding_idx = 0)
emotion_embedding = nn.Embedding(5,5,padding_idx = 0)
decoder = AttnDec(embedding,emotion_embedding,hidden_size,n_words)
output, topv = inf_uttr(decoder, enc_output, enc_hidden, emotion)
print(output)
print(topv)
