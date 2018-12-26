import torch

use_em = True

dial_f= ["1\t2","3\t4"]
emotion_f = ["1\t0","3\t4"]
data = [] 

def read_dialogues(self,dial,emotion):
    dials = [[uttr for uttr in dial.strip().split('\t')]
             for dial in dial_f]
    src_dial = [[u for u in d[-2::-2][::-1]] for d in dials]
    tgt_uttr = [[u for u in d[-1::-2][::-1]] if len(d) % 2 == 0 else
                [u for u in d[-1::-1][1::-1]] for d in dials]
    [data.append({"srs": src, "tgt": tgt}) for src, tgt in zip(src_dial, tgt_uttr)]

    if use_em:
        emotions = [[uttr for uttr in dial.strip().split('\t')]
                    for dial in emotion_f]
        src_emotions = [[int(u) for u in d[-2::-2][::-1]] for d in emotions]
        tgt_emotions = [[int(u) for u in d[-1::-2][::-1]] for d in emotions]
    else:
        src_emotions = [[0 for _ in dial] for dial in src_dial]
        tgt_emotions = [[0 for _ in dial] for dial in tgt_dial] 
    [data[i].update({"src_emotion":src,"tgt_emotion":tgt}) 
     for i,(src,tgt) in enumerate(zip(src_emotions, tgt_emotions))]


    # data ... [{"src":[ ],"tgt":[ ], "src_emotion":[],"tgt_emotion":[]} * dial_len]
    print(data)
    return data




