import torch
import torchaudio
import numpy as np
from math import ceil
import os
import sys
import time
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained(
    "davidrrobinson/BioLingual", sampling_rate=48000
)

def get_embedding10s(file, processor, model, device="cpu"):
    waveform, sample_rate = torchaudio.load(file)
    # Duration of each segment in seconds
    segment_duration = 10  
    num_samples_per_segment = segment_duration * sample_rate  # Samples per 10 sec
    # Get total number of samples and number of segments
    num_samples = waveform.shape[1]
    nb_segments = ceil(num_samples/num_samples_per_segment)
    # Loop through and extract 10-second chunks
    segment_embed = np.empty([nb_segments, 512])
    i = 0
    for start in range(0, num_samples, num_samples_per_segment):
        end = min(start + num_samples_per_segment, num_samples)  # Ensure it does not exceed total samples
        segment = waveform[:, start:end]  # Slice the waveform tensor
        inputs = processor(
            audios=segment[0], return_tensors="pt", sampling_rate=sample_rate, padding=True
        ).to(device)
        audio_embed = model.get_audio_features(**inputs)
        segment_embed[i,:] = audio_embed.detach().numpy()
        i = i+1
    return(segment_embed)

# Must be files of 1 minute
def get_multiple_embedding1min(files, processor, model, device="cpu"):
    embeddings = np.array(
        [get_embedding10s(file, processor, model, device=device) for file in tqdm(files)]
    )
    return embeddings


#############
#############
file = "birdsong.wav"
embd = get_embedding10s(file, processor, model, device)


#############
#############
files = ["ARM3_20220704_070000.wav","VAL1_20220626_151500.wav","VTN6_20220618_110000.wav"]
embds = get_multiple_embedding1min(files, processor, model, device)

csvfile = open("example.csv", "w")
line = ["filename;", "segment;"] + [f"Feat{i+1};" for i in range(511)] + ["Feat512\n"]
csvfile.writelines(line)
for ifile in range(embds.shape[0]):
    for jsegment in range(embds.shape[1]):
        line = [files[ifile]+";",str(jsegment+1)+";"]
        for kfeat in range(embds.shape[2]-1):
            line = line + [str(np.around(embds[ifile,jsegment,kfeat],4))+";"]
        line = line + [str(np.around(embds[ifile,jsegment,-1],4))+"\n"]
        csvfile.writelines(line)
        
csvfile.close()


#############
## Deprecated
import glob
import pandas as pd
files = glob.glob("PATH_TO_WAV_FILES/*wav")
embeddings = get_multiple_embedding(files, processor, model, device)
df = pd.DataFrame(embeddings, columns=[f"Feat{i+1}" for i in range(embeddings.shape[1])])
df.insert(0, "Filename", files)
df.to_csv("embeddings.csv", index=False)  # `index=False` prevents adding an extra index column

