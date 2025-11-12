import numpy as np
import librosa

def preprocess_audio(record_audio, sample_rate, target_sr = 22050):
    
    # what if someone records through headphones => stereo => left and right channels
    # to prevent them from branching, just combine them if the shape is 2d
    if(len(record_audio.shape) > 1):
        record_audio = librosa.to_mono(record_audio.T)
    
    
    # lets resample to 22050, half of cd quality but reduces computation
    if sample_rate!= target_sr:
        record_audio = librosa.resample(record_audio, orig_sr = sample_rate, target_sr = target_sr)


    # and if there's difference in peaks, like someone talks at low and one talks at high
    # we might mis-categorize/mis-predict either of them
    # so lets normalize this, by keeping the shape same but making the peaks 1

    # to normalize, just find the abs max, divide everything by it and et voila
    if np.max(np.abs(record_audio)) > 0:
        record_audio = record_audio/np.max(np.abs(record_audio))

    # finally removing silence
    # lets remove say anything that is 20-30 db less than peak
    record_audio, index = librosa.effects.trim(record_audio, top_db=30)

    return record_audio, target_sr