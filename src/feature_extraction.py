import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

def extract_pitch_features(sound):

    # low average pitch
    # monotone => less range

    pitch = call(sound, "To Pitch", 0.0, 75, 500)

    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz")
    min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz")

    return {
    'MDVP:Fo(Hz)': mean_f0,
    'MDVP:Fhi(Hz)': max_f0,
    'MDVP:Flo(Hz)': min_f0
    }

def extract_jitter_features(sound):

    # jitter = cycle variation in pitch
    # high jitter => unstable pitch => parkinson's patients have increased jitter

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(point_process, "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)

    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

    ppq_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    ddp_jitter = rap_jitter*3

    return {
    'MDVP:Jitter(%)': local_jitter,
    'MDVP:Jitter(Abs)': local_abs_jitter,
    'MDVP:RAP': rap_jitter,
    'MDVP:PPQ': ppq_jitter,
    'Jitter:DDP': ddp_jitter
    }

def extract_shimmer_features(sound):

    # shimmer = cycle variation in amplitude (loudness)
    # high shimmer => unstable volume => sign in parkinsons

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    local_shimmer = call([sound,point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    local_shimmer_dB = call([sound,point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    apq3_shimmer = call([sound,point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    apq5_shimmer = call([sound,point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    apq11_shimmer = call([sound,point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    dda_shimmer = apq3_shimmer*3

    return {
    'MDVP:Shimmer': local_shimmer,
    'MDVP:Shimmer(dB)': local_shimmer_dB,
    'Shimmer:APQ3': apq3_shimmer,
    'Shimmer:APQ5': apq5_shimmer,
    'MDVP:APQ': apq11_shimmer,
    'Shimmer:DDA': dda_shimmer
    }


def extract_harmonics_features(sound):

    # hnr = ratio of clear voice to noise
    # nhr = inverse

    # in parkinsons
    # lower hnr, higher nhr

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    hnr = call(harmonicity, "Get mean", 0, 0)

    nhr = 10**(-hnr/10) #cant use 1/hnr directly

    return{
        'HNR': hnr,
        'NHR': nhr
    }

def extract_features(audio, sr):

    try:
        
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        features = {}

        features.update(extract_pitch_features(sound))
        features.update(extract_jitter_features(sound))
        features.update(extract_shimmer_features(sound))
        features.update(extract_harmonics_features(sound))
        
        return features
    
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None
