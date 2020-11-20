import soundfile
import librosa
import numpy as np
import pickle

def getfeature(file_name):
    """
    input: le fichier audio
    output: feautures 
    """
    
    result = np.array([])
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13, ).T, axis=0)
    result = np.hstack((result, mfccs))      
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result = np.hstack((result, chroma))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0)
    
    return  mfccs, chroma, zcr, result

def parse_audio_files(*musicspaths):
    """
    
    """
    features, labels = np.empty((0, 26)), np.empty(0)
    # labels = []
    for filename in musics:
        try:
            mfccs, chroma, zcr = getfeature(filename)
        except Exception as e:
            print("Probleme avec le fichier: ", filename)
            continue
        ext_features = np.hstack([mfccs, chroma, zcr])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, [filename][0].split("/")[3].split(".")[0])

    return np.array(features), np.array(labels)


genres_musics = {'blues',
                'classical',
                'country',
                'disco',
                'hiphop',
                'jazz',
                'metal',
                'pop',
                'reggae',
                'rock'}


def get_label(audio_config):
    """Docs
    
    """
    features = ["mfcc", "zcr"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")



def get_first_letters(genres):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    Extraction des features à partir des fichiers audios `file_name`
        Features supported:
            - MFCC (mfcc)
            - zcr
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    # chroma = kwargs.get("chroma")
    zcr = kwargs.get("zcr")

    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        # if mel:
        #     mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        #     result = np.hstack((result, mel))
        # if contrast:
        #     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        #     result = np.hstack((result, contrast))
        # if tonnetz:
        #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        #     result = np.hstack((result, tonnetz))
    return result



def get_audio_config(features_list):
    """
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config
    