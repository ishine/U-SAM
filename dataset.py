
import json
import random
from re import sub
from multiprocessing import Manager

import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(examples):
    feats = [x[0] for x in examples]
    labels = [x[1] for x in examples]
    names = [x[2] for x in examples]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return {"audios": padded_feats, "text": labels, "names": names, "task": task}


def handle_wav(wav_file, target_rate, max_sample_length):
    """
    handle one wav file.
    Return:
        waveform: Tensor(1D)
    """
    waveform, sample_rate = torchaudio.load(wav_file)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)

    waveform = waveform[0]  # just get one channel data
    # if audio length is longer than max_length_sample, we randomly crop it to max length
    if waveform.shape[-1] > max_sample_length:
        max_start = waveform.shape[-1] - max_sample_length
        start = random.randint(0, max_start)
        waveform = waveform[start : start + max_sample_length]
    return waveform


def _text_preprocess(sentence):
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence


class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, max_length=10):
        super().__init__()
        self.lists = []
        with open(data_file, "r", encoding="utf8") as fin:
            for line in fin:
                self.lists.append(line)

        self.all_data = []
        for line in self.lists:
            obj = json.loads(line)
            self.all_data.append(obj)

        ### Try to wrap list using multiprocessing manager to avoid cpu oom
        manager = Manager()
        self.all_data = manager.list(self.all_data)

        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_length_sample = self.max_length * self.sample_rate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        obj = self.all_data[index]
        key = obj["key"]
        wav_file = obj["wav"]
        label = _text_preprocess(obj["label"])
        task = obj["task"]
        waveform = handle_wav(
            wav_file,
            target_rate=self.sample_rate,
            max_sample_length=self.max_length_sample,
        )
        return waveform, label, key, task
