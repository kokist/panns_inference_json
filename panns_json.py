# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
import json
from panns_inference import AudioTagging, SoundEventDetection, labels


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


def plot_sound_event_detection_result(framewise_output, file_name):
    """Visualization of sound event detection result. 
    Args:
      framewise_output: (time_steps, classes_num)
                                 701,527
      frame-wise = フレーム単位
    """
    # out_fig_path = 'results/sed_result.png'
    # os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)
    
    classwise_output = np.max(framewise_output, axis=0) # (classes_num,) 列の最大値を取得
    idxes = np.argsort(classwise_output)[::-1] #ラベルを降順にソート
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    
    score_keys = ['name', 'score0', 'score1', 'score2', 'score3', 'score4']
    label_keys = ['label0', 'label1', 'label2', 'label3', 'label4']
    label_name = []
    prob = []

    for idx in idxes:
        label_name.append(ix_to_lb[idx])
        prob.append(framewise_output[:, idx])
        
    json_value = []
    json_value.append(file_name) #動画名を追加
    for i in range(len(label_name)):
        label_prob = dict([('label', label_name[i]), ('prob', prob[i])])
        json_value.append(label_prob)

    # score_keyとlabel，probをくっつける
    data = dict(zip(score_keys, json_value))
    
    # jsonを保存
    with open('./output/'+file_name +'.json', 'w') as f:
        json.dump(data, f, indent=4, cls = MyEncoder, ensure_ascii=False)

class MyEncoder(json.JSONEncoder): #numpy配列をjsonにするために必要
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    device = 'cuda' # 'cuda' | 'cpu'
    path = '/content/folder/*.*'
    file_list = glob.glob(path) # path付きのファイル名を取得
    for audio_path in file_list:
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        print('------ Audio tagging ------') #タグ付けのために推論
        at = AudioTagging(checkpoint_path=None, device=device)
        (clipwise_output, embedding) = at.inference(audio)
        """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

        print_audio_tagging_result(clipwise_output[0])

        print('------ Sound event detection ------')#音を推論
        sed = SoundEventDetection(checkpoint_path=None, device=device)
        framewise_output = sed.inference(audio)
        """(batch_size, time_steps, classes_num)"""
        file_name = os.path.splitext(os.path.basename(audio_path))[0] #拡張子なしのファイル名
        print(f'{file_name}')
        plot_sound_event_detection_result(framewise_output[0], file_name) #print(len(framewise_output[0])) 総フレーム数701