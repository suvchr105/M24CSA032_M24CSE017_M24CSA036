import numpy as np
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
import random
import av
import torchaudio
from transformers import DistilBertTokenizer

# -------------------------------------------------------------------------
# Task definitions for Split-MUSIC-AVQA
# -------------------------------------------------------------------------
ALL_TASKS = ['Counting', 'Existential', 'Location', 'Comparative', 'Temporal']
MAX_QST_LEN = 30 
NUM_FRAMES = 8 

def collate_fn_novelty(batch):
    audios, visuals, questions, attn_masks, answers, que_ids, labels = zip(*batch)
    audios = torch.stack(audios)
    visuals = torch.stack(visuals) 
    questions = torch.stack(questions)
    attn_masks = torch.stack(attn_masks)
    answers = torch.stack(answers)
    que_ids = torch.tensor(que_ids, dtype=torch.long)
    labels = torch.stack(labels)
    return audios, visuals, questions, attn_masks, answers, que_ids, labels


class NoveltyLoader(Dataset):
    def __init__(self, args, mode='train', modality='audio-visual',
                 incremental_task=0, incremental_step=0):
        self.mode = mode
        self.args = args
        self.modality = modality
        self.incremental_task = incremental_task
        self.incremental_step = incremental_step

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.label_to_ix = {}
        self.all_ans_type = []
        
        # Paths
        self.video_dir = getattr(args, 'video_dir', '/mnt/raid/obed/Speech/MUSIC_AVQA_videos/MUSIC-AVQA-videos-Real')
        self.json_base_dir = '../../../data/split_music-avqa/json/'

        self.all_task = ALL_TASKS
        self.all_current_data_vids = []
        self.num_current_step_ans = 0
        self.step_sample_counts = []

    def _normalize_sample(self, sample):
        out = dict(sample)
        if 'question_content' in out and 'question_text' not in out:
            out['question_text'] = out['question_content']
        if 'answer' not in out and 'anser' in out:
            out['answer'] = out['anser']
        return out

    def _load_split_data(self, split, task_name):
        json_path = os.path.join(self.json_base_dir, f'{split}_{task_name}.json')
        if not os.path.exists(json_path):
            return []
        data = json.load(open(json_path, 'r'))
        return [self._normalize_sample(s) for s in data]

    def _get_step_data(self, all_data, step, total_steps=6):
        sorted_data = sorted(all_data, key=lambda s: s['answer'])
        n = len(sorted_data)
        chunk_size = n // total_steps
        start = step * chunk_size
        end = start + chunk_size if step < total_steps - 1 else n
        return sorted_data[start:end]

    def current_step_data(self):
        task_name = self.all_task[self.incremental_task]
        split = self.mode if self.mode in ('train', 'val', 'test') else 'train'
        all_data = self._load_split_data(split, task_name)
        step_data = self._get_step_data(all_data, self.incremental_step)
        
        for s in step_data:
            s['label_str'] = f"{task_name}_{s['answer']}"
        
        if self.mode == 'train':
            self.all_current_data_vids = step_data
        else:
            self.all_current_data_vids.extend(step_data)
        
        return self.all_current_data_vids

    def update_vocabs(self, task, step):
        os.makedirs('./encoder_novelty', exist_ok=True)
        vocab_path = f'./encoder_novelty/all_ans_type.json'
        label_path = f'./encoder_novelty/label_to_ix.json'

        if self.mode == 'train':
            # Load existing if any
            if os.path.exists(vocab_path):
                self.all_ans_type = json.load(open(vocab_path))
            if os.path.exists(label_path):
                self.label_to_ix = json.load(open(label_path))

            for sample in self.all_current_data_vids:
                ans = sample['answer']
                if ans not in self.all_ans_type:
                    self.all_ans_type.append(ans)
                lbl = sample['label_str']
                if lbl not in self.label_to_ix:
                    self.label_to_ix[lbl] = len(self.label_to_ix)
            
            with open(vocab_path, 'w') as f:
                json.dump(self.all_ans_type, f)
            with open(label_path, 'w') as f:
                json.dump(self.label_to_ix, f)
        else:
            if os.path.exists(vocab_path):
                self.all_ans_type = json.load(open(vocab_path))
            if os.path.exists(label_path):
                self.label_to_ix = json.load(open(label_path))
            
            # For validation mode, we still might encounter new labels in some setups, but here we expect consistency
            for sample in self.all_current_data_vids:
                lbl = sample['label_str']
                if lbl not in self.label_to_ix:
                    self.label_to_ix[lbl] = len(self.label_to_ix)

        self.num_current_step_ans = len(self.all_ans_type)
        return len(self.all_ans_type)

    def set_incremental_step(self, task, step):
        self.incremental_task = task
        self.incremental_step = step
        self.all_current_data_vids = self.current_step_data()
        self.update_vocabs(task, step)
        self.step_sample_counts.append(len(self.all_current_data_vids))

    def _extract_audio(self, video_path):
        try:
            container = av.open(video_path)
            if not container.streams.audio:
                return torch.zeros(1024, 128)
            
            chunks = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim > 1: arr = arr.mean(axis=0)
                chunks.append(torch.from_numpy(arr.astype('float32')))
            container.close()
            
            if not chunks: return torch.zeros(1024, 128)
            waveform = torch.cat(chunks).unsqueeze(0)
            
            resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
            waveform = resampler(waveform)
            
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128, n_fft=400, hop_length=160
            )(waveform)
            
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)
            if mel_spec.size(0) > 1024:
                mel_spec = mel_spec[:1024, :]
            else:
                pad = 1024 - mel_spec.size(0)
                mel_spec = torch.cat([mel_spec, torch.zeros(pad, 128)], dim=0)
            return mel_spec
        except Exception:
            return torch.zeros(1024, 128)

    def _extract_video_frames(self, video_path):
        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames
            if total_frames <= 0: total_frames = 150 # Fallback
            
            indices = np.linspace(0, max(0, total_frames - 1), NUM_FRAMES, dtype=int)
            f_idx = 0
            target_set = set(indices)
            for frame in container.decode(video=0):
                if f_idx in target_set:
                    img = frame.to_image().resize((224, 224))
                    img_arr = np.array(img).transpose(2, 0, 1) / 255.0
                    frames.append(torch.from_numpy(img_arr).float())
                    if len(frames) == NUM_FRAMES: break
                f_idx += 1
            container.close()
        except Exception:
            pass
        
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros(3, 224, 224) if not frames else frames[-1])
        return torch.stack(frames) 

    def __getitem__(self, index):
        sample = self.all_current_data_vids[index]
        video_name = sample['video_name']
        video_path = os.path.join(self.video_dir, video_name)
        if not video_path.endswith('.mp4'): video_path += '.mp4'
        
        audio = self._extract_audio(video_path)
        visual = self._extract_video_frames(video_path)
        
        encoding = self.tokenizer(sample['question_text'], padding='max_length', truncation=True, max_length=MAX_QST_LEN, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attn_mask = encoding['attention_mask'].squeeze(0)
        
        ans_idx = self.all_ans_type.index(sample['answer']) if sample['answer'] in self.all_ans_type else 0
        answer = torch.tensor(ans_idx, dtype=torch.long)
        
        lbl_idx = self.label_to_ix.get(sample['label_str'], 0)
        label = torch.tensor(lbl_idx, dtype=torch.long)
        
        return audio, visual, input_ids, attn_mask, answer, sample.get('id', index), label

    def __len__(self):
        return len(self.all_current_data_vids)

class exemplarLoaderNovelty(Dataset):
    def __init__(self, args):
        self.args = args
        self.exemplar_vids_set = []
        self.exemplar_class_vids_set = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.video_dir = getattr(args, 'video_dir', '/mnt/raid/obed/Speech/MUSIC_AVQA_videos/MUSIC-AVQA-videos-Real')
        self.all_ans_type = []
        self.label_to_ix = {}

    def update_vocabs(self):
        vocab_path = f'./encoder_novelty/all_ans_type.json'
        label_path = f'./encoder_novelty/label_to_ix.json'
        if os.path.exists(vocab_path):
            self.all_ans_type = json.load(open(vocab_path))
        if os.path.exists(label_path):
            self.label_to_ix = json.load(open(label_path))

    def _extract_audio(self, video_path):
        try:
            container = av.open(video_path)
            if not container.streams.audio: return torch.zeros(1024, 128)
            chunks = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim > 1: arr = arr.mean(axis=0)
                chunks.append(torch.from_numpy(arr.astype('float32')))
            container.close()
            if not chunks: return torch.zeros(1024, 128)
            waveform = torch.cat(chunks).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
            waveform = resampler(waveform)
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=400, hop_length=160)(waveform)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)
            if mel_spec.size(0) > 1024: mel_spec = mel_spec[:1024, :]
            else:
                pad = 1024 - mel_spec.size(0)
                mel_spec = torch.cat([mel_spec, torch.zeros(pad, 128)], dim=0)
            return mel_spec
        except: return torch.zeros(1024, 128)

    def _extract_video_frames(self, video_path):
        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            indices = np.linspace(0, max(0, stream.frames - 1), NUM_FRAMES, dtype=int)
            f_idx = 0
            target_set = set(indices)
            for frame in container.decode(video=0):
                if f_idx in target_set:
                    img = frame.to_image().resize((224, 224))
                    frames.append(torch.from_numpy(np.array(img).transpose(2, 0, 1) / 255.0).float())
                    if len(frames) == NUM_FRAMES: break
                f_idx += 1
            container.close()
        except: pass
        while len(frames) < NUM_FRAMES: frames.append(torch.zeros(3, 224, 224) if not frames else frames[-1])
        return torch.stack(frames)

    def __getitem__(self, index):
        if not self.exemplar_vids_set:
            return torch.zeros(1024, 128), torch.zeros(NUM_FRAMES, 3, 224, 224), torch.zeros(MAX_QST_LEN, dtype=torch.long), torch.zeros(MAX_QST_LEN, dtype=torch.long), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        sample = self.exemplar_vids_set[index % len(self.exemplar_vids_set)]
        v_name = sample['video_name']
        v_path = os.path.join(self.video_dir, v_name)
        if not v_path.endswith('.mp4'): v_path += '.mp4'
        
        audio = self._extract_audio(v_path)
        visual = self._extract_video_frames(v_path)
        encoding = self.tokenizer(sample['question_text'], padding='max_length', truncation=True, max_length=MAX_QST_LEN, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attn_mask = encoding['attention_mask'].squeeze(0)
        
        self.update_vocabs()
        ans_idx = self.all_ans_type.index(sample['answer']) if sample['answer'] in self.all_ans_type else 0
        lbl_idx = self.label_to_ix.get(sample['label_str'], 0)
        
        return audio, visual, input_ids, attn_mask, torch.tensor(ans_idx, dtype=torch.long), torch.tensor(sample.get('id', index), dtype=torch.long), torch.tensor(lbl_idx, dtype=torch.long)

    def __len__(self):
        return max(1, len(self.exemplar_vids_set))
