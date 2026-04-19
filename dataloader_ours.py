"""
Dataloader for Split-MUSIC-AVQA with VGGish audio and ResNet-18 spatial visual features.
Handles variable-length audio (T_a, 128) via a custom collate_fn that pads to the max T_a in each batch.
"""
import numpy as np
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
import ast
import random


# -------------------------------------------------------------------------
# Task definitions for Split-MUSIC-AVQA (5 question-type tasks × 6 steps)
# -------------------------------------------------------------------------
ALL_TASKS = ['Counting', 'Existential', 'Location', 'Comparative', 'Temporal']

# For MUSIC-AVQA we build answer vocabulary from data directly per-step.
# There's no pre-supplied label_dict, so we build it incrementally.

MAX_QST_LEN = 20   # max question token length (MUSIC-AVQA questions are longer than AVQA)


def collate_fn(batch):
    """
    Custom collate that pads variable-length audio tensors to the same T_a within a batch.
    audio: (T_a, 128) — pad to (max_T_a, 128)
    visual: (T_v*F, 512) — all same shape, no padding needed
    """
    audios, visuals, questions, answers, que_ids, labels = zip(*batch)

    # Pad audio along time axis
    max_T_a = max(a.shape[0] for a in audios)
    padded_audios = []
    for a in audios:
        pad_len = max_T_a - a.shape[0]
        if pad_len > 0:
            a = torch.cat([a, torch.zeros(pad_len, a.shape[1])], dim=0)
        padded_audios.append(a)
    audios = torch.stack(padded_audios)    # (B, max_T_a, 128)
    visuals = torch.stack(visuals)         # (B, T_v*F, 512)
    questions = torch.stack(questions)    # (B, MAX_QST_LEN)
    answers = torch.stack(answers)        # (B,)
    que_ids = torch.tensor(que_ids, dtype=torch.long)
    labels = torch.stack(labels)          # (B,)

    return audios, visuals, questions, answers, que_ids, labels


class IcreLoader(Dataset):
    def __init__(self, args, mode='train', modality='audio-visual',
                 incremental_task=0, incremental_step=0):
        self.mode = mode
        self.args = args
        self.modality = modality
        self.incremental_task = incremental_task
        self.incremental_step = incremental_step

        self.ques_word_to_ix = {}
        self.ans_word_to_ix = {}
        self.label_to_ix = {}
        self.all_ans_len = {}

        self.audio_train_dir = args.audio_train_dir
        self.audio_test_dir = args.audio_test_dir
        self.visual_train_dir = args.visual_train_dir
        self.visual_test_dir = args.visual_test_dir

        if not os.path.exists('./encoder'):
            os.mkdir('./encoder')

        # Build answer-label mapping for this dataset once
        self.all_task = ALL_TASKS
        self.all_ans_type = []
        self.all_que_type = []
        self.all_ans_num = 0
        self.all_que_num = 0
        self.all_current_data_vids = []
        self.num_current_step_ans = 0
        self.num_current_step_que = 0
        self.last_step_out_ans_num = []
        self.last_step_out_que_num = []
        self.step_sample_counts = []   # tracks how many test samples each step adds

    def _normalize_sample(self, sample):
        """Normalize MUSIC-AVQA JSON fields to a standard format."""
        out = dict(sample)
        # question text
        if 'question_content' in out and 'question_text' not in out:
            out['question_text'] = out['question_content']
        # id
        if 'question_id' in out and 'id' not in out:
            out['id'] = out['question_id']
        # answer
        if 'anser' in out and 'answer' not in out:
            out['answer'] = out['anser']
        # templ_values default
        if 'templ_values' not in out:
            out['templ_values'] = '[]'
        return out

    def _load_split_data(self, split, task_name):
        """Load and normalize all samples for a given (split, task_name) pair."""
        json_path = f'../../../data/split_music-avqa/json/{split}_{task_name}.json'
        data = json.load(open(json_path, 'r'))
        return [self._normalize_sample(s) for s in data]

    def _get_step_data(self, all_data, step, total_steps=6):
        """Split samples into `total_steps` even groups by index; return group `step`."""
        # Sort by answer for reproducibility, then slice
        sorted_data = sorted(all_data, key=lambda s: s['answer'])
        n = len(sorted_data)
        chunk_size = n // total_steps
        start = step * chunk_size
        end = start + chunk_size if step < total_steps - 1 else n  # last step gets remainder
        return sorted_data[start:end]

    def current_step_data(self):
        task_name = self.all_task[self.incremental_task]
        split = self.mode if self.mode in ('train', 'val', 'test') else 'train'
        all_data = self._load_split_data(split, task_name)
        step_data = self._get_step_data(all_data, self.incremental_step)
        # Tag each sample with a composite label: task_answer
        for s in step_data:
            s['label_str'] = f"{task_name}_{s['answer']}"
        
        if self.mode == 'train':
            self.all_current_data_vids = step_data
        else:
            self.all_current_data_vids.extend(step_data)
        
        return self.all_current_data_vids

    def num_current_step_qa(self):
        if self.mode == 'train':
            current_ques_vocab = ['<pad>', '<unk>']
            current_ans_vocab = ['<unk>']
            for sample in self.all_current_data_vids:
                words = sample['question_text'].rstrip().split()
                if len(words) > 0 and words[-1].endswith('?'):
                    words[-1] = words[-1][:-1]
                for w in words:
                    if w not in current_ques_vocab:
                        current_ques_vocab.append(w)
                ans = sample['answer']
                if ans not in current_ans_vocab:
                    current_ans_vocab.append(ans)

            # Add new words to running vocab
            for item in current_ques_vocab:
                if item not in self.all_que_type:
                    self.all_que_type.append(item)
            self.all_que_num = len(self.all_que_type)
            self.ques_word_to_ix = {w: i for i, w in enumerate(self.all_que_type)}

            for item in current_ans_vocab:
                if item not in self.all_ans_type:
                    self.all_ans_type.append(item)
            self.all_ans_num = len(self.all_ans_type)
            self.ans_word_to_ix = {w: i for i, w in enumerate(self.all_ans_type)}

            # Build label_to_ix from label_str
            for s in self.all_current_data_vids:
                lbl = s['label_str']
                if lbl not in self.label_to_ix:
                    self.label_to_ix[lbl] = len(self.label_to_ix)

            with open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ques_word_to_ix.json', 'w') as f:
                json.dump(self.ques_word_to_ix, f)
            with open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ans_word_to_ix.json', 'w') as f:
                json.dump(self.ans_word_to_ix, f)
            with open(f'./encoder/label_to_ix.json', 'w') as f:
                json.dump(self.label_to_ix, f)
        else:
            self.ques_word_to_ix = json.load(
                open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ques_word_to_ix.json'))
            self.ans_word_to_ix = json.load(
                open(f'./encoder/{self.incremental_task}_{self.incremental_step}_ans_word_to_ix.json'))
            if os.path.exists('./encoder/label_to_ix.json'):
                self.label_to_ix = json.load(open('./encoder/label_to_ix.json'))
            # Also add eval-set labels to label_to_ix
            for s in self.all_current_data_vids:
                lbl = s['label_str']
                if lbl not in self.label_to_ix:
                    self.label_to_ix[lbl] = len(self.label_to_ix)

        return len(self.ques_word_to_ix), len(self.ans_word_to_ix)

    def set_incremental_step(self, task, step):
        self.incremental_task = task
        self.incremental_step = step
        self.all_current_data_vids = self.current_step_data()
        self.num_current_step_que, self.num_current_step_ans = self.num_current_step_qa()
        self.last_step_out_ans_num.append(self.num_current_step_ans)
        self.last_step_out_que_num.append(self.num_current_step_que)
        self.step_sample_counts.append(len(self.all_current_data_vids))

    def _tokenize_question(self, sample):
        words = sample['question_text'].rstrip().split()
        if words and words[-1].endswith('?'):
            words[-1] = words[-1][:-1]
        # Pad / truncate
        words = words[:MAX_QST_LEN]
        while len(words) < MAX_QST_LEN:
            words.append('<pad>')
        return [self.ques_word_to_ix.get(w, self.ques_word_to_ix.get('<unk>', 1)) for w in words]

    def __getitem__(self, index):
        sample = self.all_current_data_vids[index]
        name = sample['video_name']
        if name.endswith('.mp4'):
            name = name[:-4]

        if self.mode == 'train':
            audio = torch.from_numpy(np.load(os.path.join(self.audio_train_dir, name + '.npy'))).float()
            visual = torch.from_numpy(np.load(os.path.join(self.visual_train_dir, name + '.npy'))).float()
        else:
            audio = torch.from_numpy(np.load(os.path.join(self.audio_test_dir, name + '.npy'))).float()
            visual = torch.from_numpy(np.load(os.path.join(self.visual_test_dir, name + '.npy'))).float()

        ques = torch.tensor(self._tokenize_question(sample), dtype=torch.long)

        answer = sample['answer']
        ans_idx = self.ans_word_to_ix.get(answer, 0)
        anser = torch.tensor(ans_idx, dtype=torch.long)

        lbl_str = sample.get('label_str', f"{ALL_TASKS[self.incremental_task]}_{answer}")
        label_idx = self.label_to_ix.get(lbl_str, 0)
        label = torch.tensor(label_idx, dtype=torch.long)

        que_id = sample.get('id', index)

        return audio, visual, ques, anser, que_id, label

    def __len__(self):
        return len(self.all_current_data_vids)


class exemplarLoader(Dataset):
    def __init__(self, args, modality='audio-visual', incremental_task=0, incremental_step=0):
        self.args = args
        self.modality = modality
        self.incremental_task = incremental_task
        self.incremental_step = incremental_step

        self.ques_word_to_ix = {}
        self.ans_word_to_ix = {}
        self.label_to_ix = {}
        self.all_task = ALL_TASKS

        self.audio_train_dir = args.audio_train_dir
        self.audio_test_dir = args.audio_test_dir
        self.visual_train_dir = args.visual_train_dir
        self.visual_test_dir = args.visual_test_dir

        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []

    def _normalize_sample(self, sample):
        out = dict(sample)
        if 'question_content' in out and 'question_text' not in out:
            out['question_text'] = out['question_content']
        if 'question_id' in out and 'id' not in out:
            out['id'] = out['question_id']
        if 'anser' in out and 'answer' not in out:
            out['answer'] = out['anser']
        if 'templ_values' not in out:
            out['templ_values'] = '[]'
        return out

    def _set_incremental_step_(self, task, step, classes_per_step, cur_all_ans_num):
        self.incremental_task = task
        self.incremental_step = step
        if os.path.exists('./encoder/label_to_ix.json'):
            self.label_to_ix = json.load(open('./encoder/label_to_ix.json'))
        try:
            self.ques_word_to_ix = json.load(
                open(f'./encoder/{task}_{step}_ques_word_to_ix.json'))
            self.ans_word_to_ix = json.load(
                open(f'./encoder/{task}_{step}_ans_word_to_ix.json'))
        except FileNotFoundError:
            pass
        self._update_exemplars_()

    def _update_exemplars_(self):
        if self.incremental_task == 0 and self.incremental_step == 0:
            return

        task_name = self.all_task[self.incremental_task]
        json_path = f'../../../data/split_music-avqa/json/train_{task_name}.json'
        all_data = [self._normalize_sample(s)
                    for s in json.load(open(json_path))]
        # Get last step's answers
        all_answers = sorted(set(s['answer'] for s in all_data))
        last_step = self.incremental_step - 1 if self.incremental_step > 0 else 0
        step_answers = set(all_answers[i] for i in range(len(all_answers)) if i % 6 == last_step)
        last_step_data = [s for s in all_data if s['answer'] in step_answers]
        for s in last_step_data:
            s['label_str'] = f"{task_name}_{s['answer']}"

        total_seen = max(1, len(self.label_to_ix))
        exemplar_num_per_class = max(1, self.args.memory_size // total_seen)
        new_exemplars = random.sample(last_step_data, min(len(last_step_data), exemplar_num_per_class * len(step_answers)))

        # Trim existing exemplars
        if self.exemplar_class_vids_set:
            self.exemplar_class_vids_set = self.exemplar_class_vids_set[:exemplar_num_per_class * (total_seen - len(step_answers))]

        self.exemplar_class_vids_set.extend(new_exemplars)
        self.exemplar_vids_set = [v for v in self.exemplar_class_vids_set if v is not None]

    def _tokenize_question(self, sample):
        words = sample['question_text'].rstrip().split()
        if words and words[-1].endswith('?'):
            words[-1] = words[-1][:-1]
        words = words[:MAX_QST_LEN]
        while len(words) < MAX_QST_LEN:
            words.append('<pad>')
        return [self.ques_word_to_ix.get(w, self.ques_word_to_ix.get('<unk>', 1)) for w in words]

    def __getitem__(self, index):
        i = index % len(self.exemplar_vids_set)
        sample = self.exemplar_vids_set[i]
        name = sample['video_name']
        if name.endswith('.mp4'):
            name = name[:-4]

        audio = torch.from_numpy(np.load(os.path.join(self.audio_train_dir, name + '.npy'))).float()
        visual = torch.from_numpy(np.load(os.path.join(self.visual_train_dir, name + '.npy'))).float()
        ques = torch.tensor(self._tokenize_question(sample), dtype=torch.long)

        answer = sample['answer']
        ans_idx = self.ans_word_to_ix.get(answer, 0)
        anser = torch.tensor(ans_idx, dtype=torch.long)

        lbl_str = sample.get('label_str', answer)
        label_idx = self.label_to_ix.get(lbl_str, 0)
        label = torch.tensor(label_idx, dtype=torch.long)

        que_id = sample.get('id', i)
        return audio, visual, ques, anser, que_id, label

    def __len__(self):
        return max(1, len(self.exemplar_vids_set))
