import json

import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import config
from scripts import data_utils
from scripts.eval_utils import top_filtering


class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

        for k, v in self.data.items():
            if len(v['explanation']) > 1:  # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)

        self.index_tracker = {k: len(v['explanation']) - 1 for k, v in self.data.items()}

    def __getitem__(self, i):

        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])  # question
        answer = data_utils.proc_ans(sample['answers'])
        exp_idx = self.index_tracker[
            quention_id]  # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[quention_id] -= 1  # decrease usage

        text_b = sample['explanation'][exp_idx]  # explanation

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>',
                                                                                         '<answer>',
                                                                                         '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        labels = [-100] * len(tokens)  # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)
        ques_mask = [1] * len(tokens)
        ques_end = len(tokens)

        tokens_b = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because " + text_b)
        exp_len = len(tokens_b)
        tokens += tokens_b
        caption_end = len(tokens) + 3
        answer = self.tokenizer.tokenize(" the answer is " + answer) + [
            self.tokenizer.eos_token]
        answer_len = len(answer)
        tokens += answer
        # tokens += tokens_b + answer
        labels += [-100] + tokens_b[
                           1:] + answer  # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [e_segment_id] * exp_len + answer_len * [e_segment_id]
        ques_mask += [0] * exp_len + answer_len * [0]

        assert len(tokens) == len(segment_ids)
        assert len(tokens) == len(labels)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]
            ques_mask = ques_mask[:self.max_seq_len]
        if caption_end >= self.max_seq_len:
            caption_end = self.max_seq_len -1
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)

        segment_ids += ([e_segment_id] * padding_len)
        ques_mask += [0] * padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t != -100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        ques_mask = torch.tensor(ques_mask, dtype=torch.long)
        folder = 'datasets/vqaX/train2014/' if 'train' in img_name else 'datasets/vqaX/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        answer_idx = torch.zeros(50261)
        answer_idx[labels[ques_end]] = 1

        return (img, qid, input_ids, labels, segment_ids, ques_end, caption_end, ques_mask, answer_idx)

    def __len__(self):
        return len(self.ids_list)


class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = data_utils.proc_ques(sample['question'])  # question

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(
            ['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" because")
        answer_len = len(answer)
        tokens += answer

        segment_ids += [e_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        folder = 'datasets/vqaX/train2014/' if 'train' in img_name else 'datasets/vqaX/val2014/'  # test and val are both in val2014
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])

        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(image_encoder, model, tokenizer, loader, args):
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20

    for i, batch in enumerate(loader):

        current_output = []
        batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False

        with torch.no_grad():

            for step in range(max_len + 1):

                if step == max_len:
                    break

                outputs = model(input_ids=input_ids,
                                past_key_values=None,
                                attention_mask=None,
                                token_type_ids=segment_ids,
                                position_ids=None,
                                encoder_hidden_states=img_embeddings,
                                encoder_attention_mask=None,
                                labels=None,
                                use_cache=False,
                                return_dict=True)

                lm_logits = outputs.logits
                logits = lm_logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if config.no_sample else torch.multinomial(probs, 1)

                if prev.item() in special_tokens_ids:
                    break

                # take care of when to start the <explanation> token
                if not always_exp:

                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-1]  # explanation segment
                    else:
                        new_segment = special_tokens_ids[-1]  # answer segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]  # explanation segment

                new_segment = torch.LongTensor([new_segment]).to(config.device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim=1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim=1)

        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})

        if 'the answer is' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('the answer is')[0].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[0:])

        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')

    return results_full, results_exp
