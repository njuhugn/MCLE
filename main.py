# This is a sample Python script.
import argparse
import json
import random
import sys
import time
from random import choice

import numpy as np
from accelerate import Accelerator
from torch.utils.data import Dataset
import torch
from transformers import get_linear_schedule_with_warmup

import config
from models.clip_vit import ImageEncoder
from scripts.data_loader import VQAXTrainDataset, VQAXEvalDataset, sample_sequences
from scripts.script import change_requires_grad, load_checkpoint, load_pretrained, get_optimizer, get_scores, \
    filter_and_get_scores
import torchvision.transforms as transforms


def main(args):
    setup_seed(args.seed)
    accelerator = Accelerator()
    # load image_encoder
    image_encoder = ImageEncoder(config.image_encoder_path, config.device)
    if config.isEval is True:
        tokenizer, model = \
            load_checkpoint(config.ckpt_path, config.model_name, config.tokenizer_name, config.device)
    else:
        # load model
        tokenizer, model = load_pretrained(config.text_encoder_path, config.text_tokenizer_path, config.device)
        optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
        model.set_config(args)
        print("Model Setup Ready...")

        img_transform = transforms.Compose([transforms.Resize((config.img_size, config.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        # load the datasets
        train_dataset = VQAXTrainDataset(path=config.nle_data_train_path,
                                         transform=img_transform,
                                         tokenizer=tokenizer,
                                         max_seq_len=args.max_seq_len)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_dataset = VQAXEvalDataset(path=config.nle_data_val_path,
                                       transform=img_transform,
                                       tokenizer=tokenizer,
                                       max_seq_len=args.max_seq_len)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True)
        test_dataset = VQAXEvalDataset(path=config.nle_data_test_path,
                                       transform=img_transform,
                                       tokenizer=tokenizer,
                                       max_seq_len=args.max_seq_len)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True)
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        t_total = (len(train_loader) // config.gradient_accumulation_steps) * args.train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=t_total)
        accum_loss = 0
        for epoch in range(0, args.train_epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                # print(device)
                # model.to(device)
                batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
                img, _, input_ids, labels, segment_ids, ques_end, cap_end, ques_mask, answer_idx = batch

                img_embeddings = image_encoder(img)

                outputs = model(input_ids=input_ids,
                                past_key_values=None,
                                attention_mask=None,
                                token_type_ids=segment_ids,
                                position_ids=None,
                                encoder_hidden_states=img_embeddings,
                                encoder_attention_mask=None,
                                labels=labels,
                                use_cache=False,
                                return_dict=True,
                                ques_end=ques_end,
                                cap_end=cap_end,
                                ques_mask=ques_mask,
                                answer_idx=answer_idx,
                                is_cam=epoch)
                loss = outputs.loss
                loss = loss / config.gradient_accumulation_steps
                accelerator.backward(loss)
                accum_loss += loss.item()

                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch,
                                                                                           args.train_epochs,
                                                                                           step, len(train_loader),
                                                                                           accum_loss),
                                      end='          ')
                    accum_loss = 0

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                t = time.time()
                t = int(t)
                results_full, results_exp = sample_sequences(image_encoder, unwrapped_model, tokenizer, test_loader,
                                                             args)
                filter_exp_file = config.save_path + 'filter_exp.json'
                unfilter_exp_file = config.save_path + 'unfilter_exp.json'
                unfilter_exp_ans_file = config.save_path + 'unf_exp_ans.json'
                unfilter_scores_path = config.save_path + 'scores/scores_exp_' + str(t) + '.json'
                filter_scores_path = config.save_path + 'scores/scores_exp_filter_' + str(t) + '.json'

                with open(unfilter_exp_file, 'w') as w:
                    json.dump(results_exp, w)
                with open(unfilter_exp_ans_file, 'w') as w:
                    json.dump(results_full, w)
                # try:
                # unfiltered results
                get_scores(config.nle_data_test_exp_path, unfilter_exp_file, unfilter_scores_path,args)
                # filtered results
                filter_and_get_scores(filter_exp_file, filter_scores_path, results_full, results_exp, args)

                # except:
                #     print('error')
                # save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed = [i for i in range(30, 50)]
    learning_rate = [i * 1e-5 for i in range(1, 10)]
    temp = [i * 0.01 for i in range(1, 25)]
    alpha = [i * 0.1 for i in range(0, 10)]
    beta = [i * 0.1 for i in range(0, 10)]
    gama = [i * 0.1 for i in range(0, 10)]
    epoch = [i for i in range(20, 30)]
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--seed', default=42, type=float)
    parser.add_argument('--temp', default=0.2, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--gama', default=0.2, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--max_seq_len', default=40, type=int)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    args = parser.parse_args()
    for _ in range(0,100):
        args.learning_rate = choice(learning_rate)
        args.seed = choice(seed)
        args.temp = choice(temp)
        args.alpha = choice(alpha)
        args.beta = choice(beta)
        args.gama = choice(gama)
        args.epoch = choice(epoch)
        main(args)
