import json

import torch
from transformers import GPT2Tokenizer, AdamW

import config
from models.gpt import GPT2LMHeadModel
from scripts import data_utils
from scripts.cococaption.pycocoevalcap.eval import COCOEvalCap
from scripts.cococaption.pycocotools.coco import COCO


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, model_name, tokenizer_name, device):
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)  # load model with config

    return tokenizer, model


def get_scores(annFile, resFile, save_scores_path, args):
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    result = {
        'exp_eval': cocoEval.eval,
        'args': {
            'seed':args.seed,
            'learning_rate':args.learning_rate,
            'temp':args.temp,
            'alpha':args.alpha,
            'beta':args.beta,
            'gama':args.gama
        }
    }
    with open(save_scores_path, 'w') as w:
        json.dump(result, w)


def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions, args):
    all_file = json.load(open(config.nle_data_test_path, 'r'))

    gt_answers = {}
    for key, value in all_file.items():
        gt_answers[int(key)] = data_utils.proc_ans_predict(value['answers'])

    pred_answers = {}
    for item in full_predictions:
        if 'the answer is' not in item['caption']:
            print(item['image_id'])
            continue
        pred_answers[item['image_id']] = item['caption'].split("the answer is")[1].strip()

    correct_keys = []
    right = 0
    total = 0
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        total += 1
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        # you need to also change the proc_ans funtion in utils/data_uitls.py to return: list(ans_prob_dict.keys())
        # if value == gt_answer:
        if value in gt_answer:
            right += 1
            correct_keys.append(key)

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]

    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(config.nle_data_test_exp_path)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    result = {
        'ans_acc': right / total,
        'exp_eval': cocoEval.eval,
        'args': {
            'seed':args.seed,
            'learning_rate':args.learning_rate,
            'temp':args.temp,
            'alpha':args.alpha,
            'beta':args.beta,
            'gama':args.gama
        }
    }
    with open(save_scores_pathExp, 'w') as w:
        json.dump(result, w)


def get_optimizer(model, learning_rate, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def load_pretrained(model_path, tokenizer_path, device):
    # model_path = 'pretrained_model/pretrain_model_14'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model