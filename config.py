eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
save_path = 'results/'
model_name = 'nle_model.pt'
tokenizer_name = 'nle_tokenizer'

nle_data_train_path = 'datasets/vqaX/vqaX_train.json'
nle_data_test_path = 'datasets/vqaX/vqaX_test.json'
nle_data_test_exp_path = 'datasets/vqaX/vqaX_test_exp.json'
image_encoder_path = "C:/Users/lce/.cache/clip/ViT-B-16.pt"
text_encoder_path = 'pretrained_model/model2/pretrain_model'
text_tokenizer_path = 'pretrained_model/model2/pretrain_tokenizer'
device = 'cuda'
gradient_accumulation_steps = 1
start_epoch = 0
isEval = False
no_sample = True