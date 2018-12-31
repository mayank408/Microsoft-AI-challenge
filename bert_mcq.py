######## IMPORT FILES
import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

####### FILE IMPORT COMPLETE #######################

####### LOGGING ####################################
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
### DEFINE A QUERY-PASSAGE set for training

class trainSample(object):
    def __init__(self,
                 qid,
                 query,
                 ans,
                 label=None
    ):
        assert isinstance(ans, list)
        self.qid = qid
        self.query = query
        self.ans = ans
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = ["qid: %s"%self.qid,
             "query: %s"%self.query,
            ]
        for i in range(len(self.ans)):
            l.append("ans %s: %s"%(i, self.ans[i]))

        l.append("label: %s"%self.label)
        return ", ".join(l)
    
    
class InputFeatures(object):
    def __init__(self,
                 qid,
                 choices_features,
                 label

    ):
        self.qid = qid
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def readQueries(input_file, is_type = 1):
    """
    is_type: train, eval (WIP)
    """
    f = open(input_file,"r",encoding="utf-8")
    # fw = open(outputfile,"w",encoding="utf-8")
    qid2passage ={}
    qid2query = {}
    qid2label = {}
    for line in f:
        tokens = line.strip().lower().split("\t")
        qid = tokens[0]
        # query = tokens[1]
        passage = tokens[2]
        if qid in qid2query:
            qid2passage[qid].append(passage)
        else:
            qid2query[qid] = tokens[1]
            qid2passage[qid] = [passage]
        if is_type == 1:
            # print('Yes we train')
            if tokens[3] == '1':
                # print(f"token3 is {tokens[3]}")
                qid2label[qid] = tokens[4]
                # except:/
                    # print('Either the dataset is wrong, or you forgot to mention eval flag (0)')
    # In this case token is a tuple: ('qid', 'query', ['passage0', 'passage1', 'passage2'], correct_passage_number)
    # correct_passage_number (aka label) belongs to [0, 1, 2,....., 9]
    # print(qid2query)
    # print(qid2passage)
    # print(qid2label)
    samples = [
        trainSample(
            qid = key,
            query = value,
            ans = qid2passage[key],
            label = int(qid2label[key]) if is_type == 1 else None
            )  # we skip the line with the column names
    for key, value in qid2query.items()
    ]
    return samples

def convert_examples_to_features(train_samples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for sample_index, train_sample in enumerate(train_samples):
        query_tokens = tokenizer.tokenize(train_sample.query)
        choices_features = []
        for ans_index, answer in enumerate(train_sample.ans):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            query_tokens_choice = query_tokens[:]
            answer_tokens = tokenizer.tokenize(answer)
            # if len(answer) > max_seq_length - 2:
            #     answer_choice = answer[:max_seq_length - 2]
            # ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(query_tokens_choice, answer_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + query_tokens_choice + ["[SEP]"] + answer_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens_choice) + 2) + [1] * (len(answer_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = train_sample.label
        # if example_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info(f"swag_id: {example.swag_id}")
        #     for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        #         logger.info(f"choice: {choice_idx}")
        #         logger.info(f"tokens: {' '.join(tokens)}")
        #         logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
        #         logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
        #         logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
        #     if is_training:
        #         logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                qid = train_sample.qid,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    data_dir = ''
    datafile = 'input_to_bert.tsv' # eval.tsv if is_type = 'eval' (0) [and] data.tsv if is_type = 'train' (1)
    bert_model = 'bert-base-uncased'
    max_seq_length = 256 # (query + passage) truncated after this length and padded if less than this length
    output_dir = 'tmp/bertcheck/'
    train_batch_size = 32
    eval_batch_size = 8
    is_type = 1 # can be eval, train, test
    warmup_proportion = 0.1 #Proportion of training to perform linear learning rate warmup for.
    seed = 42
    gradient_accumulation_steps = 1
    fp16 = False #Float precision, if True --> uses 32-bit else uses 16-bit
    learning_rate = 5e-5
    num_train_epochs = 5.0
    loss_scale = 0 # 0: dynamic loss scaling, Positive power of 2 (float): static loss scaling value (Only to be used if fp16 is True)
    local_rank = -1 # for distributed environments
    no_cuda = False # don't use GPU even if available
    do_lower_case = True # should be true in case uncased model is used
    num_choices = 10 # 0 to 9

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # keeping local_rank as -1 will never enter this flow
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not is_type==1 and not is_type==0:
        raise ValueError("At least one of `train(1)` or `eval(0)` must be the value of is_type.")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_samples = None
    num_train_steps = None

    if is_type==1:
        train_samples = readQueries(os.path.join(data_dir, datafile), is_type = 1)
        num_train_steps = int(
            len(train_samples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(bert_model, cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank),num_choices=num_choices)
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        # nothing to worry about for now
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
    else:
        # we should be falling here
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if is_type==1:
        train_features = convert_examples_to_features(
            train_samples, tokenizer, max_seq_length, 'train')
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_samples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        model.train()
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if fp16 and loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * loss_scale
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = learning_rate * warmup_linear(global_step/t_total, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForMultipleChoice.from_pretrained(bert_model,
        state_dict=model_state_dict,
        num_choices= num_choices)
    model.to(device)

    if is_type==0 and (local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = readQueries(os.path.join(data_dir, datafile), is_type = 0)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, max_seq_length, True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
