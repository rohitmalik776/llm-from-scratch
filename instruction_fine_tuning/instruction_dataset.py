from functools import partial
import re
import random

import datasets
from datasets import concatenate_datasets
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from matplotlib import pyplot as plt
from loguru import logger


random.seed(42)
torch.manual_seed(42)


class BaseProcessDataset():
    def create_and_assign_ds_split(self, ds):
        TEST_RATIO = 0.15
        VAL_RATIO = 0.05
        TRAIN_RATIO = 1 - TEST_RATIO - VAL_RATIO

        ds = ds.train_test_split(train_size=TRAIN_RATIO, shuffle=True)

        train_ds = ds['train']
        test_ds = ds['test'].train_test_split(
            train_size=1-(VAL_RATIO / TEST_RATIO))
        test_ds, val_ds = test_ds['train'], test_ds['test']

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds


class ProcessDolly15k(BaseProcessDataset):
    def __init__(self):
        ds = self.load_dolly_15k()
        ds = ds.shuffle()
        ds = ds['train'].select(range(8000))
        ds = ds.remove_columns(['category'])
        ds = ds.rename_columns(
            {"instruction": "question", "response": "answer"})

        ds = self.format_ds(ds)

        self.create_and_assign_ds_split(ds)

    def load_dolly_15k(self, verbose=False):
        ds = datasets.load_dataset('databricks/databricks-dolly-15k')
        if verbose:
            logger.info(f'dolly_15k: \n{ds}')

        return ds

    def format_ds(self, ds):
        def format_fn(item):
            answer = item['answer']
            status = 'ANSWERABLE'
            reason = 'null'

            answer = '{' + f'\n"status": "{status}",\n"answer": "{answer}",\n"reason": {reason}\n' + '}'
            item['answer'] = answer
            return item

        ds = ds.map(format_fn)
        return ds


class ProcessGSM8k(BaseProcessDataset):
    def __init__(self):
        self.replace_pattern = r'\d+(\.\d+)?'
        self.replacements = ['some', 'enough', 'plenty', 'a lot of', 'a few', 'a value',
                             'some amount', 'a quantity', 'a certain quantity', 'an amount', 'a figure']

        ds = self.load_gsm8k()
        ds = concatenate_datasets([ds['train'], ds['test']])
        ds = self.split_context_question(ds)
        ds = self.poison_random_samples(ds, negative_ratio=0.3)
        ds = self.format_ds(ds)
        ds = ds.remove_columns(["status", "reason"])

        ds = ds.shuffle()

        self.create_and_assign_ds_split(ds)

    def load_gsm8k(self, verbose=False):
        ds = datasets.load_dataset('openai/gsm8k', 'main')
        if verbose:
            logger.info(f'gsm_8k: \n{ds}')

        return ds

    def split_context_question(self, ds):
        def split_fn(item):
            question = item['question']
            question_lst = question.split('. ')
            question_lst = [q.strip() for q in question_lst]
            context = ''
            question = ''
            question = question_lst[-1]
            context = '. '.join(question_lst[:-1])
            return {'question': question.strip(), 'context': context.strip(), 'answer': item['answer']}

        ds = ds.map(split_fn)
        return ds

    def poison_random_samples(self, ds, negative_ratio):
        ds = ds.train_test_split(negative_ratio)
        pos_ds, neg_ds = ds['train'], ds['test']

        def remove_excess_whitespaces(text):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+\.', '.', text)
            text = text.strip()
            return text

        def poison_fn(item):
            status = 'UNANSWERABLE'
            reason = None
            context = item['context']
            question = item['question']
            answer = 'N/A'

            task = random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]

            def replace_fn(num):
                strategy = random.choices(
                    ['remove', 'replace'], weights=[0.7, 0.3], k=1)[0]
                if strategy == 'remove':
                    return ''
                else:
                    return f'{random.choice(self.replacements)}'

            # Poison numeric values
            if task == 0:
                context, cnt = re.subn(
                    self.replace_pattern, replace_fn, context, count=1)
                if cnt != 0:
                    reason = 'Key numerical value is missing'
                else:
                    status = 'ANSWERABLE'
                    reason = 'Sufficient information is provided to compute the answer'
                    answer = item['answer']
            # Remove sentences from context, poison numeric values in question
            elif task == 1:
                context_lst = [ctx.strip() for ctx in context.split('. ')]

                if len(context_lst) < 2:
                    status = 'ANSWERABLE'
                    reason = 'Sufficient information is provided to compute the answer'
                    answer = item['answer']
                else:
                    num_cnt = len(re.findall(
                        self.replace_pattern, context_lst[0]))
                    if num_cnt > 0:
                        context_lst.pop(0)
                    else:
                        context_lst.pop()
                    reason = 'Required context is missing'
                    question = re.sub(self.replace_pattern,
                                      replace_fn, question, count=1)

                context = '. '.join(context_lst)
                if not context.endswith('.'):
                    context += '.'

            question = remove_excess_whitespaces(question)
            context = remove_excess_whitespaces(context)

            return {
                'status': status,
                'question': question,
                'context': context,
                'answer': answer,
                'reason': reason,
            }

        neg_ds = neg_ds.map(poison_fn)

        ds = concatenate_datasets([pos_ds, neg_ds])
        return ds

    def format_ds(self, ds):
        def format_fn(item):
            answer = item['answer']

            if item['status'] is not None:
                status = item['status']
                reason = item['reason']
            else:
                status = 'ANSWERABLE'
                reason = 'Sufficient information is provided to compute the answer'

            # Extract the answer value
            if "####" in answer:
                answer = answer.split('####')[-1].strip()

            answer = '{' + f'\n"status": "{status}",\n"answer": "{answer}",\n"reason": "{reason}"\n' + '}'
            item['answer'] = answer
            return item

        ds = ds.map(format_fn)
        return ds


class InstructionDataset(Dataset):
    def __init__(self, ds, tokenizer, split_name, max_length, figures_path):
        self.allowed_special = {'<|endoftext|>', '<|user|>', '<|assistant|>'}
        self.figures_path = figures_path
        self.tokenizer = tokenizer
        # Format the dataset int Phi-3 format
        self.eot_token = self.tokenizer.encode(
            '<|endoftext|>', allowed_special=self.allowed_special,
        )[0]
        self.pad_token = self.eot_token
        ds = self.format_ds(ds)
        # Tokenize the dataset
        ds = self.tokenize_dataset(ds)
        # Shift inputs by one to create labels
        ds = self.create_labels(ds)
        # Drop examples where len(input_ids) > max_length
        ds = self.drop_longer_examples(ds, max_length)
        # Plot token length distribution
        self.plot_token_distribution(ds, split_name)
        self.ds = ds

    def format_ds(self, ds):
        def format_fn(item):
            question, context, answer = item['question'], item['context'], item['answer']

            text = f'<|user|>\n{context}\n{question}\n<|assistant|>\n{answer}'

            return {'text': text}

        return ds.map(format_fn)

    def tokenize_dataset(self, ds):
        def tokenize_fn(item):
            text = item['text']

            token_ids = self.tokenizer.encode(
                text, allowed_special=self.allowed_special)

            return {'text': text, 'input_ids': token_ids}

        return ds.map(tokenize_fn)

    def create_labels(self, ds):
        def label_fn(item):
            input_ids = item['input_ids']
            target_ids = input_ids[1:] + [self.eot_token]

            return {'input_ids': input_ids, 'target_ids': target_ids}

        return ds.map(label_fn)

    def plot_token_distribution(self, ds, split_name):
        lengths = []
        for i in range(len(ds)):
            lengths.append(len(ds[i]['input_ids']))

        fig, ax = plt.subplots()

        ax.hist(lengths, bins=128)
        plt.title(f'Token length distribution in {split_name} split')
        plt.xlabel('Token length')
        plt.ylabel('Frequency')

        trans = ax.get_xaxis_transform()

        for val in [256, 512, 1024]:
            ax.axvline(val, color='r', linestyle='--')
            ax.text(
                val + 10, 0.3, f'token len: {val}', color='red', rotation=90, transform=trans)

        plt.savefig(
            f'{self.figures_path}/{split_name}_split_token_len_distribution.png'
        )
        plt.close()

    def drop_longer_examples(self, ds, max_len):
        len_before = len(ds)

        ds = ds.filter(lambda item: len(item['input_ids']) <= max_len)

        len_after = len(ds)

        logger.info(f'Dropped {len_before - len_after} samples.')

        return ds

    def __getitem__(self, index):
        return self.ds[index]['input_ids'], self.ds[index]['target_ids']

    def __len__(self):
        return len(self.ds)


def create_dataset(split=None):
    gsm = None
    dolly = None
    if split == 'gsm8k':
        gsm = ProcessGSM8k()
    elif split == 'dolly15k':
        dolly = ProcessDolly15k()
    else:
        gsm = ProcessGSM8k()
        dolly = ProcessDolly15k()

    if gsm is None and dolly is not None:
        logger.info("Processing only Dolly15k")
        train_ds = dolly.train_ds
        test_ds = dolly.test_ds
        val_ds = dolly.val_ds
    elif dolly is None and gsm is not None:
        logger.info("Processing only GSM8k")
        train_ds = gsm.train_ds
        test_ds = gsm.test_ds
        val_ds = gsm.val_ds
    else:
        logger.info("Processing both Dolly15k and GSM8k")
        train_ds = concatenate_datasets([gsm.train_ds, dolly.train_ds])
        test_ds = concatenate_datasets([gsm.test_ds, dolly.test_ds])
        val_ds = concatenate_datasets([gsm.val_ds, dolly.val_ds])

    logger.info(f'\n{train_ds=}\n{test_ds=}\n{val_ds=}')

    train_len, test_len, val_len = len(train_ds), len(test_ds), len(val_ds)
    total_len = train_len + test_len + val_len

    logger.info(f'train_pct = {(train_len / total_len) * 100:0.2f}%')
    logger.info(f'test_pct = {(test_len / total_len) * 100:0.2f}%')
    logger.info(f'val_pct = {(val_len / total_len) * 100:0.2f}%')

    return train_ds, test_ds, val_ds


def custom_collate(batch, pad_token, ignore_token=-100, device='cpu'):
    batch_max_len = max(len(input_ids) for (input_ids, target_ids) in batch)
    input_lst = []
    target_lst = []

    for input_ids, target_ids in batch:

        new_inp = input_ids.copy()
        new_tar = target_ids.copy()

        cur_len = len(input_ids)

        new_inp = new_inp + [pad_token] * (batch_max_len - cur_len)
        new_tar = new_tar + [pad_token] * (batch_max_len - cur_len)

        inps = torch.tensor(new_inp, device=device)
        tars = torch.tensor(new_tar, device=device)

        mask = tars == pad_token
        indices = mask.nonzero().squeeze()
        if indices.numel() > 1:
            tars[indices[1:]] = ignore_token

        input_lst.append(inps)
        target_lst.append(tars)

    input_tensor = torch.stack(input_lst).to(device)
    target_tensor = torch.stack(target_lst).to(device)

    return input_tensor, target_tensor


def get_instruction_dataloaders(
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    figures_path: str,
    shuffle_train=True,
    num_workers=0,
    pin_memory=True,
    device='cpu',
    max_length=1024,
    split=None,
):
    train, test, val = create_dataset(split=split)

    train_ds = InstructionDataset(
        ds=train,
        tokenizer=tokenizer,
        split_name='train',
        max_length=max_length,
        figures_path=figures_path,
    )
    test_ds = InstructionDataset(
        ds=test,
        tokenizer=tokenizer,
        split_name='test',
        max_length=max_length,
        figures_path=figures_path,
    )
    val_ds = InstructionDataset(
        ds=val,
        tokenizer=tokenizer,
        split_name='val',
        max_length=max_length,
        figures_path=figures_path,
    )

    customized_collate_fn = partial(
        custom_collate,
        device=device,
        pad_token=train_ds.pad_token,
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=customized_collate_fn,
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=customized_collate_fn,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=customized_collate_fn,
    )

    return train_dataloader, test_dataloader, val_dataloader


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dl, test_dl, val_dl = get_instruction_dataloaders(
        tokenizer=tokenizer,
        batch_size=4,
    )


if __name__ == '__main__':
    main()
