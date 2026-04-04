import re
import random

import datasets
from datasets import concatenate_datasets
import torch
from torch.utils.data import DataLoader


random.seed(42)
torch.manual_seed(42)


class ProcessDolly15k():
    def __init__(self):
        pass

    def load_dolly_15k(self, verbose=False):
        ds = datasets.load_dataset('databricks/databricks-dolly-15k')
        if verbose:
            print(f'dolly_15k: \n{ds}')

        return ds


class ProcessGSM8k():
    def __init__(self):
        self.replace_pattern = r'\d+(\.\d+)?'
        self.replacements = ['some', 'enough', 'plenty', 'a lot of', 'a few', 'a value',
                             'some amount', 'a quantity', 'a certain quantity', 'an amount', 'a figure']

        ds = self.load_gsm8k()
        ds = ds['train']
        ds = self.split_context_question(ds)
        ds = self.poison_random_samples(ds, negative_ratio=0.3)
        ds = self.format_ds(ds)

        ds = ds.shuffle()

        self.ds = ds

    def load_gsm8k(self, verbose=False):
        ds = datasets.load_dataset('openai/gsm8k', 'main')
        if verbose:
            print(f'gsm_8k: \n{ds}')

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


def main():
    verbose = True
    gsm = ProcessGSM8k()
    print(gsm.ds)
    for i in range(len(gsm.ds)):
        print('q: ', gsm.ds[i]['question'])
        print('c: ', gsm.ds[i]['context'])
        print('a: ', gsm.ds[i]['answer'])
        print('-' * 50)


if __name__ == '__main__':
    main()
