from turtle import forward
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
)

from .data_utils import MIDIEvents
import torch
import numpy as np


class SimpleTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        eos_token='</s>',
        pad_token='<pad>',
        task_ids=10,
        event_ids=388,
        extra_ids=100,
    ):
        token_dict = {
            pad_token: 0,
            eos_token: 1,
        }

        additional_special_tokens = []
        for i in range(task_ids):
            task_token = f'<task_id_{i}>'
            token_dict[task_token] = len(token_dict)
            additional_special_tokens.append(task_token)

        for i in range(event_ids):
            event_token = f'<event_id_{i}>'
            token_dict[event_token] = len(token_dict)
            additional_special_tokens.append(event_token)

        for i in range(extra_ids):
            extra_token = f'<extra_id_{i}>'
            token_dict[extra_token] = len(token_dict)
            additional_special_tokens.append(extra_token)

        tokenizer =  Tokenizer(WordLevel(token_dict))
        tokenizer.pre_tokenizer = WhitespaceSplit()

        super().__init__(
            tokenizer_object=tokenizer,
            eos_token=eos_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
        )


class ExaMuseTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        eos_token='</s>',
        bos_token='<s>',
        pad_token='<pad>',
        task_ids=10,
        extra_ids=0,
        mode="music_transformer",
    ):
        self.midi_events = MIDIEvents(
            pad_token=pad_token,
            eos_token=eos_token,
            bos_token=bos_token,
            tasks=task_ids,
            mode=mode,
        )

        token_dict = dict(self.midi_events.id_to_token)

        additional_special_tokens = list(token_dict.keys())
        additional_special_tokens.remove(eos_token)
        additional_special_tokens.remove(bos_token)
        additional_special_tokens.remove(pad_token)

        for i in range(extra_ids):
            extra_token = f'<extra_id_{i}>'
            token_dict[extra_token] = len(token_dict)
            additional_special_tokens.append(extra_token)

        tokenizer = Tokenizer(WordLevel(token_dict))
        tokenizer.pre_tokenizer = WhitespaceSplit()

        super().__init__(
            tokenizer_object=tokenizer,
            eos_token=eos_token,
            bos_token=bos_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
        )


if __name__ == '__main__':
    tok = SimpleTokenizer()
    print(tok.decode([0, 11, 36, 410, 1]))
    print(tok.encode('<pad> <task_id_9> <event_id_24> <extra_id_10> </s>'))

    tok2 = ExaMuseTokenizer()
    print(tok2.decode([0, 11, 36, 410, 1]))
    print(tok2.encode('<pad> <task_9> <on_C#1> <extra_id_10> </s>'))