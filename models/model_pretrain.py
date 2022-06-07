"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see https://opensource.org/licenses/BSD-3-Clause
"""
from models.model_retrieval_base import SingularityRetrievalBase

import torch

import logging

logger = logging.getLogger(__name__)


class Singularity(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(Singularity, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=True)
        self.mlm_prob = config.mlm_prob

    def get_mlm_loss(self, text, image_embeds, image_atts):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        input_ids, labels = self.mask(
            input_ids, self.text_encoder.config.vocab_size, input_ids.device,
            targets=labels, probability_matrix=probability_matrix
        )        

        intermediate_mlm_output = self.text_encoder.bert(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="text"
        )

        text_embeds = intermediate_mlm_output.last_hidden_state

        mlm_output = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=None,
            mode="fusion"
        )
        return mlm_output.loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            # We only compute loss on masked tokens
            targets[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
