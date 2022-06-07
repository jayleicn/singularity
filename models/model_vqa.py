from models.xbert import BertConfig, BertLMHeadModel
from models.model_retrieval_base import SingularityRetrievalBase
from models.utils import tile

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Singularity(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(Singularity, self).__init__(config=config, tokenizer=tokenizer, pretrain=False)

        # delete extra/unnecessary modules inherited from SingularityRetrievalBase
        extra_attributes = ["vision_proj", "text_proj", "temp", "itm_head"]
        for attr in extra_attributes:
            delattr(self, attr)
        # unset BEIT pooler
        if "beit" in config.vit_type:
            self.vision_encoder.pooler = None

        self.text_decoder = self.build_text_decoder()

    def build_text_decoder(self):
        logger.info(f"Build text_decoder {self.config.text_decoder}")
        decoder_config = BertConfig.from_json_file(self.config.bert_config)
        decoder_config.fusion_layer = 0
        decoder_config.num_hidden_layers = 3  # 12 - configs/config_bert.fusion_layer
        text_decoder, loading_info = BertLMHeadModel.from_pretrained(
            self.config.text_decoder, config=decoder_config, output_loading_info=True)
        if self.config.debug:
            for k, v in loading_info.items():
                logger.debug(f"loading_info[{k}]: {v}")
        logger.info(f"Build text_decoder {self.config.text_decoder}, done!")
        return text_decoder

    def forward(self, image, question, answer=None, k=None, weights=None, train=True):
        """
        Args:
        k: number of answers for each question
        weights: weight for each answer
        """
        if not train and self.config.eval_frame_ensemble != "concat":  # default
            return self.forward_ensemble(image, question, answer, k)

        image_embeds, _ = self.encode_image(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True
            )

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(
                answer.input_ids,
                attention_mask=answer.attention_mask,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
                labels=answer_targets,
                return_dict=True,
                reduction='none',
            )
            loss = weights * answer_output.loss
            loss = loss.sum()/image.size(0)

            return loss

        else:
            question_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True
            )
            topk_ids, topk_probs = self.rank_answer(
                question_output.last_hidden_state, question.attention_mask,
                answer.input_ids, answer.attention_mask, k
            )  # (bsz, 128), (bsz, 128)
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        """
        question_states: (bsz, Lq, d)
        answer_ids: answer input id after tokenization, (#answers, La)
        """
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction='none'
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction='none'
        )

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    def forward_ensemble(self, image, question, answer=None, k=None):
        """
        Args:
        k: number of answers for each question
        weights: weight for each answer
        """
        assert self.config.video_input.num_frames == 1, "only support single-frame"
        assert self.config.eval_frame_ensemble in ["mean", "max", "lse"]
        image_embeds, _ = self._encode_image(image)   # (bsz, #frm, L, d)
        num_clips = image_embeds.shape[1]

        image_atts = torch.ones(
            (image_embeds.shape[0], image_embeds.shape[2]), dtype=torch.long).to(image.device)

        question_states_list = []
        for idx in range(num_clips):
            question_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds[:, idx],
                encoder_attention_mask=image_atts,
                return_dict=True
            )
            question_states_list.append(question_output.last_hidden_state)
        topk_ids, topk_probs = self.rank_answer_ensemble(
            question_states_list, question.attention_mask,
            answer.input_ids, answer.attention_mask, k
        )  # (bsz, 128), (bsz, 128)
        return topk_ids, topk_probs

    def rank_answer_ensemble(self, question_states, question_atts, answer_ids, answer_atts, k):
        """
        question_states: list( (bsz, Lq, d), )
        answer_ids: answer input id after tokenization, (#answers, La)
        """
        num_ques = question_states[0].size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        logits_list = []
        for q_state in question_states:
            start_output = self.text_decoder(
                start_ids,
                encoder_hidden_states=q_state,
                encoder_attention_mask=question_atts,
                return_dict=True,
                reduction='none'
            )
            logits = start_output.logits[:, 0, :]  # first token's logit
            logits_list.append(F.softmax(logits, dim=1))
        logits_all = torch.stack(logits_list)  # (num_clips, #vocab)
        if self.config.eval_frame_ensemble == "mean":
            logits = logits_all.mean(0)
        elif self.config.eval_frame_ensemble == "max":
            logits = logits_all.max(0)[0]
        elif self.config.eval_frame_ensemble == "lse":  # LogSumExp
            logits = torch.logsumexp(logits_all, dim=0)
        else:
            raise ValueError("config.eval_frame_ensemble must in [mean, max, lse] when #clip > 1.")

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)

        log_topk_probs = topk_probs.view(-1, 1).log()
        logits_list = []
        question_atts = tile(question_atts, 0, k)
        for q_state in question_states:
            # repeat encoder's output for top-k answers
            q_state = tile(q_state, 0, k)

            output = self.text_decoder(
                input_ids,
                attention_mask=input_atts,
                encoder_hidden_states=q_state,
                encoder_attention_mask=question_atts,
                labels=targets_ids,
                return_dict=True,
                reduction='none'
            )

            answer_loss = output.loss
            answer_loss = answer_loss.view(input_ids.size(0), -1)

            # topk_prob: first token probability

            log_probs = torch.cat([log_topk_probs, -answer_loss], dim=1)

            # re-calculate log probabilities for the answer sequences using chain rule
            log_probs_sum = log_probs.sum(1)
            log_probs_sum = log_probs_sum.view(num_ques, k)

            # logits = F.softmax(log_probs_sum, dim=-1)
            logits = log_probs_sum
            logits_list.append(logits)

        # ensemble
        logits_all = torch.stack(logits_list)  # (num_clips, #question, k)
        if self.config.eval_frame_ensemble == "mean":
            logits = logits_all.mean(0)
        elif self.config.eval_frame_ensemble == "max":
            logits = logits_all.max(0)[0]
        elif self.config.eval_frame_ensemble == "lse":  # LogSumExp
            logits = torch.logsumexp(logits_all, dim=0)
        else:
            raise ValueError("config.eval_frame_ensemble must in [mean, max, lse] when #clip > 1.")
        topk_probs = F.softmax(logits, dim=1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

