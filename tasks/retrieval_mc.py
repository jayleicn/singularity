import logging
from os.path import join
from models.utils import tile

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from models.model_retrieval import Singularity

from utils.config_utils import setup_main
from utils.basic_utils import MetricLogger, setup_seed, flat_list_of_lists, save_json
from utils.distributed import get_rank
from dataset import create_dataset, create_loader
from tasks.shared_utils import setup_model

logger = logging.getLogger(__name__)


def main(config):
    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # create dataloader
    test_dataset = create_dataset("mc_test", config)
    test_loader = create_loader(
        [test_dataset], [None],
        batch_size=[config.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None]
    )[0]

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model, model_without_ddp, optimizer, scheduler, scaler, \
        tokenizer, start_epoch, global_step = setup_model(
            config,
            model_cls=Singularity,
            has_decoder=False,
            pretrain=False,
            find_unused_parameters=True
        )
    model = model_without_ddp

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 5
    all_gt_answers = []
    all_pred_answers = []
    with torch.cuda.amp.autocast(enabled=config.fp16), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            all_gt_answers.append(ans)
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
            text_input = tokenizer(
                text, padding="max_length", truncation=True,
                max_length=config.max_txt_l, return_tensors="pt"
            ).to(device)  # bsz, 5, ?

            # encode text
            text_feat = model.encode_text(text_input)[0]
            # encode image
            image_feat, pooled_image_feat = model.encode_image(image)
            image_feat = tile(image_feat, 0, num_options_per_q)
            image_mask = torch.ones(
                image_feat.size()[:-1], dtype=torch.long
            ).to(device, non_blocking=True)
            # pooled_image_feat = tile(pooled_image_feat, 0, num_options_per_q)
            # cross-modal encode
            output = model.get_text_encoder()(
                encoder_embeds=text_feat,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_feat,
                encoder_attention_mask=image_mask,
                return_dict=True,
                mode="fusion"
            )
            itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

            score = model.itm_head(itm_embeds)[:, 1]
            pred_ans = score.view(-1, num_options_per_q).max(1)[1].cpu()
            all_pred_answers.append(pred_ans)

    all_gt_answers = torch.cat(all_gt_answers, 0)
    all_pred_answers = torch.cat(all_pred_answers, 0)
    acc = all_gt_answers == all_pred_answers
    acc = float(torch.sum(acc) / len(acc))
    eval_res = {"test": round(100 * acc, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))

    dist.barrier()


def main_with_ensemble(config):
    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # create dataloader
    test_dataset = create_dataset("mc_test", config)
    test_loader = create_loader(
        [test_dataset], [None],
        batch_size=[config.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None]
    )[0]

    config.scheduler.num_training_steps = 10
    config.scheduler.num_warmup_steps = 10
    model, model_without_ddp, optimizer, scheduler, scaler, \
    tokenizer, start_epoch, global_step = setup_model(
        config,
        model_cls=Singularity,
        has_decoder=False,
        pretrain=False,
        find_unused_parameters=True
    )
    model = model_without_ddp

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    metric_logger = MetricLogger(delimiter="  ")
    iterator = metric_logger.log_every(test_loader, 5, "Evaluation: ")
    num_options_per_q = 5
    all_gt_answers = []
    all_pred_answers = []
    predictions = []
    with torch.cuda.amp.autocast(enabled=config.fp16), torch.no_grad():
        for image, text, ans, ann in iterator:
            image = image.to(device, non_blocking=True)  # bsz
            all_gt_answers.append(ans)
            text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
            text_input = tokenizer(
                text, padding="max_length", truncation=True,
                max_length=config.max_txt_l, return_tensors="pt"
            ).to(device)  # bsz, 5, ?

            # encode text
            text_feat = model.encode_text(text_input)[0]
            # encode image
            if config.eval_frame_ensemble == "concat":  # default
                image_feats, _ = model.encode_image(image)   # (bsz, #frm*L, d), (bsz, #frm, d)
                image_feats = image_feats.unsqueeze(1)  # (bsz, 1, #frm*L, d)
            else:
                assert config.video_input.num_frames == 1, "only support single-frame"
                assert config.eval_frame_ensemble in ["mean", "max", "lse"]
                image_feats, _ = model._encode_image(image)   # (bsz, #frm, L, d), (bsz, #frm, d)

            n_clip_per_video = image_feats.shape[1]  # generate score for each clip, and aggregate all clip scores for a video
            clip_scores = []
            for clip_idx in range(n_clip_per_video):
                image_feat = image_feats[:, clip_idx]
                image_feat = tile(image_feat, 0, num_options_per_q)
                image_mask = torch.ones(
                    image_feat.size()[:-1], dtype=torch.long
                ).to(device, non_blocking=True)
                # pooled_image_feat = tile(pooled_image_feat, 0, num_options_per_q)
                # cross-modal encode
                output = model.get_text_encoder()(
                    encoder_embeds=text_feat,
                    attention_mask=text_input.attention_mask,
                    encoder_hidden_states=image_feat,
                    encoder_attention_mask=image_mask,
                    return_dict=True,
                    mode="fusion"
                )
                itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

                score = model.itm_head(itm_embeds)[:, 1]
                clip_scores.append(score)

            if len(clip_scores) == 1:
                score = clip_scores[0]
            else:
                assert config.eval_frame_ensemble in ["mean", "max", "lse"]
                clip_scores = torch.stack(clip_scores)  # (#clips, k)
                if config.eval_frame_ensemble == "mean":
                    score = clip_scores.mean(0)
                elif config.eval_frame_ensemble == "max":
                    score = clip_scores.max(0)[0]
                elif config.eval_frame_ensemble == "lse":  # LogSumExp
                    score = torch.logsumexp(clip_scores, dim=0)
                else:
                    raise ValueError("config.eval_frame_ensemble must in [mean, max, lse] when #clip > 1.")

            pred_ans = score.view(-1, num_options_per_q).max(1)[1].cpu()
            all_pred_answers.append(pred_ans)

            # assemble predictions
            ensemble_scores = score.view(-1, num_options_per_q).cpu()  # (bsz, 5)
            if n_clip_per_video > 1:
                clip_scores = clip_scores.view(n_clip_per_video, -1, num_options_per_q).cpu()  # (#clips, bsz, 5)
            for q_idx in range(len(ensemble_scores)):  # bsz
                _pred = dict(
                    video=ann["video"][q_idx],
                    options=[e[q_idx] for e in ann["caption"]],
                    answer=ann["answer"][q_idx].item(),
                    pred_ans_ensemble=pred_ans[q_idx].item(),
                    pred_scores_ensemble=ensemble_scores[q_idx].numpy(),  # (5, )
                )
                # clip scores
                if n_clip_per_video > 1:
                    _pred["pred_scores_frame"] = clip_scores[:, q_idx].numpy()  # (#clips, 5)
                    _pred["pred_ans_frame"] = clip_scores[:, q_idx].max(1)[1].numpy()  # (#clips, )
                predictions.append(_pred)

    all_gt_answers = torch.cat(all_gt_answers, 0)
    all_pred_answers = torch.cat(all_pred_answers, 0)
    acc = all_gt_answers == all_pred_answers
    acc = float(torch.sum(acc) / len(acc))
    eval_res = {"test": round(100 * acc, 2)}
    logger.info(f"\n{eval_res}")
    save_json(eval_res, join(config.output_dir, "eval_res.json"))
    torch.save(predictions, join(config.output_dir, "prediction_scores.pth"))

    dist.barrier()


if __name__ == "__main__":
    cfg = setup_main()
    main_with_ensemble(cfg)
