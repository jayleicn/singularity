import torch
import copy
from models.utils import interpolate_pos_embed, interpolate_pos_relative_bias_beit, load_temp_embed_with_mismatch
from models.tokenization_bert import BertTokenizer

from utils.scheduler import create_scheduler
from utils.optimizer import create_optimizer

import logging

logger = logging.getLogger(__name__)


def setup_model(config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
    model = model_cls(config=config, tokenizer=tokenizer)

    model = model.to(torch.device(config.device))
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters  # `False` for image-only task
        )

    optimizer = create_optimizer(config.optimizer, model)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    start_epoch = 0
    global_step = 0
    if config.pretrained_path:
        logger.info(f"Loading checkpoint from {config.pretrained_path}")
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["model"]

        if config.evaluate:
            pass
        elif config.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
        elif not pretrain:  # downstream init from pretrained ckpt
            # reshape positional embeddings
            is_beit = "beit" in config.vit_type
            if is_beit:
                # interpolate relative pos bias
                state_dict = interpolate_pos_relative_bias_beit(
                    state_dict_old=state_dict,
                    state_dict_new=model_without_ddp.state_dict(),
                    patch_shape_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.patch_shape
                )
            else:
                # interpolate pos_embed
                state_dict["vision_encoder.embeddings.position_embeddings"] = \
                    interpolate_pos_embed(
                        pos_embed_old=state_dict["vision_encoder.embeddings.position_embeddings"],
                        pos_embed_new=model_without_ddp.vision_encoder.embeddings.position_embeddings,
                        num_patches_new=model_without_ddp.vision_encoder.embeddings.patch_embeddings.num_patches
                    )

            # load temporal_embeddings, clip or expand when necessary
            state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
                temp_embed_old=state_dict["temporal_embeddings"],
                temp_embed_new=model_without_ddp.temporal_embeddings.data
            )

            for key in list(state_dict.keys()):
                if "bert" in key:
                    encoder_key = key.replace("bert.", "")
                    state_dict[encoder_key] = state_dict[key]
                    if not has_decoder:
                        del state_dict[key]

                # init text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                # only for generation tasks like VQA
                if has_decoder and "text_encoder" in key:
                    if "layer" in key:
                        encoder_keys = key.split(".")
                        layer_num = int(encoder_keys[4])
                        if layer_num < 9:  # configs/config_bert.fusion_layer
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num-9)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = ".".join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace("text_encoder", "text_decoder")
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

        # load temporal_embeddings, clip or expand when necessary
        state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
            temp_embed_old=state_dict["temporal_embeddings"],
            temp_embed_new=model_without_ddp.temporal_embeddings.data
        )

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")

    return model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step

