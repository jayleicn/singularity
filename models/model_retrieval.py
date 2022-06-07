from models.model_retrieval_base import SingularityRetrievalBase


class Singularity(SingularityRetrievalBase):
    def __init__(self, config=None, tokenizer=None):
        super(Singularity, self).__init__(
            config=config, tokenizer=tokenizer, pretrain=False
        )
