
from modeling_t5 import VLT5

from vqa_model import VLT5VQA
from gqa_model import VLT5GQA
from nlvr_model import VLT5NLVR
from caption_model import VLT5COCOCaption

class VLT5MultiTask(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.train_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.train_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.train_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.train_step(self, batch, **kwargs)

    def valid_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.valid_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.valid_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.valid_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.valid_step(self, batch, **kwargs)

    def test_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.test_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.test_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.test_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.test_step(self, batch, **kwargs)


from modeling_bart import VLBart

from vqa_model import VLBartVQA
from gqa_model import VLBartGQA
from nlvr_model import VLBartNLVR
from caption_model import VLBartCOCOCaption

class VLBartMultiTask(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.train_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.train_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.train_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.train_step(self, batch, **kwargs)

    def valid_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.valid_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.valid_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.valid_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.valid_step(self, batch, **kwargs)

    def test_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.test_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.test_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.test_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.test_step(self, batch, **kwargs)