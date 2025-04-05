#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_encoder.classifier import ChangeClassifier


class ChangeClassifierConfig(PretrainedConfig):
    model_type = "change_classifier"
    mm_hidden_size = 1024
    mm_vision_tower = "hf-models/clip-vit-large-patch14-336"
    mm_vision_select_layer = -2
    mm_vision_select_feature = "patch"


class ChangeClassifierModel(PreTrainedModel):
    config_class = ChangeClassifierConfig

    def __init__(self, config):
        super(ChangeClassifierModel, self).__init__(config)
        self.vision_tower = build_vision_tower(config, delay_load=True)
        self.classifier = ChangeClassifier(config.mm_hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        images: Optional[torch.FloatTensor] = None,
        change_labels: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ):
        # 计算分类损失
        assert images.ndim == 5
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.vision_tower(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(
            image_features, split_sizes, dim=0
        )  # b tuples of [2, N, L]
        image_features = torch.stack(image_features, dim=0)  # [b, 2, N, L]
        change_pred = self.classifier(image_features)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(change_pred, change_labels)

        output = dict()
        output["loss"] = loss

        return output
