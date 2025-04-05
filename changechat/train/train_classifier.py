import os
from dataclasses import dataclass, field
import json, logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from changechat.train.changechat_trainer import ChangeClassifierTrainer
from changechat.model import ChangeClassifierModel, ChangeClassifierConfig
from PIL import Image
from torchvision import transforms
import datetime

RAND_AUG_F = transforms.Compose(
    [
        transforms.RandomCrop(size=(252, 252)),  # 随机裁剪为252x252的图像
        transforms.ColorJitter(brightness=0.35, contrast=0.35),  # 随机调整亮度和对比度
    ]
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_classifier_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # Only save Adapter
    keys_to_match = ["classifier"]
    weight_to_save = get_classifier_state_maybe_zero_3(
        trainer.model.named_parameters(), keys_to_match
    )
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "classifier")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"classifier.bin"))
    return


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        assert "image" in sources[0]

        image_file = self.list_data_dict[i]["image"]
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor

        if isinstance(image_file, str):
            image_file_list = [image_file]
        elif isinstance(image_file, list):
            image_file_list = image_file
        else:
            raise NotImplementedError

        imageList = []
        for _image_file in image_file_list:
            image = Image.open(
                (os.path.join(image_folder, _image_file)).strip()
            ).convert("RGB")
            image = RAND_AUG_F(image)
            image = processor.preprocess(
                image,
                do_resize=True,
                crop_size={"height": 252, "width": 252},
                size={"shortest_edge": 252},
                return_tensors="pt",
            )["pixel_values"][0]
            imageList.append(image)

        data_dict = {}
        data_dict["image"] = torch.stack(imageList, dim=0)  # (2, c, h, w)

        assert "changeflag" in self.list_data_dict[i]
        data_dict["change_labels"] = torch.tensor(self.list_data_dict[i]["changeflag"])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        images = [instance["image"] for instance in instances]
        # 确保所有图像的尺寸相同
        assert all(x is not None and x.shape == images[0].shape for x in images)
        batch["images"] = torch.stack(images)
        batch["change_labels"] = torch.stack(
            [instance["change_labels"] for instance in instances]
        )

        return batch


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args.output_dir = training_args.output_dir + "_" + now

    local_rank = training_args.local_rank

    config = ChangeClassifierConfig()
    model = ChangeClassifierModel(config)

    # freeze vison_tower
    model.vision_tower.requires_grad_(False)
    data_args.image_processor = model.vision_tower.image_processor

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_train_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    rank0_print(
        f"Trainable params: {trainable_params}, non trainable params: {non_train_params}"
    )

    data_module = make_supervised_data_module(data_args=data_args)
    trainer = ChangeClassifierTrainer(model=model, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
