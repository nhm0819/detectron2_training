from detectron2.engine import DefaultTrainer, PeriodicWriter
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.utils.events import (CommonMetricPrinter, JSONWriter,
                                     TensorboardXWriter, get_event_storage)

from LossEvalHook import LossEvalHook
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from wandb_log import WandB_Printer

import torch
import os
import copy

# def custom_mapper(dataset_dict):
#     # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     # T.Resize((800,800)),
#     transform_list = [
#         # T.Resize((720, 1280)),
#         T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 720), max_size=1280,
#                              sample_style='choice'),
#         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#         T.RandomBrightness(0.8, 1.2),
#         T.RandomSaturation(0.8, 1.2),
#         T.RandomContrast(0.8, 1.2)
#     ]
#
#     image, transforms = T.apply_transform_gens(transform_list, image)
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#
#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True,
                                                                      augmentations=[
                                                                          T.ResizeShortestEdge(
                                                                              short_edge_length=(640, 672, 704, 720),
                                                                              max_size=1280,
                                                                              sample_style='choice'),
                                                                          T.RandomFlip(prob=0.5, horizontal=False,
                                                                                       vertical=True),
                                                                          T.RandomFlip(prob=0.5, horizontal=True,
                                                                                       vertical=False),
                                                                          T.RandomBrightness(0.8, 1.2),
                                                                          T.RandomSaturation(0.8, 1.2),
                                                                          T.RandomContrast(0.8, 1.2)
                                                                      ]))

    # @classmethod
    # def build_test_loader(cls, cfg):
    #     return build_detection_test_loader(cfg, mapper=DatasetMapper(cfg, False, augmentations=[T.Resize((720,1280))]))

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        # hooks.append(PeriodicWriter(
        #     [WandB_Printer(name=self.cfg.wandb_name, project=self.cfg.wandb_project, cfg=self.cfg)], period=1))
        # writerList = [
        #     CustomMetricPrinter(self.showTQDM, self.cfg.SOLVER.MAX_ITER),
        #     JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        #     TensorboardXWriter(self.cfg.OUTPUT_DIR),
        #     WandB_Printer(name = self.cfg.OUTPUT_DIR.split("/")[1], project="object-detection",entity="cv4")
        # ]


        return hooks

    def build_writers(self):
        """Extends default writers with a Wandb writer if Wandb logging was enabled.

        See `d2.engine.DefaultTrainer.build_writers` for more details.
        """

        writers = [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

        if self.cfg.wandb_project:
            writers.append(WandB_Printer())

        return writers

