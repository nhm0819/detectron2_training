from detectron2.utils.logger import setup_logger
setup_logger() # Setup detectron2 logger

import os
import glob
import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from trainer.MyTrainer import MyTrainer
from utils.labelme2coco import labelme2coco

import wandb, yaml



parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--file_storage_path', type=str, default="E:\\work\\kesco\\file_storage", metavar='S',
                    help='file storage path')
parser.add_argument('--pretrained_path', type=str, default="E:\\work\\kesco\\file_storage\\weights\\mask_rcnn_AP_98.48.pt", metavar='S',
                    help='ptretrained model path')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='num workers')

parser.add_argument('--checkpoint', type=int, default=2, metavar='N',
                    help='eval period')

parser.add_argument('--wandb_project', type=str, default="kesco_mask", metavar='S',
                    help='wandb project')
parser.add_argument('--wandb_name', type=int, default='1', metavar='N',
                    help='wandb name')

parser.add_argument('--train_json', type=str, default="E:\\work\\kesco\\code\\autolabeling+maskrcnn\\output\\train.json", metavar='S',
                    help='train json path')
parser.add_argument('--test_json', type=str, default="E:\\work\\kesco\\code\\autolabeling+maskrcnn\\output\\test.json", metavar='S',
                    help='test json path')


def train(args):

    ### collect json files
    # train
    train_json_folder = os.path.join(args.file_storage_path, "train_data", "json")
    train_jsons = glob.glob(os.path.join(train_json_folder, "*.json"))

    if os.path.isfile(args.train_json):
        train_collect = args.train_json
    else:
        try:
            train_collect = os.path.join(args.output_dir, "train.json")
            labelme2coco(train_jsons, train_collect)
        except:
            raise ValueError(f"Check the json folder path")

    # register json file for detectron
    register_coco_instances("train", {}, train_collect, args.train_path)  # train_collect must be coco dataset style
    thing_classes = MetadataCatalog.get("train").thing_classes.copy()

    # test (same with train)
    test_json_folder = os.path.join(args.file_storage_path, "test_data", "json")
    test_jsons = glob.glob(os.path.join(test_json_folder, "*.json"))

    if os.path.isfile(args.test_json):
        test_collect = args.test_json
    else:
        test_collect = os.path.join(args.output_dir, "test.json")
        labelme2coco(test_jsons, test_collect, thing_classes=thing_classes)

    register_coco_instances("test", {}, test_collect, args.test_path)


    ### set detectron configure
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")) # model yaml file
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ("test", )
    cfg.TEST.EVAL_PERIOD = args.checkpoint
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    try:
        cfg.MODEL.WEIGHTS = args.pretrained_path
    except:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # training initialize from model zoo


    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.BASE_LR_END = 1e-6   # use with WarmupCosineLr
    cfg.SOLVER.GAMMA = 0.5          # use with WarmupMultiStepLR
    cfg.SOLVER.STEPS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000] # use with WarmupMultiStepLR
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR" # WarmupCosineLR

    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint # validation period
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # num classes = 80+1 (coco dataset + wire)
    cfg.OUTPUT_DIR = args.output_dir # cfg.OUTPUT_DIR = f"./output/output_{args.output_num+1}"

    # resize default func : resize shortest edge
    cfg.INPUT.MIN_SIZE_TRAIN = [640, 672, 704, 720]
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1280


    # wandb
    cfg.wandb_project = args.wandb_project
    cfg.wandb_name = args.wandb_name
    wandb.login()
    cfg_wandb = yaml.safe_load(cfg.dump())
    wandb.init(project=args.wandb_project, name=str(args.wandb_name), config=cfg_wandb, sync_tensorboard=True)


    # train
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



if __name__ == '__main__':
    args = parser.parse_args()
    args.train_path = os.path.join(args.file_storage_path, "train_data")
    args.test_path = os.path.join(args.file_storage_path, "test_data")

    # output number
    output_folders = glob.glob("./output/output_*")
    output_num = 0
    for output_folder in output_folders:
        output_num = int(output_folder.split('_')[-1]) if int(output_folder.split('_')[-1]) > output_num else output_num
    args.output_num = output_num
    args.output_dir = os.path.join(os.getcwd(), "output", f"output_{args.output_num+1}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("---------Mask RCNN Training---------")
    #try:
    train(args)
    # except:
    #     print("RCNN Training Error")
    #     shutil.rmtree(args.output_dir)
    #       raise

