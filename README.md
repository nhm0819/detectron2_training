# detectron2_training

- custom trainer
- validation loss
- custom mapper (augmentation)
- segmentation autolabelling (use mask rcnn)
- wandb
- plotting
- default_argument_parser --> detectron 전용 argparse (multi gpu 에 대한 인자들)

- multi gpu 사용 가능 : train.py --num-gpus 2 
```
docker pull pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
docker build -t detectron2:latest .
docker run -it -v $volume:$volume --gpus all --name detectron2 --shm-size=8G detectron2:latest /bin/bash
```
