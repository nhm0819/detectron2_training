'''
crop mask -> save with white background
'''
from detectron2.utils.logger import setup_logger
setup_logger()
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import glob

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ()
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (8)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = "model_final.pt"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    return predictor

def load_mask(predictor, image):
    im = cv2.imread(image)
    outputs = predictor(im)

    # v = Visualizer(im[:, :, ::-1],
    #                scale=0.5,
    #                )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # res = cv2.resize(v.get_image()[:, :, ::-1], dim, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('', v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)

    # mask = outputs["instances"].pred_masks[0].to("cpu")
    # mask = np.greater(mask, 0)  # get only non-zero positive pixels/labels
    # mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    # mask = np.concatenate((mask, mask, mask), axis=-1)
    # bit_and = np.multiply(im, mask)
    # mask_0 = bit_and
    # where_0 = np.where(mask_0 == 0)
    # mask_0[where_0] = 255
    # print(mask_0)
    # cv2.imshow('', mask_0)
    # cv2.waitKey(0)

    mask = outputs["instances"].pred_masks[0].to("cpu")
    if len(outputs["instances"].pred_classes) > 1:
        for i in range(1,len(outputs["instances"].pred_classes)):
            mask = np.bitwise_or(mask,outputs["instances"].pred_masks[i].to("cpu"))
    mask = np.greater(mask, 0)  # get only non-zero positive pixels/labels
    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    bit_and = np.multiply(im, mask)
    mask_0 = bit_and
    where_0 = np.where(mask_0 == 0)
    mask_0[where_0] = 255

    # cv2.imshow('', mask_0)
    # cv2.waitKey(0)

    return mask_0


if __name__ == '__main__':
    from tqdm import tqdm
    predictor = get_predictor()
    for idx, image in tqdm(enumerate(glob.glob('E:\\work\\kesco\\raw_data\\20211008\\bad_data\\*\\*.jpg'))):
    # image = '/home/sym/Downloads/wire_data_0730/wire_data/flame/f_test/c1_16-4.jpg'
        try:
            mask = load_mask(predictor,image)
            image_path = image.replace("bad_data", "segmented_bad_data")
            cv2.imwrite(image_path, mask)
        except:
            print("error index :", idx)
            continue

    # image = '/home/sym/Downloads/wire_data_0730/wire_data/flame/f_field/330-6.jpg'
    # mask = load_mask(predictor, image)
    # filename = image.split('/')[-1]
    # cv2.imwrite(os.path.join('/home/sym/Downloads/wire_data_0730/wire_data_mask/flame/f_field',filename), mask)



