import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


if __name__ == "__main__":
    img = cv2.imread('E:\\work\\kesco\\file_storage\\train_data\\flame\\flame_00000007.jpg')

    ### set detectron configure
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")) # model yaml file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MIN_SIZE_TRAIN = 720
    cfg.INPUT.MAX_SIZE_TRAIN = 1280

    cfg.MODEL.WEIGHTS = "E:\\work\\kesco\\file_storage\\weights\\mask_rcnn_81_AP_98.44.pt"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81

    predictor = DefaultPredictor(cfg)


    outputs = predictor(img)


    output = outputs["instances"].to("cpu")
    wire_idxs = (output.pred_classes == 79).nonzero().squeeze()

    if wire_idxs.nelement() == 0:
        print(0)

    elif wire_idxs.nelement() == 1:
        wire_idx = wire_idxs.item()

    else:
        wire_idx = wire_idxs.numpy()[0]

    # masking
    mask = output.pred_masks[wire_idx]
    mask = np.greater(mask, 0)
    mask_img = mask.numpy()*255

    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    bit_and = np.multiply(img, mask)
    mask = bit_and
    where_0 = np.where(mask == 0)
    mask[where_0] = 255

    ret, src = cv2.threshold(mask_img, 127, 255, 0)

    # dst = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB).copy()

    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, 0, (0,0,255), 3, cv2.LINE_8, hierarchy)
    # for idx in range(len(contours)):
    #     c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hierarchy)

    cv2.imshow('src', img)
    cv2.imshow('dst', mask)

    dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB).copy()
    epsilon = 0.003*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    cv2.drawContours(dst2, [approx], 0, (255,0,0), 2, cv2.LINE_8, hierarchy)
    plt.imshow(dst2)

    list(approx) # return
