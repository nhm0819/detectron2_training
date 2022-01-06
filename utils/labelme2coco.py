import os
import json
import numpy as np
import glob
import PIL.Image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json", thing_classes=None):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        if isinstance(thing_classes, list):
            self.label = thing_classes
        else:
            self.label = []

        self.save_json()


    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.height = data["imageHeight"]
                self.width = data["imageWidth"]

                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]

                    try:
                        self.annotations.append(self.annotation(points, label, num))
                        self.annID += 1
                    except:
                        pass


        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        # img = utils.img_b64_to_arr(data["imageData"])
        # height, width = img.shape[:2]
        img = None
        image["height"] = self.height
        image["width"] = self.width
        image["id"] = num
        image["file_name"] = data["imagePath"] # .split("/")[-1]
        # image["file_name"] = data["imagePath"].replace('\\', '/') # .split("/")[-1]

        # self.height = height
        # self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)
        category["name"] = label
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points, dtype=np.int32).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))
        annotation["category_id"] = label  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco


    def save_json(self):
        print("saving coco json...")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4, cls=NpEncoder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "--images",
        help="Directory to labelme images and annotation json files.",
        default="E:\\work\\kesco\\raw_data\\file_storage",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="E:\\work\\kesco\\train.json"
    )
    args = parser.parse_args()

    # labelme_json = glob.glob(os.path.join(args.images, "*.json"))
    labelme_json = glob.glob("E:\\work\\kesco\\raw_data\\file_storage\\test_data\\json\\*.json")

    labelme2coco(labelme_json, args.output)
