import os
import json
import glob
import cv2

files1 = glob.glob("E:\\work\\kesco\\file_storage\\train_data\\json\\*.json")
files2 = glob.glob("E:\\work\\kesco\\file_storage\\test_data\\json\\*.json")

files = files1 + files2

file = files[0]
for idx, file in enumerate(files):
    file_name = file.split('\\')[-1]

    with open(file, "r") as f:
        json_data = json.load(f)
        # image_path = f"../segmentation/{json_data['imagePath']}"
        image_path = f"backup\\{file_name}".replace("json", "jpg")
        # image_path = os.path.join('..', 'segmentation', json_data['imagePath'])
        json_data['imagePath'] = image_path

        # json_data.pop('imageData')

    if file.split('\\')[-3] == 'train_data':
        new_json = f"E:\\work\\kesco\\file_storage\\train_data\\new_json\\{file_name}"
        with open(new_json, 'w') as file:
            json.dump(json_data, file, indent=4)

    elif file.split('\\')[-3] == 'test_data':
        new_json = f"E:\\work\\kesco\\file_storage\\test_data\\new_json\\{file_name}"
        with open(new_json, 'w') as file:
            json.dump(json_data, file, indent=4)

import shutil
files = glob.glob("E:\\work\\kesco\\raw_data\\file_storage\\segmentation_data\\test\\*.jpg")
for file in files:
    file_name = file.split("\\")[-1]
    save_path = f"E:\\work\\kesco\\raw_data\\file_storage\\test_data\\segmentation\\{file_name}"
    shutil.copy2(file, save_path)

import json
with open("train.json", "r") as f:
    json_data = json.load(f)


folder = os.listdir('/home/sym/Desktop/인수인계/kesco/raw_data/Labeling2010824')

n = 0
for i in folder:
    n+=1
    path1 = '/home/sym/Desktop/인수인계/kesco/raw_data/Labeling2010824/'+ i + '/*.json'
    for f_name in glob(path1):
        new_file_name = f_name.split('/')[-1].split('.')[0]
        with open(f_name, "r") as fp:
            json_data = json.load(fp)
            new_name = str(n)+'_' + new_file_name + 'f.jpg'
            json_data['imagePath'] = new_name
            json_data['shapes'][0]['label'] = 'wire'
        new_json = '/home/sym/Desktop/인수인계/kesco/raw_data/new/'+str(n)+'_' + new_file_name + 'f.json'
        with open(new_json, 'w') as file:
            json.dump(json_data, file, indent=1)
        # print(new_file_name)
    # print("============================================")
    path2 = '/home/sym/Desktop/인수인계/kesco/raw_data/Labeling2010824/' + i + '/*.jpg'
    for f_name in glob(path2):
        new_file_name = f_name.split('/')[-1].split('.')[0]
        new_name = '/home/sym/Desktop/인수인계/kesco/raw_data/new/'+str(n)+'_' + new_file_name + 'f.jpg'
        img = cv2.imread(f_name)
        cv2.imwrite(new_name,img)
        # print(new_name)


for f_name in glob('/home/sym/Desktop/인수인계/kesco/raw_data/new/*.json'):
    file_name = f_name.split('/')[-1].split('.')[0]
    file_name = file_name + '.jpg'
    with open(f_name, "r") as fp:
        json_data = json.load(fp)
        if file_name == json_data['imagePath']:
            continue
        else:
            print(file_name, json_data['imagePath'])