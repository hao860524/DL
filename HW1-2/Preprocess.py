import os
import pandas as pd
import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)


def face_crop(csv_data):
    # print(csv_data.split('/')[1])
    set = {'name': [],
           'image': [],
           'label': []}
    data = pd.read_csv(csv_data)
    img_name = ''
    for index, face in data.iterrows():
        if img_name != face['filename']:
            img_name = face['filename']

        path = os.path.join(csv_data.split('/')[1], 'images', img_name)

        try:
            img = Image.open(path).convert("RGB")
        except OSError as e:  # work on python 3.x
            print(str(e))
            continue

        img = img.crop(box=(face['xmin'], face['ymin'], face['xmax'], face['ymax']))
        img = np.array(img.resize(IMG_SIZE).getdata())
        if not img.shape[1] == 3:
            img = img[:, 3]
        if index % 100 == 0:
            print('Loading {} image '.format(csv_data) + str(index), img.shape)
        # img = np.swapaxes(img, 0, 1).reshape(3, IMG_SIZE[1], IMG_SIZE[0]) / 255
        # img = torch.FloatTensor(img)

        set['name'].append(img_name)
        set['image'].append(img)
        set['label'].append(face['label'])

        # path = csv_data.split('.')[-2].split('/')[-1]
        # os.makedirs(path, exist_ok=True)
        # img.save(os.path.join(path, 'train_' + str(index).zfill(4) + '.png'))
    return set
