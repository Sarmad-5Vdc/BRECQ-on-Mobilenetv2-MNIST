from PIL import Image
import os

# path = '/coco/train2014/COCO_train2014_000000437282.jpg'
# path = 'E:/Sarmad/BRECQ-main/coco/train2014/COCO_train2014_000000000491.jpg'
path = 'train2014/COCO_train2014_000000000508.jpg'

b = Image.open(os.path.join('coco/', path)).convert("RGB")
# a = Image.open(path).convert("RGB")
# b.show()