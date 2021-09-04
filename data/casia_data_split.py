import os
import shutil


im_dir = "/home/anh/Downloads/CASIA-WebFace"
start_val = 5015107

train_path = os.path.join("casia", "train")
val_path = os.path.join("casia", "val")
os.makedirs(val_path)
shutil.move(im_dir, train_path)
vids = [x for x in os.listdir(train_path) if int(x) >= start_val]
for vid in vids:
    shutil.move(os.path.join(train_path, vid), val_path)
