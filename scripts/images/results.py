import shutil
import os

# copy images from ./../../results/images to ./../../paper/images
src = "./../../results/images"
dst = "./../../paper/figures/images"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)