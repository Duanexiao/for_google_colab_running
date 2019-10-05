import os
import uuid
import glob
from random import randint
from captcha.image import ImageCaptcha
from character import *


generation_size = 20000
captcha_content = SIMPLE_CHAR_SET["ALPHANUMERIC"]
captcha_length = 4
captcha_size = {"height": 50, "width": 150}
data_save_dir = "./dataset/train_raw"
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
generator = ImageCaptcha(**captcha_size)
while len(glob.glob(data_save_dir + "/*.png")) < generation_size:
    print("{0} pic left".format(generation_size - len(glob.glob(data_save_dir+"/*.png"))))
    chars = [captcha_content[randint(0, len(captcha_content) - 1)]\
             for _ in range(captcha_length)]
    img = generator.generate_image(chars)
    img.save(data_save_dir + "/" + "".join(chars)+ "_" + str(uuid.uuid4()) +".png")
