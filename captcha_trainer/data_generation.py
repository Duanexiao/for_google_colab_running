import os
import uuid
import glob
from random import randint
import multiprocessing as mp
from captcha.image import ImageCaptcha
from character import *


def generate_captcha(sequence):
    generator = ImageCaptcha(**captcha_size)
    chars = [captcha_content[randint(0, len(captcha_content) - 1)]\
             for _ in range(captcha_length)]
    img = generator.generate_image(chars)
    img.save(data_save_dir + "/" + "".join(chars)+ "_" + str(uuid.uuid4()) +".png")


if __name__ == "__main__":
    generation_size = 100000
    captcha_content = SIMPLE_CHAR_SET["ALPHANUMERIC"]
    captcha_length = 4
    captcha_size = {"height": 50, "width": 150}
    data_save_dir = "./dataset/train_raw"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    pool = mp.Pool(processes=10)
    for i in range(generation_size):
        pool.apply(generate_captcha, args=(i,))
    pool.close()
    pool.join()
