import requests
import uuid
import random
import time


def download_yunsuo_captcha():
    url = "http://vip.yunsuo.com.cn/seccode/makecode"
    r = requests.get(url)
    img = r.content
    with open("/Users/duanexiao/Downloads/yunsuo_captcha/yunsuo_{0}.png".format(str(uuid.uuid4())), "wb") as f:
        f.write(img)


if __name__ == "__main__":
    generation_size = 100
    for _ in range(generation_size):
        download_yunsuo_captcha()
        time.sleep(random.random())
