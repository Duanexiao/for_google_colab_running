set -e
pip install captcha
pip install tensorflow-gpu==1.14
git clone https://github.com/Duanexiao/for_google_colab_running.git
cd for_google_colab_running/captcha_trainer
python data_generation.py
python make_dataset.py
python trains.py
