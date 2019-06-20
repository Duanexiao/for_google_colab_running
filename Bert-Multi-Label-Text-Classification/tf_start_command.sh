set -e
pip install tensorflow_hub
pip install bert-tensorflow
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O ./uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
mkdir Downloads
cp ./pybert/dataset/raw/train.csv ./Downloads
cp ./pybert/dataset/raw/test.csv ./Downloads
mkdir -p working/output
python tf_bert_multi_label_classification.py
