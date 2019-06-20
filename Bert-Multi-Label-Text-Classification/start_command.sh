set -e
pip install pytorch-pretrained-bert 
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O pybert/model/pretrain/uncased_L-12_H-768_A-12.zip
cd pybert/model/pretrain
rm -rf uncased_L-12_H-768_A-12
unzip uncased_L-12_H-768_A-12.zip
cd ../../../
python convert_tf_checkpoint_to_pytorch.py
python train_bert_multi_label.py
python inference.py
