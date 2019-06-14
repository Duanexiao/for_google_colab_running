wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O /content/for_google_colab_running/BERT-Classification-Tutorial/uncased_L-12_H-768_A-12.zip
cd /content/for_google_colab_running/BERT-Classification-Tutorial;unzip uncased_L-12_H-768_A-12.zip
cd /content/for_google_colab_running/BERT-Classification-Tutorial;python run_classifier.py \
--data_dir=./data \
--vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
--output_dir=./model_output
