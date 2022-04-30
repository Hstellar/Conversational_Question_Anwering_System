#Script downloads libraries and embeddings
# Download CoQA
mkdir -p CoQA
wget https://worksheets.codalab.org/rest/bundles/0xe3674fd34560425786f97541ec91aeb8/contents/blob/ -O CoQA/train.json
wget https://worksheets.codalab.org/rest/bundles/0xe254829ab81946198433c4da847fb485/contents/blob/ -O CoQA/dev.json

# Download GloVe
mkdir -p glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
unzip glove/glove.840B.300d.zip -d glove

#Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O glove/MT-LSTM.pth

pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Download SpaCy English language models
pip install --upgrade google-cloud-storage

