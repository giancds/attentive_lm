#!/bin/bash

echo "=== Acquiring datasets ==="
echo "---"
mkdir -p data
cd data

echo "- Downloading ptb Treebank (PTB)"
mkdir -p ptb
cd ptb
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/train.txt
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/valid.txt
wget --quiet --continue https://github.com/pytorch/examples/raw/master/word_language_model/data/penn/test.txt

mv train.txt ptb.train.txt
mv valid.txt ptb.valid.txt
mv test.txt ptb.test.txt

cd ..

echo "- Downloading WikiText-2 (WT2)"
wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip

cd ..
python3 -u pre_process_wikitext.py

cd data/wikitext-2
mv wiki.train.tokens.sents wiki2.train.txt
mv wiki.valid.tokens.sents wiki2.valid.txt
mv wiki.test.tokens.sents wiki2.test.txt

echo "---"
echo "Happy language modeling :)"
