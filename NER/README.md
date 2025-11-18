# Named Entity Recognition (NER) with BERT

This project implements a Named Entity Recognition (NER) system using **BERT (`bert-base-cased`)** for token classification. It reads CoNLL-like datasets, tokenizes the data, fine-tunes a BERT model, and evaluates it on a validation set. 

I think I didn’t do well in this task, so I plan to finish/redone it in the near future.

## Setup Instructions

1. **Clone the repository**
2. **pip install -r requirements.txt**
3. **Run the project**

## Dataset

The dataset consists of three files: train.txt, test.txt, and valid.txt. Each file is in CoNLL-like format (each line contains a token and its tag separated by a space).