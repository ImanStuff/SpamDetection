### Datasets

#### 1. Persian SMS Dataset
A comprehensive dataset of Persian SMS messages labeled as spam and ham.
- **Source (GitHub):** [semnan-university-ai/persian-sms-dataset](https://github.com/semnan-university-ai/persian-sms-dataset)
- **Source (Kaggle):** [Persian SMS Dataset by Amirshnll](https://www.kaggle.com/datasets/amirshnll/persian-sms-dataset)

#### 2. PHICAD (Persian Health Insurance Coverage Analysis Dataset)
- **Source:** [davardoust/PHICAD](https://github.com/davardoust/PHICAD)

#### 3. HamSpam Email Dataset
A collected dataset for email spam detection tasks.
- **Source:** [Melanee-Melanee/HamSpam-EMAIL](https://github.com/Melanee-Melanee/HamSpam-EMAIL)


To run the code, simply install the requirements and run `main.py`.
```shell
pip install -r requirements.txt
python main.py
```

### Updates
- Increased vocabulary size from 20000 to about 156000 tokens.
- Added mechanism to load data if they are exist.
- Saving vocab indexes.
- Changed optimizer to AdamW, better training achieved for higher number of tokens.
- Previous work discards the majority of large text information ( Problem in encode_text max_len parameter.). Fixed this by ecoding the whole texts, and padding each batch to zero ( Pytorch collate_fn for DataLoader ).
- Achieved Train accuracy: 92.31% and Validation accuracy: 89.93%