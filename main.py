import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from download_datasets import *

VOCAB_PATH = 'vocab_to_idx.json'

class Data:
    def __init__(self, kaggle_username: str, kaggle_key: str):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key

    def _process_datasets(self) -> pd.DataFrame:
        (
            first_df, 
            second_df, 
            third_df, 
            spam_sms_df
        ) = download_and_load_data(self.kaggle_username, self.kaggle_key)
        target_classes = ['spam', 'spamobscene', 'hate', 'hateobscene', 'obscene']
        spam_first_df = first_df[first_df['tclass'].isin(target_classes)]
        spam_second_df = second_df[second_df['tclass'].isin(target_classes)]
        spam_sms_df = spam_sms_df.replace(r'\t+', '', regex=True)

        target_classes = ['clean']
        clean_first_df = first_df[first_df['tclass'].isin(target_classes)]
        clean_second_df = second_df[second_df['tclass'].isin(target_classes)]

        # https://github.com/Melanee-Melanee/HamSpam-EMAIL/blob/main/src/HamSpam.ipynb
        labels = []
        for email in third_df[0]:
            labels.append(email.split(',')[-1])
        labels.remove(labels[0])

        text = third_df[0].copy()
        for i in range(len(text)):
            if i == 0:
                continue
            text[i] = text[i].split(',')[:-1][0]
        text.pop(0)
        processed = text.str.replace(r'[^\w\d\s]', ' ',regex=True)
        processed = processed.str.replace(r'\s+', ' ',regex=True)
        processed = processed.str.replace(r'^\s+|\s+?$', '',regex=True)
        processed = processed.str.lower()

        third_df = pd.DataFrame({"comment_normalized": processed, "labels": labels})
        spam_third_df = third_df[third_df['labels'].isin(['spam'])]
        clean_third_df = third_df[third_df['labels'].isin(['ham'])]

        spam_df = pd.concat(
            [spam_first_df[['comment_normalized']], spam_second_df[['comment_normalized']], spam_third_df[['comment_normalized']], spam_sms_df[['comment_normalized']]],
            ignore_index=True
        )
        clean_df = pd.concat(
            [clean_first_df[['comment_normalized']], clean_second_df[['comment_normalized']], clean_third_df[['comment_normalized']]],
                ignore_index=True
        )

        spam_df = spam_df.drop_duplicates(subset=['comment_normalized'])
        clean_df = clean_df.drop_duplicates(subset=['comment_normalized'])
        labels = []
        for _ in range(len(spam_df)): labels.append('spam')
        for _ in range(len(clean_df)): labels.append('clean')
        combined_df = pd.DataFrame(
            {
                "comment_normalized": spam_df["comment_normalized"].tolist() + clean_df["comment_normalized"].tolist(),
                "labels": labels
            }
        )
        return combined_df

class SpamLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, output_dim=1, n_layers=2):
            super(SpamLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.5
            )
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)

        def forward(self, text, text_lengths=None):
            embedded = self.dropout(self.embedding(text))
            output, (hidden, cell) = self.lstm(embedded)
            hidden_forward = hidden[-2,:,:]
            hidden_backward = hidden[-1,:,:]
            combined = torch.cat((hidden_forward, hidden_backward), dim=1)
            return self.sigmoid(self.fc(combined))    

class SpamDataset(Dataset):
        def __init__(
                    self, 
                    texts: np.array, 
                    labels: np.array, 
                    word_to_idx: dict,
                ):
            self.texts = texts
            self.labels = labels
            self.word_to_idx = word_to_idx
            self.label_map = {
                'spam': 1,
                'clean': 0
            }

        def __len__(self):
            return len(self.texts)
        
        def encode_text(self, text: str):
            tokens = text.split()
            vec = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in tokens]
            return torch.tensor(vec, dtype=torch.long)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.label_map[self.labels[idx]]
            token_ids = self.encode_text(text)
            return {
                'input': torch.tensor(token_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }


class TrainAndEvaluate:
    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_description(f"Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self, model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in dataloader:
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total
    

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            vocab_size: int,
            embed_dim: int,
            hidden_dim: int,
            DEVICE: str = 'cuda',
            EPOCHS: int = 50,
        ) -> nn.Module:
        LR = 0.001
        use_fused = True if DEVICE == 'cuda' else False

        model = SpamLSTM(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim
        ).to(DEVICE)
        criterion = torch.nn.BCELoss()
        optimizer = optim.AdamW(
                        model.parameters(), 
                        lr=LR, 
                        betas=(0.9, 0.95), 
                        weight_decay=1e-3, 
                        eps=1e-6, 
                        fused=use_fused
                    )
        best_val_acc = 0

        for epoch in range(EPOCHS):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            print("-" * 30)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_spam_model.pth')
                print("-> Best model saved!")
        return model
    
def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    labels = [item['label'] for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_stacked = torch.stack(labels)
    
    return {
        'input': inputs_padded,
        'label': labels_stacked
    }


if __name__ == "__main__":
    BATCH_SIZE = 64
    MAX_LEN = 50
    EPOCHS = 50
    EMBED_DIM = 100
    HIDDEN_DIM = 64
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    load_model = True
    chunk_size = 10 # for inference
    map_ = {
            0: "clean",
            1: "spam"
        }
    
    if not load_model or not os.path.exists(VOCAB_PATH):
        kaggle_username = None
        kaggle_key = None
        if not os.path.exists(os.path.join(DATASET_FOLDER, 'sms_data.txt')):
            kaggle_username = str(input('Enter your kaggle username: '))
            kaggle_key = str(input('Enter your kaggle key: '))
        data = Data(kaggle_username, kaggle_key)
        combined_df = data._process_datasets()
        
        if not os.path.exists(VOCAB_PATH):
            all_text = " ".join(combined_df['comment_normalized'].astype(str).tolist())
            words = all_text.split()
            counter = Counter(words)
            vocab = sorted(counter, key=counter.get, reverse=True)
            word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
            word_to_idx['<PAD>'] = 0
            word_to_idx['<UNK>'] = len(word_to_idx)
            with open(VOCAB_PATH, mode='w', encoding='utf-8') as f:
                json.dump(
                    word_to_idx,
                    f,
                    indent=4
                )
        else:
            with open(VOCAB_PATH, encoding='utf-8') as f:
                word_to_idx = json.loads(f.read())
    else:
        with open(VOCAB_PATH, encoding='utf-8') as f:
            word_to_idx = json.loads(f.read())

    if load_model:
        def encode_text(text):
            tokens = text.split()
            vec = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in tokens]
            return torch.tensor(vec, dtype=torch.long)
        model = SpamLSTM(
            vocab_size=len(word_to_idx), 
            embed_dim=EMBED_DIM, 
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)
        model.load_state_dict(torch.load('best_spam_model.pth', map_location=DEVICE))
        model.eval()

        with torch.inference_mode():
            while True:
                text = str(input("Enter persian spam/clean text: "))
                encoded = encode_text(text)
                x = encoded.to(DEVICE)
                token_chunks = [x[idx: idx + chunk_size] for idx in range(0, x.shape[0], chunk_size)]
                results = {
                    0: 0,
                    1: 0
                }
                for chunk in token_chunks:
                    x = chunk.unsqueeze(0)
                    outputs = model(x)
                    predicted = (outputs.squeeze() > 0.6).float()
                    results[predicted.item()] += 1
                predicted = 0 if results[0] > results[1] else 1
                print(map_[predicted])
    else:
        X = combined_df['comment_normalized'].values
        y = combined_df['labels'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = SpamDataset(X_train, y_train, word_to_idx)
        val_dataset = SpamDataset(X_val, y_val, word_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        train_and_evaluate = TrainAndEvaluate()

        model = train_and_evaluate.train(
                            train_loader, 
                            val_loader, 
                            len(word_to_idx), 
                            EMBED_DIM,
                            HIDDEN_DIM,
                            DEVICE, 
                            EPOCHS
                        )
        