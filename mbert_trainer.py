# imports
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import logging
from sklearn.utils import class_weight
logging.set_verbosity_error()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device('cpu')

# class preprocess -> return X, Y


class Preprocessor():
    def __init__(self, checkpoint, token_length):
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.token_length = token_length

    def __call__(self, df):
        data_X = df.sentence.values
        data_Y = torch.tensor(df.label.values)
        encoded_data_X = {}
        encoded_data_X['input_ids'] = []
        encoded_data_X['attention_mask'] = []

        encode_batch_size = 1000
        for i in tqdm(range(0, len(data_X), encode_batch_size), desc="batches encoded"):
            output = self.tokenizer.batch_encode_plus(
                data_X[i:i+encode_batch_size],
                add_special_tokens=True,
                max_length=self.token_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            encoded_data_X['input_ids'].append(output['input_ids'])
            encoded_data_X['attention_mask'].append(output['attention_mask'])

        encoded_data_X['input_ids'] = torch.cat(
            encoded_data_X['input_ids'], dim=0)
        encoded_data_X['attention_mask'] = torch.cat(
            encoded_data_X['attention_mask'], dim=0)

        return encoded_data_X, data_Y

    def process_one(self, sentence):
        output = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.token_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return output

# class dataloader
class CustomDataLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, data_X, data_Y):
        data = TensorDataset(data_X['input_ids'],
                             data_X['attention_mask'], data_Y)
        dataloader = DataLoader(
            dataset=data, batch_size=self.batch_size, shuffle=True)
        return dataloader

# class model -> model, backward, forward, optimizer, bleh bluh


class Classifier(nn.Module):
    def __init__(self, checkpoint):
        super(Classifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        torch.cuda.empty_cache()
        self.to(DEVICE)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output

    def train(self, train_loader, hparams):
        criterion = nn.CrossEntropyLoss(weight=hparams['weights'].to(DEVICE))
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=hparams['lr'],
            betas=(hparams['beta_1'], hparams['beta_2']),
            eps=1e-08
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * hparams['epochs']
        )

        for epoch in range(hparams['epochs']):
            for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc="minibatches trained on")):
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.type(torch.LongTensor).to(DEVICE)

                # forward pass
                output = self.forward(input_ids, attention_mask)
                loss = criterion(output.logits, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()  # does back prop
                optimizer.step()
                scheduler.step()

            print(
                f'epoch {epoch+1} / {hparams["epochs"]}, loss = {loss.item():.4f}'
            )

    def test(self, test_loader):
        classified_correct = 0
        samples = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_loader, desc="minibatches processed"):
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.type(torch.LongTensor).to(DEVICE)

                output = self.forward(input_ids, attention_mask)
                predictions = torch.argmax(output.logits, dim=1)
                samples += input_ids.shape[0]
                classified_correct += (predictions ==
                                       labels).sum().item()

            accuracy = 100.00 * classified_correct / samples
            print(f'accuracy = {accuracy:.4f}')
            return accuracy

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            output = self.model(input_ids, attention_mask)
            pred = torch.argmax(output.logits, dim=1)
            return pred

    def predict_prob(self, input_ids, attention_mask):
        with torch.no_grad():
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            output = self.model(input_ids, attention_mask)
            prob = torch.softmax(output.logits, dim=1)
            return prob


def save_model(model, checkpoint, accuracy):
    PATH = 'saved_models/'
    filename = checkpoint+"-"+str(round(accuracy, 2))+".pt"
    torch.save(model.state_dict(), os.path.join(PATH, filename))


def main():

    # hyperparameters
    hparams = {}
    hparams['epochs'] = 2
    hparams['batch_size'] = 8
    hparams['lr'] = 3e-5
    hparams['beta_1'] = 0.9
    hparams['beta_2'] = 0.999
    hparams['token_length'] = 200

    checkpoint = 'bert-base-multilingual-cased'

    train_df = pd.read_csv(
        'train.csv')
    dev_df = pd.read_csv('dev.csv')
    labels = train_df.label.to_numpy()
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels)

    hparams['weights'] = torch.Tensor(weights)

    preprocessor = Preprocessor(checkpoint, hparams['token_length'])
    train_X, train_Y = preprocessor(train_df)
    dev_X, dev_Y = preprocessor(dev_df)

    dataloader = CustomDataLoader(hparams['batch_size'])
    train_loader = dataloader(train_X, train_Y)
    dev_loader = dataloader(dev_X, dev_Y)

    mbert = Classifier(checkpoint)
    mbert.train(train_loader, hparams)
    accuracy = mbert.test(dev_loader)
    save_model(mbert, checkpoint, accuracy)


if __name__ == '__main__':
    main()