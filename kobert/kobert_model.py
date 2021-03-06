import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
import random
import time
import datetime
from tqdm import tqdm
import pandas as pd
import os

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def kobert_train(train, column, num_classes, max_len, batch_size, epochs, model_name):
    train = train[train[column].isnull() == False]
    train = train.drop_duplicates([column])

    train['score'] = train['score'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

    sentences = train[column]
    labels = train['score'].values

    # train test split
    from sklearn.model_selection import train_test_split
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(sentences, labels, random_state=42, test_size=0.3)

    train_inputs = pd.DataFrame(train_inputs)
    train_inputs = train_inputs.reset_index(drop=True)
    train_labels = pd.DataFrame(np.array(train_labels))
    train = pd.concat([train_inputs, train_labels], axis = 1, join='inner')
    train = train.reset_index(drop=True)
    train.columns = ['preprocessed_review', 'score']

    validation_inputs = pd.DataFrame(validation_inputs)
    validation_inputs = validation_inputs.reset_index(drop=True)
    validation_labels = pd.DataFrame(np.array(validation_labels))
    validation = pd.concat([validation_inputs, validation_labels], axis = 1, join='inner')
    validation = validation.reset_index(drop=True)
    validation.columns = ['preprocessed_review', 'score']

    bertmodel, vocab = get_pytorch_kobert_model()

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_train = BERTDataset(train, 'preprocessed_review', 'score', tok, max_len, True, False)
    data_validation = BERTDataset(validation, 'preprocessed_review', 'score', tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=8)  # num_workers = (gpu ??????) * 4
    validation_dataloader = torch.utils.data.DataLoader(data_validation, batch_size=batch_size, num_workers=8)

    # ???????????? ??????
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    model = BERTClassifier(bertmodel, num_classes=num_classes, dr_rate=0.5)
    model = torch.nn.DataParallel(model)
    # torch.cuda.set_device('cuda:0')
    model.cpu()

    from transformers import get_linear_schedule_with_warmup
    # ??????????????? ??????
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # ????????? 5e-5
                      eps=1e-8  # 0?????? ????????? ?????? ???????????? ?????? epsilon ???
                      )


    # ??? ?????? ?????? : ???????????? ?????? * ??????
    total_steps = len(train_dataloader) * epochs

    # ????????? ???????????? ????????? ??????????????? ???????????? ??????
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    loss_fn = nn.CrossEntropyLoss()



    # ????????? ?????? ???????????? ??????
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    log_interval = 200

    # ??????????????? ?????????
    model.zero_grad()

    # ???????????? ??????
    for epoch_i in range(0, epochs):
        train_acc = 0.0
        val_acc = 0.0

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # ?????? ?????? ??????
        t0 = time.time()

        # ?????? ?????????
        total_loss = 0

        # ??????????????? ??????
        model.train()

        # ????????????????????? ???????????? ???????????? ?????????
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            # Forward ??????
            outputs = model(token_ids, valid_length, segment_ids)

            # ?????? ??????
            loss = loss_fn(outputs, label)

            # ??? ?????? ??????
            total_loss += loss.item()

            # Backward ???????????? ??????????????? ??????
            loss.backward()

            # ??????????????? ?????????
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # ?????????????????? ?????? ????????? ???????????? ????????????
            optimizer.step()

            # ??????????????? ????????? ??????
            scheduler.step()

            # ??????????????? ?????????
            model.zero_grad()

            train_acc += calc_accuracy(outputs, label)

            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {:.2f} train acc {}".format(epoch_i + 1, batch_id + 1,
                                                                         loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))

        # ?????? ?????? ??????
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        print("  epoch {} train acc {:.2f}".format(epoch_i + 1, train_acc / (batch_id + 1)))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        # ?????? ?????? ??????
        t0 = time.time()

        # ??????????????? ??????
        model.eval()

        # ?????? ?????????
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        total_loss = 0
        # ????????????????????? ???????????? ???????????? ?????????
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(validation_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            # ??????????????? ?????? ??????
            with torch.no_grad():
                outputs = model(token_ids, valid_length, segment_ids)

            val_acc += calc_accuracy(outputs, label)
            loss = loss_fn(outputs, label)
            total_loss += loss.item()

        avg_val_loss = total_loss / len(validation_dataloader)

        print("  epoch {} val acc {:.2f}".format(epoch_i + 1, val_acc / (batch_id + 1)))
        print("  epoch {} val loss {:.2f}".format(epoch_i + 1, avg_val_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    torch.save(model, 'results/' + model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M') + '.pt')
    print("")
    print("Training complete!")


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([row[sent_idx]]) for i, row in dataset.iterrows()]
        self.labels = [np.int32(row[label_idx]) for i, row in dataset.iterrows()]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=None,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# ????????? ?????? ??????
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

# ?????? ?????? ??????
def format_time(elapsed):
    # ?????????
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss?????? ?????? ??????
    return str(datetime.timedelta(seconds=elapsed_rounded))

# ----------------------------------------------------------------------------------------------------------------------
# predict
def kobert_predict(train, model_name):
    data_path = './weights'
    model = torch.load(os.path.join(data_path, model_name+'.pt'))

    train = train[train['review'].isnull() == False]

    pred_score = test_sentences(model, train)
    train['o2o_score'] = pred_score

    return train

# ?????? ?????????
def test_sentences(model, df):
    device = torch.device("cpu")

    model.eval()

    result = []

    test_dataloader = convert_input_data(df)
    for token_ids, valid_length, segment_ids in test_dataloader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length

        # ??????????????? ?????? ??????
        with torch.no_grad():
            # Forward ??????
            outputs = model(token_ids, valid_length, segment_ids)

        for output in outputs:
            # CPU??? ????????? ??????
            logits = output.detach().cpu().numpy()
            result.append(np.argmax(logits).astype(int)+1)

    return result

# ?????? ????????? ??????
def convert_input_data(df):
    bertmodel, vocab = get_pytorch_kobert_model()
    data_test = BERTPredictDataset(df, 'review', nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False), 256, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=128, num_workers=8)

    return test_dataloader

class BERTPredictDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([row[sent_idx]]) for i, row in dataset.iterrows()]

    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))