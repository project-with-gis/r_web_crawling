from preprocessing import *
from csv_handler import *

import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

def bert_prepro(df):
    # 훈련셋 테스트셋으로 분리
    train, test = train_test_split(df[['preprocessed_review', 'score']], test_size=0.2, random_state=42)
    # 문장별 전처리
    review_bert = ["[CLS] " + str(s) + " [SEP]" for s in train.preprocessed_review]

    # 토크나이징 - 사전학습된 BERT multilingual 모델 내 포함되어있는 토크나이저를 활용
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(s) for s in review_bert]

    # 패딩 - token들의 max length보다 크게 MAX_LEN을 설정 # 토크나이징 된 문장들의 길이를 모두 통일하는 과정
    MAX_LEN = 355 #(테스트 데이터 기준임. 데이터 추가했을 땐 바꿔주기)
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids[0] # input_ids : train 데이터 수 만큼 들어있음

    # 어텐션 마스크 - 학습속도를 높이기 위해 실 데이터가 있는 곳과 padding이 있는 곳을 attention에게 알려줍니다.
    attention_masks = []
    for seq in input_ids: # 리스트속 0이 아닌 값은 1로 채워서 세로운 리스트 생성 = attention_mask
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)


    # train - validation set 분리
    train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids, train['score'].values, random_state=42, test_size=0.1)

    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.1)

    # 파이토치 텐서로 변환 - numpy ndarray로 되어있는 input, label, mask들을 torch tensor로 변환
    train_inputs = torch.tensor(train_inputs) # 똑같은 리스트 모양인데 타입이 torch로 바뀌는 듯
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    # 배치 및 데이터로더 설정
    BATCH_SIZE = 32

    train_data = TensorDataset(train_inputs, train_masks, train_labels) # inputs이랑 mask랑 합쳐진 형태
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    # 테스트셋 전처리 - 위의 train-val 셋 전처리와 동일
    sentences = test['preprocessed_review']
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = test['score'].values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, test_dataloader


# 학습에 필요한 함수 정의
def flat_accuracy(preds, labels): # 정확도 계산 함수
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed): # 시간 표시 함수
    elapsed_rounded = int(round((elapsed))) # 반올림
    return str(datetime.timedelta(seconds=elapsed_rounded)) # hh:mm:ss으로 형태 변경

def bert_model(train_dataloader, test_dataloader):
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    # 분류를 위한 BERT 모델 생성 - transformers의 BertForSequenceClassification 모듈을 이용
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=6)
    model.cpu()

    # 학습스케쥴링
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # 학습률
                      eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
      ) # 옵티마이저 설정
    epochs = 4 # 에폭수
    total_steps = len(train_dataloader) * epochs # 총 훈련 스텝
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps = total_steps) # lr 조금씩 감소시키는 스케줄러

    # 재현을 위해 랜덤시드 고정
    seed_val = 32
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 그래디언트 초기화
    model.zero_grad()

    # 에폭만큼 반복
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time() # 시작 시간 설정

        total_loss = 0 # 로스 초기화

        model.train() # 훈련모드로 변경

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_dataloader):
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids).to(device).long()

            print(b_labels)

            # Forward 수행
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0] # 로스 구함
            total_loss += loss.item() # 총 로스 계산
            loss.backward() # Backward 수행으로 그래디언트 계산
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
            optimizer.step() # 그래디언트를 통해 가중치 파라미터 업데이트
            scheduler.step() # 스케줄러로 학습률 감소
            model.zero_grad() # 그래디언트 초기화

        avg_train_loss = total_loss / len(train_dataloader) # 평균 로스 계산

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        # ========================================
        #               Validation
        # ========================================
        print("")
        print("Running Validation...")

        t0 = time.time() # 시작 시간 설정
        model.eval() # 평가모드로 변경

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in test_dataloader: # 데이터로더에서 배치만큼 반복하여 가져옴
            batch = tuple(t.to(device) for t in batch) # 배치를 device를 넣음
            print(batch)

            b_input_ids, b_input_mask, b_labels = batch # 배치에서 데이터 추출
            b_input_ids = torch.tensor(b_input_ids).to(device).long()
            print(b_labels)

            # 그래디언트 계산 안함
            with torch.no_grad():
                # Forward 수행
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0] # 로스 구함

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    torch.save(model, 'model_min1.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), 'model_state_min1.pt')