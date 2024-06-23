
from catboost import Pool, CatBoostClassifier
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import re
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import torchaudio

from transformers import pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tqdm.pandas()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_wav_to_text_model(model_name='openai/whisper-large-v3'):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    return processor, model


def get_text_from_wav(df, path, processor, model):
    def resample_audio(audio_input, original_sample_rate, target_sample_rate=16000):
        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            audio_input = resampler(torch.tensor(audio_input).float())
        return audio_input

    def process_batch(files):
        for file_path in tqdm(files):
            if file_path.endswith('.wav'):
                file_id = re.findall('\d+.\d+', os.path.basename(file_path))[0]
                try:
                    audio_input, original_sample_rate = sf.read(file_path)
                    audio_input = resample_audio(audio_input, original_sample_rate).numpy()
                    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
                    input_features = input_features.to(DEVICE)
                    predicted_ids = model.generate(input_features)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    df.loc[df['ID записи'] == file_id, 'text_conv'] = transcription
                except Exception as e:
                    df.loc[df['ID записи'] == file_id, 'text_conv'] = None

    wav_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
    process_batch(wav_files)
    return df


def get_bert_features_extractor():
    pipe = pipeline("feature-extraction", model="nielsr/lilt-xlm-roberta-base")
    return pipe


def get_embeddings(df, bert_pipe):
    def get_emb(text):
        try:
            return bert_pipe(text)[0][-1]
        except:
            return None

    df['emb'] = df['text_conv'].progress_apply(get_emb)
    return df


def clear_df(df, name):
    df = df[~df[name].isna()]
    return df


def train_model_embeddings(df):
    X = np.array(df['emb'].tolist())
    y = df['Успешный результат']
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=0.015564950023394423, kernel='rbf', gamma=0.02719557109129227, probability=True))
    ])
    pipeline.fit(X, y)
    return pipeline


def get_proba(df, pipeline):
    X = np.array(df['emb'].tolist())
    df['proba'] = pipeline.predict_proba(X)[:, 0]
    return df


def prepare_to_table(df):
    df = df.drop(['text_conv', 'emb'], axis=1)
    df = df.drop(['Время перезвона', 'Состояние перезвона', 'ID заказа звонка', 'Теги', 'Метка', 'Оценка'], axis=1)
    df = df.drop(['ID записи'], axis=1)

    def conv_time(ss):
        return pd.to_datetime(ss, format='%d.%m.%Y %H:%M:%S')

    df['Время'] = df['Время'].progress_apply(conv_time)
    df['Схема'] = df['Схема'].astype(str)
    df['Ответственный из CRM'] = df['Ответственный из CRM'].astype(str)
    X = df
    y = None
    if "Успешный результат" in df:
        y = df['Успешный результат']
        X = df.drop('Успешный результат', axis=1)

    pool_all = Pool(
        X,
        label=y,
        cat_features=['Тип', 'Статус', 'Схема', 'Куда', 'Кто ответил', 'Ответственный из CRM'],
        feature_names=['Тип', 'Статус', 'Время', 'Схема', 'Откуда', 'Куда', 'Кто ответил',
                       'Длительность звонка', 'Длительность разговора', 'Время ответа',
                       'Запись существует', 'Новый клиент', 'Ответственный из CRM', 'proba'])
    return pool_all


def train_error_corrector(pool_all):
    best_params = {
        "objective": "Logloss",
        "colsample_bylevel": 0.09440173601958003,
        "depth": 12,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bernoulli",
        "verbose": False,
        "eval_metric": 'Precision',
        "subsample": 0.9584094346366171

    }
    error_corrector = CatBoostClassifier(**best_params)
    error_corrector.fit(pool_all)
    return error_corrector


def clear_test(df,name):
    ids = df[df[name].isna()]['ID записи'].tolist()
    df = df[~df[name].isna()]
    return df, ids
def add_zero(text):
    if len(text)!=17:
        return text+'0'
    else:
        return text
def main():

    print(f"root of project: {os.getcwd()}")
    path_train = input("Enter path to training data: ")
    path_test = input("Enter path to test data: ")


    df_train = pd.read_csv(f'{path_train}/info.csv', sep=';')
    df_train['ID записи'] = df_train['ID записи'].astype(str)
    df_train['ID записи'] = df_train['ID записи'].apply(add_zero)
    df_test = pd.read_csv(f'{path_test}/info.csv', sep=';')
    df_test['ID записи'] = df_test['ID записи'].astype(str)
    df_test['ID записи'] = df_test['ID записи'].apply(add_zero)
    df_test_ids = df_test['ID записи'].tolist()

    if os.path.isfile(f'{path_train}/stage1.csv') and os.path.isfile(f'{path_test}/stage1.csv'):
        df_train = pd.read_csv(f'{path_train}/stage1.csv')
        if 'Unnamed: 0' in df_train.columns:
            df_train = df_train.drop('Unnamed: 0', axis=1)
        df_test = pd.read_csv(f'{path_test}/stage1.csv')
        if 'Unnamed: 0' in df_test.columns:
            df_test = df_test.drop('Unnamed: 0', axis=1)
        print('Files already exist')
    else:
        print('Preprocessing...(long time ~ 2 hours)')
        processor, model = get_wav_to_text_model()
        df_train = get_text_from_wav(df_train, path_train, processor, model)
        df_test = get_text_from_wav(df_test, path_test, processor, model)

        df_train.to_csv(f'{path_train}/stage1.csv', index=False)
        df_test.to_csv(f'{path_test}/stage1.csv', index=False)

    df_train = clear_df(df_train, 'text_conv')
    df_test, ids = clear_test(df_test, 'text_conv')

    print('Get embeddings...')
    bert_pipe = get_bert_features_extractor()
    df_train = get_embeddings(df_train, bert_pipe)
    df_test = get_embeddings(df_test, bert_pipe)

    df_test, ids1 = clear_test(df_test, 'emb')
    ids+=ids1
    df_train = clear_df(df_train, 'emb')

    pipeline = train_model_embeddings(df_train)
    df_train = get_proba(df_train, pipeline)
    pool_to_train = prepare_to_table(df_train)

    error_corrector = train_error_corrector(pool_to_train)

    df_test = get_proba(df_test,pipeline)
    # df_test = df_test.drop('Успешный результат', axis=1) #test
    pool_to_test = prepare_to_table(df_test)

    df_test['Успешный результат'] = error_corrector.predict(pool_to_test)
    ids = set(ids)
    with open('result.txt', 'w') as result:
        for i in df_test_ids:
            if i not in ids:
                result.write(f'{i}-{df_test[df_test["ID записи"] == i]["Успешный результат"].item()}\n')
            else:
                result.write(f'{i}-Fail\n')


if __name__ == "__main__":
    main()
