import os
import glob
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""script to search for all english data and make a single file
with all the english transcriptions and audio paths, encode the labels and save the data"""


def prepare_data(data_dir, output_dir):
    audio_dir = os.path.join(data_dir, "audio")
    text_dir = os.path.join(data_dir, "text")

    eng_files = glob.glob(
        os.path.join(text_dir, "en-*.csv")
    )  # get list of english subdirs

    eng_data = pd.DataFrame()

    # for every subdir in eng_files, add all transciptions + modified audio path to final df
    for csv_file in eng_files:
        temp_df = pd.read_csv(csv_file)
        eng_data = pd.concat([eng_data, temp_df], ignore_index=True)

    eng_data["full_audio_path"] = eng_data["filepath"].apply(
        lambda x: os.path.join(audio_dir, x)
    )
    eng_data.drop(columns=["filepath", "text_asr"], inplace=True)

    label_encoder = LabelEncoder()
    eng_data["intent_encoded"] = label_encoder.fit_transform(eng_data["intent"])

    os.makedirs(output_dir, exist_ok=True)

    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"LabelEncoder saved to: {encoder_path}")


    train_data, test_data = train_test_split(
        eng_data, test_size=0.3, random_state=42, stratify=eng_data["intent_encoded"]
    )
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, random_state=42, stratify=test_data["intent_encoded"]
    )

    train_data["split"] = "train"
    val_data["split"] = "val"
    test_data["split"] = "test"

    combined_data = pd.concat([train_data, test_data, val_data], ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "intent_data.csv")
    combined_data.to_csv(csv_path, index=False)

    print(f"Data saved to: {csv_path}")

    return csv_path


def get_data_info(df, data_dir):
    with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    print("Data Overview:")
    print(f"Total utterances: {len(df)}")

    print(f"Train set size: {len(df[df['split'] == 'train'])} utterances\n")
    print(
        f"Train set intent distribution:\n{df[df['split'] == 'train']['intent'].value_counts()}"
    )
    print(f"Validation set size: {len(df[df['split'] == 'val'])} utterances\n")
    print(
        f"Validation set intent distribution:\n{df[df['split'] == 'val']['intent'].value_counts()}"
    )
    print(f"Test set size: {len(df[df['split'] == 'test'])} utterances\n")
    print(
        f"Test set intent distribution:\n{df[df['split'] == 'test']['intent'].value_counts()}"
    )
    
    print(f"Label classes: {label_encoder.classes_}")
    print(f"Number of intent classes: {len(label_encoder.classes_)}")


if __name__ == "__main__":
    data_dir = "../data/MInDS-14"
    output_dir = "../data/processed_minds14"
    #full_path="../data/processed_minds14/intent_data.csv"

    full_path = prepare_data(data_dir, output_dir)

    data = pd.read_csv(full_path)

    data.head(5)

    get_data_info(data, output_dir)
