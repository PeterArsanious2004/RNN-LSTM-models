#WEEK 2
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from imagecaption import generate_caption          
from lstm_model import LSTM, text_to_numbers

DB_PATH = "database.db"
TABLE_NAME = "records"
CSV_PATH = "cellula toxic data.csv"


# ---------- Classifier ----------
@st.cache_resource(show_spinner="Training the LSTM (first launch only)...")
def load_classifier():
    torch.manual_seed(42)

    data = pd.read_csv(CSV_PATH)
    texts = (data["query"] + " " + data["image descriptions"]).tolist()
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["Toxic Category"].tolist())

    X_train, _, y_train, _ = train_test_split(
        texts, y, test_size=0.15, random_state=42, stratify=y
    )

    word_to_number = {"<PAD>": 0, "<UNK>": 1}
    for text in X_train:
        for word in text.lower().split():
            if word not in word_to_number:
                word_to_number[word] = len(word_to_number)

    X = torch.tensor([text_to_numbers(t, word_to_number) for t in X_train], dtype=torch.long)
    Y = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

    model = LSTM(len(word_to_number), len(encoder.classes_))
    counts = torch.bincount(Y, minlength=len(encoder.classes_)).float()
    weights = counts.sum() / (len(encoder.classes_) * counts.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for _ in range(25):
        for bx, by in loader:
            loss = loss_fn(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    return model, word_to_number, list(encoder.classes_)


def classify(text: str) -> str:
    model, word_to_number, classes = load_classifier()
    x = torch.tensor([text_to_numbers(text, word_to_number)], dtype=torch.long)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).item()
    return classes[pred]


# ---------- database ----------
def initialize_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                classification TEXT NOT NULL
            )
            """
        )


def save_record(source: str, text: str, label: str):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            f"INSERT INTO {TABLE_NAME} (timestamp, source, text, classification) VALUES (?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source, text, label),
        )


def load_records() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            f"SELECT timestamp, source, text, classification FROM {TABLE_NAME} ORDER BY id",
            connection,
        )


initialize_database()


# ---------- UI ----------
st.title("Content Classification App")
page = st.sidebar.radio("Menu", ["Classify", "View Database"])

if page == "Classify":
    input_type = st.radio("Choose input type", ["Type text", "Upload picture"])
    user_text = ""
    uploaded = None

    if input_type == "Type text":
        user_text = st.text_area("Text input")
    else:
        uploaded = st.file_uploader("Image input", type=["jpg", "jpeg", "png"])

    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

    if st.button("Classify"):
        if not user_text.strip() and image is None:
            st.warning("Please enter text and/or upload an image.")
        else:
            caption = ""
            if image is not None:
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image)
                st.write(f"**Caption:** {caption}")

            # The LSTM was trained on "query + image description"
            combined = f"{user_text.strip()} {caption}".strip()
            label = classify(combined)

            if user_text.strip():
                save_record("user_text", user_text.strip(), label)
            if caption:
                save_record("image_caption", caption, label)
            st.success(f"Classification: {label}")

else:
    st.subheader("Stored inputs and classifications")
    df = load_records()
    if df.empty:
        st.info("The database is empty.")
    else:
        st.dataframe(df, use_container_width=True)
