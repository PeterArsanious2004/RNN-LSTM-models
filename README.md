# Cellula Toxic Data Classification

This project uses PyTorch recurrent neural networks to classify text into toxic-content categories. Each text sample combines a user query with its image description.

## Files

- `cellula toxic data.csv`: Dataset containing queries, image descriptions, and toxic categories.
- `RNN.ipynb`: Text classification model using a simple RNN.
- `LSTM.ipynb`: Text classification model using an LSTM.
- `requirements.txt`: Python dependencies used by the project.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Running the notebooks

1. Open `RNN.ipynb` or `LSTM.ipynb` in VS Code or Jupyter.
2. Select the Python environment containing the installed requirements.
3. Run the cells from top to bottom.
4. Review the training loss, F1 score, and confusion matrix.

## Running the app

Start the Streamlit application with:

```bash
streamlit run app.py
```

Use the `Classify` page to classify text or an uploaded image. Every classified
input is stored in `database.db`. Open `View Database` from the sidebar at any
time to see all stored inputs, their source, timestamps, and classifications.

## What the notebooks do

Both notebooks follow the same text-classification workflow:

1. Load the CSV dataset with pandas.
2. Combine the `query` and `image descriptions` columns into one text input.
3. Convert the toxic-category labels into numeric class labels.
4. Split the data into training and test sets while preserving the class distribution.
5. Build a vocabulary from the training text. Unknown words are mapped to `<UNK>` and shorter sequences are filled with `<PAD>` tokens.
6. Convert each text into a fixed-length sequence of word IDs.
7. Train a neural network using weighted cross-entropy loss to give more importance to underrepresented categories.
8. Evaluate predictions with macro F1 score and a confusion matrix.

### `RNN.ipynb`

This notebook implements a simple vanilla RNN. An embedding layer converts word IDs into vectors, and the RNN processes each text sequence in order. The final hidden state summarizes the sequence and is passed through dropout and a linear classification layer to predict the toxic category.

The notebook uses packed sequences so padding tokens do not affect the final hidden state.

### `LSTM.ipynb`

This notebook uses an LSTM instead of a vanilla RNN. LSTMs contain additional gates and memory states that help preserve important information across longer sequences. The final sequence representation is passed to a classification layer to predict the toxic category.

The two notebooks can be compared using their training loss, macro F1 score, and confusion matrices.

## Model evaluation

The models use macro F1 score, which gives each toxic category equal importance. This is useful when the categories have different numbers of examples.
