import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QLineEdit, QListWidget
)
from PyQt5.QtCore import Qt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inference GUI for Twitter Sentiment Analysis Model")
        self.setGeometry(100, 100, 1200, 700)

        # Central Widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Variables
        self.model = None
        self.tokenizer = None
        self.train_data = []
        self.test_data = []

        # Layouts
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        # Title
        self.title_label = QLabel("Inference GUI for Twitter Sentiment Analysis Model")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        # Input Box for Text
        self.input_label = QLabel("Enter Text Input to Analyze its Sentiment:")
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Enter your text here...")
        top_layout.addWidget(self.input_label)
        top_layout.addWidget(self.input_text)

        # Buttons
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_dataset_button = QPushButton("Load Dataset")
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.infer_button = QPushButton("Run Inference Manual")
        self.infer_button.clicked.connect(self.run_inference)
        top_layout.addWidget(self.load_model_button)
        top_layout.addWidget(self.load_dataset_button)
        top_layout.addWidget(self.infer_button)

        # Add Top Layout
        main_layout.addLayout(top_layout)

        # List Widgets for Train and Test Data
        self.train_data_list = QListWidget()
        self.train_data_list.clicked.connect(self.on_train_data_selected)
        self.test_data_list = QListWidget()
        self.test_data_list.clicked.connect(self.on_test_data_selected)

        middle_layout.addWidget(self.train_data_list)
        middle_layout.addWidget(self.test_data_list)

        # Add Middle Layout
        main_layout.addLayout(middle_layout)

        # Output Labels for True and Predicted Sentiment
        self.true_sentiment_label = QLabel("True Sentiment: N/A")
        self.true_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")
        self.predicted_sentiment_label = QLabel("Predicted Sentiment: N/A")
        self.predicted_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")
        bottom_layout.addWidget(self.true_sentiment_label)
        bottom_layout.addWidget(self.predicted_sentiment_label)

        # Add Bottom Layout
        main_layout.addLayout(bottom_layout)

        # Set Layout
        self.central_widget.setLayout(main_layout)

    def load_model(self):
        """Loads the DistilBERT model and tokenizer."""
        model_name = "C:/Users/Awais/Desktop/Task9a/Model"  # Replace with your model path
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model.eval()
            self.status_update("Model loaded successfully!")
        except Exception as e:
            self.status_update(f"Error loading model: {e}")

    def load_dataset(self):
        """Loads the dataset and displays samples."""
        try:
            file_name = "C:/Users/Awais/Desktop/Task9a/data_sent.csv"  # Hardcoded path for CSV file

            with open(file_name, 'r', encoding='utf-8') as file:
                data = file.readlines()

            # Remove any empty lines or lines with improper formatting
            cleaned_data = []
            for line in data:
                line = line.strip()
                if line and ',' in line:  # Check if the line contains a comma (text,true_label)
                    try:
                        text, true_label = line.split(",", 1)  # Split at the first comma only
                        true_label = true_label.strip()
                        cleaned_data.append((text, true_label))
                    except ValueError:
                        continue  # Skip malformed lines

            # Shuffle the data randomly
            random.shuffle(cleaned_data)

            # Split the data into training and test sets (e.g., 80% train, 20% test)
            train_size = int(len(cleaned_data) * 0.8)
            test_size = len(cleaned_data) - train_size

            # Randomly select data for train and test
            self.train_data = cleaned_data[:train_size]
            self.test_data = cleaned_data[train_size:]

            # Display the text data in ListWidget (display only text part)
            self.train_data_list.clear()
            self.test_data_list.clear()
            self.train_data_list.addItems([text for text, _ in self.train_data])
            self.test_data_list.addItems([text for text, _ in self.test_data])

            self.status_update("Dataset loaded successfully!")

        except Exception as e:
            self.status_update(f"Error loading dataset: {e}")

    def on_train_data_selected(self):
        """Handles train data row selection."""
        selected_row = self.train_data_list.currentRow()
        if selected_row != -1:
            text, true_label = self.train_data[selected_row]
            self.true_sentiment_label.setText(f"True Sentiment: {true_label}")
            self.true_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")
            self.run_inference_for_text(text)

    def on_test_data_selected(self):
        """Handles test data row selection."""
        selected_row = self.test_data_list.currentRow()
        if selected_row != -1:
            text, true_label = self.test_data[selected_row]
            self.true_sentiment_label.setText(f"True Sentiment: {true_label}")
            self.true_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")
            self.run_inference_for_text(text)

    def run_inference(self):
        """Runs inference on the input text."""
        input_text = self.input_text.text()
        if not input_text.strip():
            self.status_update("Input text is empty!")
            return

        self.run_inference_for_text(input_text)

    def run_inference_for_text(self, input_text):
        """Runs inference for a given input text."""
        try:
            if self.model is None or self.tokenizer is None:
                self.status_update("Please load the model first!")
                return

            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).item()

            sentiment_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
            predicted_sentiment = sentiment_map.get(predictions, "Unknown")

            self.predicted_sentiment_label.setText(f"Predicted Sentiment: {predicted_sentiment}")

            # Color output based on prediction
            if predicted_sentiment == "Bullish":
                self.predicted_sentiment_label.setStyleSheet("background-color: lightgreen; font-size: 16px; padding: 5px;")
            elif predicted_sentiment == "Bearish":
                self.predicted_sentiment_label.setStyleSheet("background-color: lightcoral; font-size: 16px; padding: 5px;")
            else:
                self.predicted_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")

            # Color the true sentiment based on the label
            if self.true_sentiment_label.text() != "True Sentiment: N/A":
                true_sentiment = self.true_sentiment_label.text().split(":")[1].strip()
                if true_sentiment == "Bullish":
                    self.true_sentiment_label.setStyleSheet("background-color: lightgreen; font-size: 16px; padding: 5px;")
                elif true_sentiment == "Bearish":
                    self.true_sentiment_label.setStyleSheet("background-color: lightcoral; font-size: 16px; padding: 5px;")
                else:
                    self.true_sentiment_label.setStyleSheet("background-color: lightgray; font-size: 16px; padding: 5px;")

        except Exception as e:
            self.status_update(f"Error during inference: {e}")

    def status_update(self, message):
        """Updates status or feedback for the user."""
        print(message)  # You can also display this in a label or message box

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SentimentAnalysisApp()
    window.show()
    sys.exit(app.exec_())
