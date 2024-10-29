# Import required libraries
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
train_data = pd.read_csv('C:/Users/511522104061/Downloads/train.csv')

# Text cleaning and preprocessing function (using Lemmatization)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    else:
        return ''  # Or any other suitable default value

# Apply preprocessing to the dataset
train_data['cleaned_text'] = train_data['crimeaditionalinfo'].apply(preprocess_text)

# Split data into training and test sets
X = train_data['cleaned_text']
y = train_data['category']  # Main category
y_sub = train_data['sub_category']  # Sub-category

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
label_encoder_sub = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_sub_encoded = label_encoder_sub.fit_transform(y_sub)

X_train, X_test, y_train, y_test, y_train_sub, y_test_sub = train_test_split(X, y_encoded, y_sub_encoded, test_size=0.2, random_state=42)

# Tokenization and Padding (for deep learning models)
max_words = 5000  # Maximum number of words to consider in the vocabulary
max_len = 100     # Maximum length of the sequences

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post')

# Define the GRU Model with Bidirectional Layer and Increased Epochs
embedding_dim = 100  # Increased embedding dimension

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(GRU(64, return_sequences=False)))  # Bidirectional GRU layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  # Increased hidden layer size
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

# Compile the model with a lower learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with more epochs
history = model.fit(X_train_padded, y_train, epochs=10, validation_split=0.2, batch_size=64)  # Increased epochs

# Make predictions
y_pred_proba = model.predict(X_test_padded)
y_pred = y_pred_proba.argmax(axis=1)

# Accuracy Measurement
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Confusion matrix as a DataFrame
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
print("\nConfusion Matrix DataFrame:\n", cm_df)

# Confusion matrix as a bar chart
cm_sum = cm.sum(axis=1)  # Sum the true instances for each class
cm_correct = np.diag(cm)  # Correctly predicted instances (diagonal elements)
cm_incorrect = cm_sum - cm_correct  # Incorrectly predicted instances

# Create a DataFrame for plotting
cm_plot_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Correct': cm_correct,
    'Incorrect': cm_incorrect
})

# Plot the bar chart
cm_plot_df.set_index('Class').plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(10, 6))
plt.title('Correct vs Incorrect Predictions per Class')
plt.ylabel('Number of Instances')
plt.xlabel('Class')
plt.show()


# Save the model and tokenizer for future use
model.save('gru_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

#Tkinter GUI based model

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model, tokenizer, and label encoder
model = load_model('gru_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# NLTK setup
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    else:
        return ''  # Or any other suitable default value

# Prediction function
def predict_category(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')  # Adjust maxlen if needed
    prediction_proba = model.predict(padded_sequence)
    prediction = prediction_proba.argmax(axis=1)
    predicted_category = label_encoder.inverse_transform(prediction)
    return predicted_category[0]

# Function to handle predictions for multiple entries in a CSV file
def predict_from_file(filepath):
    try:
        data = pd.read_csv(filepath)
        if 'crimeaditionalinfo' not in data.columns:
            messagebox.showerror("Error", "The file must contain 'crimeaditionalinfo' column.")
            return None
        data['cleaned_text'] = data['crimeaditionalinfo'].apply(preprocess_text)
        sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
        predictions_proba = model.predict(padded_sequences)
        predictions = predictions_proba.argmax(axis=1)
        data['predicted_category'] = label_encoder.inverse_transform(predictions)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if save_path:
            data.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Predictions saved to {save_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("NLP Model - Category Prediction")
root.geometry("500x400")

# Input label and text box
input_label = tk.Label(root, text="Enter text for prediction:")
input_label.pack(pady=5)

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack(pady=5)

# Prediction display
result_label = tk.Label(root, text="Predicted Category: ", font=("Arial", 14))
result_label.pack(pady=10)

# Predict button function
def predict_button_click():
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        prediction = predict_category(text)
        result_label.config(text=f"Predicted Category: {prediction}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text for prediction.")

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_button_click)
predict_button.pack(pady=5)

# File prediction button function
def file_button_click():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        predict_from_file(file_path)

# File prediction button
file_button = tk.Button(root, text="Predict from CSV File", command=file_button_click)
file_button.pack(pady=5)

# Run the application
root.mainloop()



