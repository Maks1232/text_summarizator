import math
import nltk
from nltk import PorterStemmer
import tkinter as tk
from tkinter import ttk
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import re
from transformers import pipeline
import string
import time
from tkinter import filedialog
import PyPDF2
import cv2
import os
import numpy as np
import easyocr
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_batch_cost
from PIL import Image, ImageTk
import tensorflow as tf
import pickle



# Download the 'punkt' and 'stopwords' resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


def sentence_tokenizer(text):
    """
    Function used to tokenize each sentence of text
    :param text: text to tokenize
    :return tokenized  sentences
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentence_tokens = [sent.text for sent in doc.sents]
    return sentence_tokens


def word_tokenizer(text):
    """
       Function used to tokenize each word from text
       :param text: text to tokenize
       :return: tokenized  sentences
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    word_tokens = [token.text.lower() for token in doc if token is not string.punctuation]
    word_tokens = [item for item in word_tokens if item != '\n']
    return word_tokens


def calculate_occurrences_for_sent(sentences):
    """
    Function used to calculate word occurrences in each sentence

    PorterStemmer -  removing the commoner morphological and inflexional endings from words in English:
    # running -> run
    # runs -> run
    # ran -> ran
    # easily -> easili
    # fairly -> fairli #sometimes it creates new words

    :param sentences: sentences in which word occurrences will be calculated
    :return: dictionary in which keys are sentences and values are dictionaries with word counts in each sentence
    """

    occurrences_dict = {}
    stopwords = list(STOP_WORDS)

    ps = PorterStemmer()

    for sent in sentences:
        count_occurrences = {}
        words = word_tokenizer(sent)
        for word in words:
            word = ps.stem(word)  # cutting endings of word
            if word not in stopwords:
                if word not in count_occurrences:
                    count_occurrences[word] = 1
                else:
                    count_occurrences[word] += 1
        occurrences_dict[sent[:10]] = count_occurrences
    return occurrences_dict


def calculate_occurrences_for_text(occurrences):
    """
    Function used to calculate word occurrences in whole text
    :param occurrences: dictionary in which keys are sentences and values are dict with word counts in each sentence
    :return: Dict with words and their number of occurrences in the whole text
    """
    word_in_text_dict = {}

    for sent, word_n_count in occurrences.items():
        for word, count in word_n_count.items():
            if word not in word_in_text_dict:
                word_in_text_dict[word] = 1
            else:
                word_in_text_dict[word] += 1

    return word_in_text_dict


def calculate_tfidf(occurrences_in_sent, occurrences_in_text, df):
    """
    Function used to calculate IDF(w, D) = log(N / (DF(w, D) + 1))
    N - number of sentences
    DF(w, D) - number of word occurrences in text
    :param occurrences_in_sent: Dict with words and word occurrences in sentence for each sentence
    :param occurrences_in_text: Dict with words and word occurrences in the whole text
    :param df: number of sentences
    :return: Dict of Inverse Document Frequency (IDF)
    """
    tf_idf = {}

    for sentence, words_n_number in occurrences_in_sent.items():
        tf_idf_temp = {}

        count_words_in_sentence = len(words_n_number)

        for word, number in words_n_number.items():
            tf_idf_temp[word] = (number / count_words_in_sentence) * \
                                (math.log10(df / (float(occurrences_in_text[word]) + 1)))

        tf_idf[sentence] = tf_idf_temp

    return tf_idf


def score_sentences_by_tfidf(tfidf):
    """
    Function used to calculate sentence score using TF-IDF
    :param tfidf: Dict of TF-IDF
    :return: Dict of sentences and their scores
    """

    sent_scores = {}

    for sent, td_idf_dict in tfidf.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(td_idf_dict)
        for word, score in td_idf_dict.items():
            total_score_per_sentence += score
        sent_scores[sent] = total_score_per_sentence / count_words_in_sentence

    return sent_scores


def generate_summary_tfidf(sentences, sentence_scores_tfidf, summ_length):
    """
    Function used to chose sentences for a summary
    :param sentences: Text tokenized sentences
    :param sentence_scores_tfidf: Sentence scores
    :param summ_length: Chosen summary length
    :return: Summarized text
    """
    summ = 0
    for entry in sentence_scores_tfidf:
        summ += sentence_scores_tfidf[entry]
    threshold = (summ / len(sentence_scores_tfidf))

    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentence_scores_tfidf and sentence_scores_tfidf[sentence[:10]] >= (summ_length * threshold):
            summary += " " + sentence

    return summary


def run_tf_idf_summarization(text, summ_length=0.3):
    sentences = sentence_tokenizer(text)
    occurrences_in_sent = calculate_occurrences_for_sent(sentences)
    occurrences_in_text = calculate_occurrences_for_text(occurrences_in_sent)
    tfidf = calculate_tfidf(occurrences_in_sent, occurrences_in_text, len(sentences))
    sentence_scores_tfidf = score_sentences_by_tfidf(tfidf)
    summary = generate_summary_tfidf(sentences, sentence_scores_tfidf, summ_length + 1)

    return summary


def run_basic_summarization(text, max_length=0.3):
    # List of stop words that have to be removed preparation
    stopwords = list(STOP_WORDS)
    # Removing reference numbers
    text = re.sub(r'\[\d+\]', '', text)
    # Tokenized text below
    doc = nlp(text.replace('\n', ''))
    # Tokenization
    tokens = [token.text for token in doc]
    # Punctuation load
    punctuation_list = list(punctuation)
    punctuation_list.append('\n')
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation_list:
                if word.text.lower() not in word_frequencies.keys():
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1
    # Sentence tokenization
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    # Taking x% sentences as a upper limit for text summary
    select_length = int(len(sentence_tokens) * max_length)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([sent.text for sent in summary])
    return final_summary


def run_hugging_face_transformer(text, max_length=200, min_length=30):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]


def run_summarization(text, model, offset):
    if model == "TF-IDF":
        return run_tf_idf_summarization(text, offset)
    elif model == "Basic":
        return run_basic_summarization(text, offset)
    elif model == "HuggingFace":
        max_len = int(offset * count_characters_without_spaces(text))
        return run_hugging_face_transformer(text, max_len)
    else:
        return "Unknown model."


def count_characters_without_spaces(text):
    return len(text.replace(" ", "").replace("\n", ""))


def summarize_text():
    input_text = input_text_area.get("1.0", tk.END).strip()
    selected_model = model_combobox.get()
    if not input_text:
        result_text_area.delete("1.0", tk.END)
        result_text_area.insert(tk.END, "Insert text to be summarized!")
        return

    power_of_sum = summary_len

    start = time.time()
    result = run_summarization(input_text, selected_model, power_of_sum)
    stop = time.time()

    result_text_area.tag_configure("bold_blue", font=("Helvetica", 12, "bold"), foreground="blue")

    result_text_area.delete("1.0", tk.END)
    result_text_area.insert(tk.END, f"Execution time: {round(stop - start, 3)}s\n\n", "bold_blue")
    result_text_area.insert(tk.END, result)


def show_character_count():
    input_text = input_text_area.get("1.0", tk.END).strip()
    char_count = count_characters_without_spaces(input_text)
    result_text_area.delete("1.0", tk.END)
    result_text_area.insert(tk.END, f"Characters without spaces and newlines count: {char_count}")


def update_scale_label(value):
    scale_value.set(f"{int(float(value) * 100)}%")  # Update label to show percentage
    global summary_len
    summary_len = float(value)  # Update the summary length variable


def fetch_url_text():
    url = url_entry.get()
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from relevant sections of the website (modify as needed)
        article_text = soup.find_all('p')  # Assuming paragraphs hold the article text
        text = " ".join([p.text.strip() for p in article_text])  # Join paragraph texts
        cleaned_text = re.sub(r'\[\d+\]', '', text)  # Removes patterns like [1], [2] using regex
        input_text_area.delete("1.0", tk.END)
        input_text_area.insert(tk.END, cleaned_text)
    except requests.exceptions.RequestException as e:
        result_text_area.delete("1.0", tk.END)
        result_text_area.insert(tk.END, f"Error fetching URL: {e}")


def fetch_pdf_text():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

                # Remove newlines and hyphens
                text = text.replace("-", "").replace("\n", " ")
                #

                input_text_area.delete("1.0", tk.END)
                input_text_area.insert(tk.END, text)
        except Exception as e:
            result_text_area.delete("1.0", tk.END)
            result_text_area.insert(tk.END, f"PDF Error read out: {e}")


# Global variables for image processing
current_image = None
word_images = []


def show_alert(message):
    result_text_area.tag_configure("alert", foreground="green")
    result_text_area.delete("1.0", tk.END)
    result_text_area.insert(tk.END, message + "\n", "alert")


def process_image():
    global word_images
    # Inicjalizacja czytnika z odpowiednimi parametrami
    reader = easyocr.Reader(['en', 'pl'],
                            gpu=False,
                            decoder='greedy',  # Tryb "chciwy" dla pojedynczych słów
                            batch_size=1)  # Wymusza przetwarzanie pojedynczych elementów

    # Preprocessing obrazu
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    processed_img = clahe.apply(gray)

    # Detekcja tekstu z zaostrzonymi parametrami
    results = reader.readtext(processed_img,
                              paragraph=False,
                              min_size=20,  # Minimalna wysokość znaków w pikselach
                              text_threshold=0.4,
                              link_threshold=0.4,
                              mag_ratio=3)  # Powiększenie obszaru analizy

    # Ekstrakcja ROI i filtracja
    word_images = []
    for (bbox, text, prob) in results:
        # Pomijanie zbyt szerokich bboxów (prawdopodobnie całe zdania)
        x_min, y_min = map(int, bbox[0])
        x_max, y_max = map(int, bbox[2])
        bbox_width = x_max - x_min

        if bbox_width > current_image.shape[1] * 0.2:  # Jeśli bbox > 20% szerokości obrazu
            # Dzielenie wykrytego tekstu na pojedyncze słowa
            words = text.split()
            avg_char_width = bbox_width / len(text.replace(" ", ""))

            # Szacowanie pozycji poszczególnych słów
            for i, word in enumerate(words):
                word_start = int(x_min + avg_char_width * text.find(word))
                word_end = int(word_start + avg_char_width * len(word))
                word_img = processed_img[y_min:y_max, word_start:word_end]
                word_images.append(word_img)
        else:
            word_img = processed_img[y_min:y_max, x_min:x_max]
            word_images.append(word_img)

    # Dodatkowa weryfikacja
    word_images = [img for img in word_images if img.shape[0] > 10 and img.shape[1] > 10]

    save_test_dataset(word_images)
    show_alert(f"Extracted {len(word_images)} words. Ready for prediction!")


def enhance_image_quality(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def save_test_dataset(word_images):
    test_dir = "test_dataset"
    os.makedirs(test_dir, exist_ok=True)
    for i, img in enumerate(word_images):
        cv2.imwrite(f"{test_dir}/word_{i}.png", img)
    run_predictions_on_dataset(test_dir)


def run_full_pipeline():
    global current_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return

    try:
        # Load and process image
        current_image = cv2.imread(file_path)
        enhanced = enhance_image_quality(current_image)

        # Show processing status
        result_text_area.delete("1.0", tk.END)
        result_text_area.insert(tk.END, "Processing image...\n", "alert")
        root.update_idletasks()

        # Extract words
        reader = easyocr.Reader(['en', 'pl'])
        results = reader.readtext(enhanced, paragraph=False)
        word_images = []

        for (bbox, text, prob) in results:
            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[2])
            word_images.append(enhanced[y_min:y_max, x_min:x_max])

        # Save and predict
        save_test_dataset(word_images)

        # predictions = run_predictions_on_dataset("test_dataset")
        #
        # # # Show results
        # # result_text_area.insert(tk.END, f"\nExtracted {len(word_images)} words:\n")
        # # for i, pred in enumerate(predictions):
        # #     result_text_area.insert(tk.END, f"Word {i + 1}: {pred}\n")

    except Exception as e:
        result_text_area.insert(tk.END, f"\nError: {str(e)}", "error")


# Modified version of your load_images function
def load_images(image_paths, target_size=(128, 32), padding_color=255, binarize=False):
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = load_img(path, color_mode='grayscale')
            img_array = img_to_array(img).squeeze()
            h, w = img_array.shape[:2]
            scale = min(target_size[1] / h, target_size[0] / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Padding
            padded_img = np.full((target_size[1], target_size[0]), padding_color, dtype=np.uint8)
            pad_top = (target_size[1] - new_h) // 2
            pad_left = (target_size[0] - new_w) // 2
            padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

            # Binarization if specified
            if binarize:
                # GaussianBlur for noise reduction
                # blurred_img = cv2.GaussianBlur(padded_img, (3, 3), 0)
                blurred_img = padded_img

                # Otsu's Threshold with adjustment
                otsu_thresh, _ = cv2.threshold(
                    blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                adjusted_thresh = max(0, min(255, otsu_thresh + 10))  # Slightly adjust the threshold
                _, binary_img = cv2.threshold(
                    blurred_img, adjusted_thresh, 255, cv2.THRESH_BINARY_INV
                )

                # Skeletonization
                skel = np.zeros_like(binary_img)
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, element)
                    temp = cv2.subtract(binary_img, open_img)
                    eroded = cv2.erode(binary_img, element)
                    skel = cv2.bitwise_or(skel, temp)
                    binary_img = eroded.copy()
                    if cv2.countNonZero(binary_img) == 0:
                        break

                # Combine binary skeleton with original
                combined_img = cv2.addWeighted(padded_img.astype(np.float32) / 255.0, 0.7,
                                               1 - skel.astype(np.float32) / 255.0, 0.3, 0)

                # Normalize the result to [0, 1]
                padded_img = combined_img
            else:
                # Normalize to [0, 1]
                padded_img = padded_img / 255.0

            padded_img = np.expand_dims(padded_img, axis=-1)
            images.append(padded_img)
            valid_paths.append(path)

        except Exception as e:
            print(f"Skipping invalid or missing image: {path}. Error: {e}")

    return np.array(images), valid_paths


def ctc_loss(y_true, y_pred):
    """
    :param y_true: tf of true labels, with shape (batch_size, max_label_length). The padding character is -1.
    :param y_pred: tf of predicted labels, with shape (batch_size, time_steps, num_classes).
    :return: CTC (connectionist temporal classification) loss for batch.
    """
    batch_size = tf.shape(y_true)[0]
    input_length = tf.fill([batch_size, 1], tf.shape(y_pred)[1])
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1, keepdims=True)
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)


def paste_prediction_to_input(prediction_text):
    """Wkleja wybrany tekst predykcji do pola tekstowego input_text_area."""
    input_text_area.delete("1.0", tk.END)  # Czyści pole tekstowe
    input_text_area.insert("1.0", prediction_text)  # Wkleja tekst


def display_predictions_in_gui(image_paths, predictions, vocab):
    result_window = tk.Toplevel(root)
    result_window.title("Predykcje ręcznego pisma")

    canvas = tk.Canvas(result_window)
    scrollbar = ttk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    decoded_texts = []

    # Sortowanie ścieżek po numerach w nazwach plików
    sorted_indices = sorted(range(len(image_paths)),
                            key=lambda x: int(os.path.basename(image_paths[x]).split('_')[1].split('.')[0]))

    for idx in sorted_indices:
        path = image_paths[idx]
        pred = predictions[idx]

        frame = ttk.Frame(scrollable_frame)
        frame.pack(pady=10, fill='x')

        # Oryginalny obraz
        img = Image.open(path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        label_img = ttk.Label(frame, image=photo)
        label_img.image = photo
        label_img.pack(side='left', padx=10)

        # Tekst predykcji
        decoded_text = decode_prediction(pred[idx], vocab)
        decoded_texts.append(decoded_text)
        label_text = ttk.Label(frame,
                               text=f"{decoded_text}",
                               font=('Helvetica', 12, 'bold'),
                               foreground='blue')
        label_text.pack(side='left', padx=20)

        # Przycisk dla pojedynczej predykcji
        paste_button = ttk.Button(frame, text="Paste to Input",
                                  command=lambda text=decoded_text: paste_prediction_to_input(text))
        paste_button.pack(side='right', padx=10)

        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x')

    # Przycisk do wklejania wszystkich predykcji
    paste_all_frame = ttk.Frame(scrollable_frame)
    paste_all_frame.pack(pady=20, fill='x')

    ttk.Label(paste_all_frame, text="Full text reconstruction:",
              font=('Helvetica', 14, 'bold')).pack()

    full_text = " ".join(decoded_texts)
    ttk.Label(paste_all_frame, text=full_text,
              wraplength=600).pack(pady=10)

    ttk.Button(paste_all_frame, text="Paste ALL to Input",
               command=lambda: paste_prediction_to_input(full_text),
               style='Bold.TButton').pack(pady=10)

    # Styl dla przycisku
    style = ttk.Style()
    style.configure('Bold.TButton', font=('Helvetica', 12, 'bold'))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")


def load_vocab():
    try:
        with open('vocab.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Błąd: Brak pliku vocab.pkl. Generowanie domyślnego słownika...")
        default_vocab = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(default_vocab, f)
        return default_vocab


def run_predictions_on_dataset(dataset_path):
    # Load images using your existing function
    images, paths = load_images([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)])

    # Wczytaj wytrenowany model
    model_path = "model12_03__09_00.keras"  # lub .h5
    model = load_model(model_path, custom_objects={"ctc_loss": ctc_loss})

    predictions = model.predict(images)

    # Mock predictions - zastąp rzeczywistymi predykcjami
    vocab = load_vocab()
    mock_predictions = [predictions for _ in images]

    # Wyświetl wyniki w GUI
    display_predictions_in_gui(paths, mock_predictions, vocab)

    return [decode_prediction(p, vocab) for p in mock_predictions]


def decode_prediction(predictions, vocab):
    """Konwertuje output modelu na tekst, zgodnie z logiką prediction_vector_to_label + number_to_label"""
    # Tworzenie słownika z posortowanym vocab (jak w number_to_label)
    vocab_dict = {idx: char for idx, char in enumerate(sorted(vocab))}

    # Obsługa pojedynczej predykcji lub batcha predykcji
    if len(predictions.shape) == 2:  # Pojedyncza sekwencja
        predictions = np.expand_dims(predictions, axis=0)

    result = []
    for seq in predictions:
        decoded_seq = []
        for char in np.argmax(seq, axis=-1):
            if char != len(vocab):  # Ignoruj blank (tak jak w prediction_vector_to_label)
                try:
                    decoded_seq.append(int(char))
                except ValueError:
                    pass

        # Konwersja indeksów na znaki (jak w number_to_label)
        decoded_text = "".join([vocab_dict[num] for num in decoded_seq if num != -1])
        result.append(decoded_text)

    # Jeśli była tylko jedna predykcja, zwróć string zamiast listy z jednym elementem
    if len(result) == 1:
        return result[0]
    return result


# Main application loop
root = tk.Tk()
root.title("Summarizer 2000")

# Dynamic sizing configuration
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)


root.grid_rowconfigure(4, weight=1)

# Added a row for the URL entry
root.grid_columnconfigure(0, weight=1)

# Input text box
input_text_area = tk.Text(root, height=15, width=60)
input_text_area.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

# Model dropdown list
model_label = tk.Label(root, text="Select summarization model:")
model_label.grid(row=1, column=0, padx=5, pady=5, sticky='sw')

models = ["TF-IDF", "Basic", "HuggingFace"]
model_combobox = ttk.Combobox(root, values=models)
model_combobox.grid(row=2, column=0, padx=5, pady=5, sticky='sw')
model_combobox.current(0)  # Ustawienie domyślnego wyboru

# Scale for selecting a percentage
summary_len = 0.3
scale_value = tk.StringVar()  # Variable to store scale value as a string
scale_label = tk.Label(root, text="Select summary length:")
scale_label.grid(row=1, column=0, padx=5, pady=5, sticky='s')  # Positioned just above the scale
scale = tk.Scale(root, from_=0, to=1, resolution=0.1, orient='horizontal', length=200, command=update_scale_label, showvalue=0)
scale.grid(row=2, column=0, padx=5, pady=5, sticky='n')
scale.set(0.3)  # Set default value to 30%
scale_value_label = tk.Label(root, textvariable=scale_value)
scale_value_label.grid(row=2, column=0, padx=5, pady=30, sticky='n')  # Positioned just above the scale

# Summarization button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text, width=20)
summarize_button.grid(row=2, column=0, padx=5, pady=5, sticky='se')

# Count characters without spaces
char_count_button = tk.Button(root, text="Count Characters", command=show_character_count, width=20)
char_count_button.grid(row=1, column=0, padx=5, pady=5, sticky='se')

# Summary text box
result_text_area = tk.Text(root, height=15, width=60)
result_text_area.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

url_label = tk.Label(root, text="Enter URL:")
url_label.grid(row=4, column=0, padx=5, pady=5, sticky='nw')

url_entry = tk.Entry(root, width=165)
url_entry.grid(row=4, column=0, padx=75, pady=5, sticky='nw')

fetch_button = tk.Button(root, text="Fetch from URL", command=fetch_url_text, width=20)
fetch_button.grid(row=4, column=0, padx=5, pady=5, sticky='ne')

# Existing PDF button
pdf_button = tk.Button(root, text="Fetch from PDF file", command=fetch_pdf_text, width=20, background='lightgray')
pdf_button.grid(row=5, column=0, padx=5, pady=5, sticky='ne')

# Modified image processing buttons
img_button = tk.Button(root, text="Handwriting Recognition", command=run_full_pipeline, width=20, bg='#e0f0ff')
img_button.grid(row=5, column=0, padx=5, pady=5, sticky='nw')

prediction_button = tk.Button(root,
                            text="Run Predictions",
                            command=lambda: run_predictions_on_dataset("test_dataset"),
                            width=20)
prediction_button.grid(row=6, column=0, padx=5, pady=5, sticky='sw')

# Run app
root.mainloop()
