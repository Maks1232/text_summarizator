import math
import nltk
from nltk import PorterStemmer
import tkinter as tk
from tkinter import ttk
import requests  # Library for making HTTP requests to fetch website content
from bs4 import BeautifulSoup  # Library for parsing HTML content
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

# Dodanie przycisku do wybierania pliku PDF
pdf_button = tk.Button(root, text="Fetch from PDF file", command=fetch_pdf_text, width=20, background='lightgray')
pdf_button.grid(row=5, column=0, padx=5, pady=5, sticky='ne')

# Run app
root.mainloop()
