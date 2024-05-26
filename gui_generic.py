import math
import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import ttk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import re
from transformers import pipeline

# Download the 'punkt' and 'stopwords' resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    average = (sumValues / len(sentenceValue))
    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary


def run_tf_idf_summarization(text):
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    freq_matrix = _create_frequency_matrix(sentences)
    tf_matrix = _create_tf_matrix(freq_matrix)
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = _score_sentences(tf_idf_matrix)
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary


def run_spacy_summarization(text):
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
    # Taking ~30% sentences as a upper limit for text summary
    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([sent.text for sent in summary])
    return final_summary


def run_hugging_face_transformer(text, max_length=200, min_length=30):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]


def run_summarization(text, model):
    if model == "TF-IDF":
        return run_tf_idf_summarization(text)
    elif model == "spaCy":
        return run_spacy_summarization(text)
    elif model == "HuggingFace":
        max_len = int(0.3 * count_characters_without_spaces(text))
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
        result_text_area.insert(tk.END, "Proszę wprowadzić tekst do podsumowania.")
        return

    result = run_summarization(input_text, selected_model)
    result_text_area.delete("1.0", tk.END)
    result_text_area.insert(tk.END, result)


def show_character_count():
    input_text = input_text_area.get("1.0", tk.END).strip()
    char_count = count_characters_without_spaces(input_text)
    result_text_area.delete("1.0", tk.END)
    result_text_area.insert(tk.END, f"Ilość znaków bez spacji i enterów: {char_count}")


# Main application loop
root = tk.Tk()
root.title("Aplikacja do Sumaryzacji Tekstu")

# Dynamic sizing configuration
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)

# Input text box
input_text_area = tk.Text(root, height=10, width=60)
input_text_area.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

# Model dropdown list
model_label = tk.Label(root, text="Wybierz model do sumaryzacji:")
model_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')

models = ["TF-IDF", "spaCy", "HuggingFace"]
model_combobox = ttk.Combobox(root, values=models)
model_combobox.grid(row=2, column=0, padx=5, pady=5, sticky='w')
model_combobox.current(0)  # Ustawienie domyślnego wyboru

# Summarization button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text)
summarize_button.grid(row=2, column=0, padx=5, pady=5, sticky='e')

# Count characters whithout spaces
char_count_button = tk.Button(root, text="Count Characters", command=show_character_count)
char_count_button.grid(row=2, column=0, padx=5, pady=5, sticky='n')

# Summary text box
result_text_area = tk.Text(root, height=10, width=60)
result_text_area.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

# Run app
root.mainloop()