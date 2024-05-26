import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """Automatic summarization is the process of shortening a set of data computationally, to create a subset (a summary) that represents the most important or relevant information within the original content. Artificial intelligence algorithms are commonly developed and employed to achieve this, specialized for different types of data. Text summarization is usually implemented by natural language processing methods, designed to locate the most informative sentences in a given document. On the other hand, visual content can be summarized using computer vision algorithms. Image summarization is the subject of ongoing research; existing approaches typically attempt to display the most representative images from a given image collection, or generate a video that only includes the most important content from the entire collection. Video summarization algorithms identify and extract from the original video content the most important frames (key-frames), and/or the most important video segments (key-shots), normally in a temporally ordered fashion. Video summaries simply retain a carefully selected subset of the original video frames and, therefore, are not identical to the output of video synopsis algorithms, where new video frames are being synthesized based on the original video content."""

stopwords = list(STOP_WORDS)

nlp = spacy.load('en_core_web_sm')  # Load nlp model

# Tokenize words
doc = nlp(text)  # Apply model to text
tokens = [token.text.lower() for token in doc]  # Tokenize words
punctuation = punctuation + '\n'  # Special characters

# Text cleaning from special characters and stop words
word_frequencies = {}
for word in doc:  # For each word in doc
    # if it's not a stopword and special characer
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_frequencies.keys():  # if it's not already in word frequencies
            word_frequencies[word.text] = 1
        else:
            word_frequencies[word.text] += 1

# max_frequency = max(word_frequencies.values())  # Max occurrence of a word
#
# # for word in word_frequencies.keys(): #  Normalize
# #     word_frequencies[word] = word_frequencies[word] / max_frequency

sentence_tokens = [sent for sent in doc.sents]  # Take out sentences from original text

# Calculate sentence scores basing on word frequencies
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]

# Take only x% of sentences
x = 0.3
select_length = int(len(sentence_tokens) * x)

summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)

print("\nOryginal tekst\n", text)
print("\nSummary\n", summary)

from transformers import pipeline  # pretrained model for NLP task

#summarization - pipeline for text summarization
# model i tokenizer t5-base
# pt - pytorch
summarizer = pipeline("summarization", model='t5-base', tokenizer='t5-base', framework='pt')

# input text, maximum lenght of summary, minimum length, False = model will not use sampling (it will be list)
summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
print(summary[0]['summary_text'])