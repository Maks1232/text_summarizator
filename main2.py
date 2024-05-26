import math
import string
import spacy
from nltk import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

# Before first use:
import nltk


# nltk.download('punkt')
# nltk.download('stopwords')
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
    tf_idf = {}

    for sentence, words_n_number in occurrences_in_sent.items():
        tf_idf_temp = {}

        count_words_in_sentence = len(words_n_number)

        for word, number in words_n_number.items():
            tf_idf_temp[word] = number / count_words_in_sentence * math.log10(df / (float(occurrences_in_text[word]) + 1))

        tf_idf[sentence] = tf_idf_temp

    return tf_idf


# def calculate_tf(occurrences_in_sent):
#     """
#     Function used to calculate word frequency (Term Frequency (TF)) in sentence.
#     :param occurrences_in_sent: Dict with words and word occurrences in sentence for each sentence
#     :return: Dict of Term Frequency (TF)
#     """
#     tf_dict = {}
#
#     for sent, word_n_value in occurrences_in_sent.items():
#         tf_table = {}
#
#         count_words_in_sentence = len(word_n_value)
#         for word, value in word_n_value.items():
#             tf_table[word] = value / count_words_in_sentence
#
#         tf_dict[sent] = tf_table
#
#     return tf_dict
#
#
# def calculate_idf(occurrences_in_sent, occurrences_in_text, df):
#     """
#     Function used to calculate IDF(w, D) = log(N / (DF(w, D) + 1))
#     N - number of sentences
#     DF(w, D) - number of word occurrences in text
#     :param occurrences_in_sent:
#     :param occurrences_in_text:
#     :param df: number of sentences
#     :return: Dict of Inverse Document Frequency (IDF)
#     """
#     idf = {}
#
#     for sentence, words_n_number in occurrences_in_sent.items():
#         idf_table = {}
#
#         for word in words_n_number.keys():
#             idf_table[word] = math.log10(df / (float(occurrences_in_text[word]) + 1))
#
#         idf[sentence] = idf_table
#
#     return idf


# def calculate_tfidf(tf, idf):
#     """
#     Function used to calculate TF-IDF = TF * IDF
#     :param tf: Dict of Term Frequency (TF)
#     :param idf: Dict of Inverse Document Frequency (IDF)
#     :return: Dict of TF-IDF
#     """
#     tf_idf = {}
#
#     for (sent1, word_n_number1), (sent2, word_n_number2) in zip(tf.items(), idf.items()):
#         tf_idf_temp = {}
#         for (word1, number1), (word2, number2) in zip(word_n_number1.items(), word_n_number2.items()):
#             tf_idf_temp[word1] = float(number1 * number2)
#
#         tf_idf[sent1] = tf_idf_temp
#
#     return tf_idf


def score_sentences_by_tfidf(tf_idf):
    """
    Function used to calculate sentence score using TF-IDF
    :param tf_idf: Dict of TF-IDF
    :return: Dict of sentences and their scores
    """

    sent_scores = {}

    for sent, td_idf_dict in tf_idf.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(td_idf_dict)
        for word, score in td_idf_dict.items():
            total_score_per_sentence += score
        sent_scores[sent] = total_score_per_sentence / count_words_in_sentence

    return sent_scores


def generate_summary_tfidf(sentences, sentence_scores_tfidf, summ_length=1.3):
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


def run_summarization(text, summ_length):
    sentences = sentence_tokenizer(text)
    occurrences_in_sent = calculate_occurrences_for_sent(sentences)
    occurrences_in_text = calculate_occurrences_for_text(occurrences_in_sent)
    # tf = calculate_tf(occurrences_in_sent)
    # idf = calculate_idf(occurrences_in_sent, occurrences_in_text, df)
    # tf, idf = calculate_tf_n_idf(occurrences_in_sent, occurrences_in_text, len(sentences))

    tf_idf = calculate_tfidf(occurrences_in_sent, occurrences_in_text, len(sentences))
    sentence_scores_tfidf = score_sentences_by_tfidf(tf_idf)
    summary = generate_summary_tfidf(sentences, sentence_scores_tfidf, summ_length)

    return summary


text_str = '''
Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.

I’ve coached mummy and mom clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”

I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.


“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.

If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.

Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.

So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance.
'''

text_str2 = '''The Importance of Learning Programming

In today's digital age, learning programming is becoming increasingly important. Programming skills are not just for software developers anymore; they are valuable in many fields including finance, healthcare, education, and even art. Understanding how to code can help individuals automate tasks, analyze large sets of data, and develop problem-solving skills.

For instance, in finance, programming is used to create algorithms that can predict market trends and execute trades at optimal times. In healthcare, it helps in the analysis of patient data to improve diagnostics and treatment plans. Educators use programming to develop interactive learning tools that engage students and enhance their understanding of complex subjects.

Furthermore, learning programming fosters creativity. Artists are using code to create digital art, music, and interactive installations. By combining technical skills with creativity, individuals can produce innovative works that push the boundaries of traditional art forms.

Overall, programming is a versatile skill that opens up numerous opportunities. It equips individuals with the tools to innovate and adapt in a rapidly changing world. As technology continues to evolve, the demand for programming skills will only grow, making it an essential skill for the future.'''
# dla tego treshold ustawić 1

result = run_summarization(text_str, 1.3)
print(text_str)
print("---------------------")
print(result)
