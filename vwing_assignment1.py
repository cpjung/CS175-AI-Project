import re
from typing import Iterable, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import gutenberg
# This import has been commented out because it's slow and you don't need it for all of the
# problems.  Feel free to comment or uncomment it as you neeed.
# from nltk.book import *


###################
# Helper function #
###################
def load_hamlet():
    """
    Loads the contents of the play Hamlet into a string.

    Returns
    -------
    str
        The one big, raw, unprocessed string.

    Example
    -------
    >>> document = load_hamlet()
    >>> document[:80]
    '[The Tragedie of Hamlet by William Shakespeare 1599]\n\n\nActus Primus. Scoena Prim'
    """
    return gutenberg.raw('shakespeare-hamlet.txt')


#############
# Problem 1 #
#############
def tokenize_simple(document):
    """
    Performs a simple form of tokenization on a raw document string.

    Your function should perform the following steps:
        Convert the document to lower case
        Replace all punctuation marks with single spaces
        Replace all whitespace clumps with single spaces
        Remove all leading and trailing whitespace from the document string
        Split the document on single spaces into a list of tokens
        Return the list of tokens

    Note that this tokenizer doesn't keep contractions intact or respect sentence boundaries, and
    it doesn't need to respect non-ASCII unicode.

    Parameters
    ----------
    document : str
        A string containing the full raw text of a document.

    Returns
    -------
    List of strs
        One string for each token in the document, in the same order as in the original document.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokenize_simple(document)[:10]
    ['the', 'tragedie', 'of', 'hamlet', 'by', 'william', 'shakespeare', '1599', 'actus', 'primus']
    """
    doc = document.lower()                                           #convert to lower case
    doc = re.sub(r'[\.\?\{\}\[\]\\\|\(\)!,:\'-;\"]',' ',doc)        #remove punctuations
    doc = re.sub(r'[\s]+',' ', doc)                                  #remove whitespace clumps
    doc =  doc.strip()                                               #remove trailing whitespace
    return doc.split(' ')


def remove_stopwords_inner(tokens, stopwords):
    """
    Removes stopwords from the given tokens.

    Parameters
    ----------
    tokens : Iterable[str]
        The tokens from which to remove stopwords.

    stopwords : Iterable[str]
        The tokens to remove.

    Returns
    -------
    List[str]
        The input tokens with stopwords removed.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_simple(document)
    >>> remove_stopwords_inner(tokens, ['of', 'to'])[:10]
    ['the', 'tragedie', 'hamlet', 'by', 'william', 'shakespeare', '1599', 'actus', 'primus', 'scoena']
    """
    # convert to a set for speed of lookup in the next step
    stopwords = set(stopwords)
    return [x for x in tokens if x not in stopwords]


def remove_stopwords(tokens):
    """
    Removes the a static set of stopwords from the given tokens using remove_stopwords_inner.

    This function should remove the 100 most common English words from the given tokens
    Parameters
    ----------
    tokens : Iterable[str]
        The tokens from which to remove stopwords.

    Returns
    -------
    List[str]
        The input tokens with stopwords removed.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_simple(document)
    >>> remove_stopwords(tokens)[:10]
    ['tragedie', 'hamlet', 'william', 'shakespeare', '1599', 'actus', 'primus', 'scoena', 'prima', 'enter']
    """
    # the stopwords to remove
    stopwords = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for',
                 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
                 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
                 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
                 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
                 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
                 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
                 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
                 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
                 'most', 'us']
    return remove_stopwords_inner(tokens,stopwords)


def get_bag_of_words(tokens):
    """
    Returns a bag-of-words representation of the given tokens

    Parameters
    ----------
    tokens : Iterable[str]
        The tokens to convert to bag of words

    Returns
    -------
    Counter
        The bag-of-words count for each token

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_simple(document)
    >>> tokens = remove_stopwords(tokens)
    >>> bow = get_bag_of_words(tokens)
    >>> bow.most_common(10)
    [('i', 574), ('ham', 337), ('is', 328), ('d', 223), ('lord', 211), ('haue', 178), ('king', 172), ('s', 124), ('are', 123), ('thou', 107)]
    """
    return Counter(tokens)
    #return list(c.items())


def bow_pipeline_python(document):
    """
    Uses the functions tokenize_simple, remove_stopwords, and get_bag_of_words to implement a full,
    Python-only, bag-of-words processing pipeline.

    Parameters
    ----------
    document : str
        The full, raw text of the Hamlet loaded from data/hamlet.txt

    Returns
    -------
    Counter
        A bag-of-words representation of the given document.

    Example
    -------
    >>> document = load_hamlet()
    >>> bow = bow_pipeline_python(document)
    >>> bow.most_common(10)
    [('i', 574), ('ham', 337), ('is', 328), ('d', 223), ('lord', 211), ('haue', 178), ('king', 172), ('s', 124), ('are', 123), ('thou', 107)]
    """
    doc = tokenize_simple(document)
    doc = remove_stopwords(doc)
    bag = get_bag_of_words(doc)
    return bag


##########################################
# Problem 2 - Already Filled In For You! #
##########################################
def tokenize_nltk(document):
    """
    Use the nltk.tokenize.word_tokenize to tokenize a lower-cased version of the given document.

    Note that this tokenizer DOES split contractions or respect sentence boundaries, and it
    doesn't need to respect non-ASCII unicode.

    Parameters
    ----------
    document : str
        A string containing the full raw text of a document.

    Returns
    -------
    List of strs
        One string for each token in the document, in the same order as in the original document.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_nltk(document)
    >>> tokens[:10]
    ['[', 'the', 'tragedie', 'of', 'hamlet', 'by', 'william', 'shakespeare', '1599', ']']
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(document.lower())


def remove_stopwords_nltk(tokens):
    """
    Remove the NLTK English stopwords from the given tokens using remove_stopwords_inner.

    Parameters
    ----------
    tokens : Iterable[str]
        The tokens from which to remove stopwords.

    Returns
    -------
    List[str]
        The input tokens with NLTK's English stopwords removed.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_nltk(document)
    >>> tokens = remove_stopwords_nltk(tokens)
    >>> tokens[:10]
    ['[', 'tragedie', 'hamlet', 'william', 'shakespeare', '1599', ']', 'actus', 'primus', '.']
    """
    from nltk.corpus import stopwords
    return remove_stopwords_inner(tokens, stopwords=stopwords.words('english'))


def get_bag_of_words_nltk(tokens):
    """
    Returns a bag-of-words representation of the given tokens that you create with NLTK's FreqDist
    class.

    Note that FreqDist is a subclass of Counter.

    Parameters
    ----------
    tokens : Iterable[str]
        The tokens to convert to bag of words.

    Returns
    -------
    FreqDist
        The bag-of-words representation of the given document.

    Example
    -------
    >>> document = load_hamlet()
    >>> tokens = tokenize_nltk(document)
    >>> tokens = remove_stopwords_nltk(tokens)
    >>> bow = get_bag_of_words(tokens)
    >>> bow.most_common(10)
    [(',', 2892), ('.', 1877), (':', 566), ('?', 459), ('ham', 337), (';', 298), ('lord', 211), ("'d", 200), ('haue', 175), ('king', 172)]
    """
    return FreqDist(tokens)


def bow_pipeline_nltk(document):
    """
    Uses the functions tokenize_nltk, remove_stopwords_nltk, and get_bag_of_words_nltk to
    implement full bag-of-words processing pipeline using the NLTK.

    Parameters
    ----------
    document : str
        The full, raw text of the Hamlet loaded from data/hamlet.txt

    Returns
    -------
    Counter
        A bag-of-words representation of the given document.

    Example
    -------
    >>> document = load_hamlet()
    >>> bow = bow_pipeline_nltk(document)
    >>> bow.most_common(10)
    [(',', 2892), ('.', 1877), (':', 566), ('?', 459), ('ham', 337), (';', 298), ('lord', 211), ("'d", 200), ('haue', 175), ('king', 172)]
    """
    tokens = tokenize_nltk(document)
    tokens = remove_stopwords_nltk(tokens)
    return get_bag_of_words_nltk(tokens)


#############
# Problem 3 #
#############
def most_common_tokens_inner(bag_of_words, n_tokens):
    """
    Returns the most common tokens in the given bag_of_words.

    Parameters
    ----------
    bag_of_words : Counter
        The bag-of-words representation for a document.
    n_terms : int
        The number of terms to return.

    Returns
    -------
    List[Tuple[str, int]]
        The n_tokens most common tokens in bag_of_words, sorted from most common to least common.

    Example
    -------
    >>> bow = Counter({'token1': 12, 'token2': 23, 'token3': 999})
    >>> most_common_tokens_inner(bow, 2)
    [('token3', 999), ('token2', 23)]
    """
    return bag_of_words.most_common(n_tokens)


def most_common_hamlet_tokens_python_pipeline(n_tokens=10):
    """
    Uses load_hamlet, bow_pipeline_python, and most_common_tokens_inner to return the 10 most common
    tokens from the hamlet text that were found using the Python pipeline.

    Note: you'll get 0 points for this problem (and any other) if you hardcode the solution.

    Parameters
    ----------
    n_tokens : int
        The number of most common tokens to return.  Default: 10.

    Returns
    -------
    List[Tuple[str, int]]
        The n_tokens most common tokens in the hamlet text according to the pure Python processing
        pipeline, sorted from most common to least common.

    Example
    -------
    >>> most_common_hamlet_tokens_python_pipeline()
    [('i', 574), ('ham', 337), ('is', 328), ('d', 223), ('lord', 211), ('haue', 178), ('king', 172), ('s', 124), ('are', 123), ('thou', 107)]
    """
    doc = load_hamlet()
    bag = bow_pipeline_python(doc)
    return most_common_tokens_inner(bag,n_tokens)


def most_common_hamlet_tokens_nltk_pipeline(n_tokens=10):
    """
    Uses load_hamlet, bow_pipeline_nltk, and most_common_tokens_inner to return the 10 most common
    tokens from the hamlet text that were found using the NLTK pipeline.

    Note: you'll get 0 points for this problem (and any other) if you hardcode the solution.

    Parameters
    ----------
    n_tokens : int
        The number of most common tokens to return.  Default: 10.

    Returns
    -------
    List[Tuple[str, int]]
        The n_tokens most common tokens in the hamlet text according to the NLTK processing
        pipeline, sorted from most common to least common.

    Example
    -------
    >>> most_common_hamlet_tokens_nltk_pipeline()
    [(',', 2892), ('.', 1877), (':', 566), ('?', 459), ('ham', 337), (';', 298), ('lord', 211), ("'d", 200), ('haue', 175), ('king', 172)]
    """
    doc = load_hamlet()
    bag = bow_pipeline_nltk(doc)
    return most_common_tokens_inner(bag,n_tokens)


def remove_infrequent_tokens(bag_of_words, min_freq=2):
    """
    Remove the most infrequent tokens from the given bag-of-words.

    Parameters
    ----------
    bag_of_words : Counter
        The bag-of-words representation for a document.
    min_freq : int
        The minimum number of times a token must appear in the document in order to not be
        filtered out.

    Returns
    -------
    Counter
        A bag-of-words which has been filtered to remove the least common tokens.

    Example
    -------
    >>> bow = Counter({'token1': 1, 'token2': 2, 'token3': 3, 'token4': 1})
    >>> remove_infrequent_tokens(bow, min_freq=2)
    Counter({'token3': 3, 'token2': 2})
    """
    bag = bag_of_words.copy()
    for w in bag_of_words.keys():
        if bag_of_words[w] < min_freq:
            bag.pop(w)
    return bag


def remove_punctuation(bag_of_words):
    """
    Remove ASCII punctuation tokens from the given bag-of-words.

    Parameters
    ----------
    bag_of_words : Counter
        The bag-of-words representation for a document.

    Returns
    -------
    Counter
        A bag-of-words which has been filtered to remove punctuation tokens.

    Example
    -------
    >>> bow = Counter({'hamlet': 1, 'prince': 2, ',': 3, 'this-stays': 4, ':': 5})
    >>> remove_punctuation(bow)
    Counter({'this-stays': 4, 'prince': 2, 'hamlet': 1})
    """
    import string
    punctuation = set(string.punctuation)
    bag = bag_of_words
    for p in punctuation:
        if p in bag.keys():
            bag.pop(p)
    return bag


#############
# PROBLEM 4 #
#############
def percent_str(count, total):
    """
    Return a string denoting the percent of total that count is rounded to four decimal places.

    Parameters
    ----------
    count : numeric
        A number indicating the part of the total which we are interested in
    total : numeric
        A number indicating the quantity of the total of which count is a part.

    Returns
    -------
    str
        Return a number denoting the percent of total that count is.

    Raises
    ------
        ValueError if count is < 0 or total is <= 0

    Examples
    --------
    >>> percent_str(3, 42)
    '7.1429'

    >>> percent_str(0, 12)
    '0.0000'

    >>> percent_str(-3, 12)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 45, in percent_str
    ValueError: must have count >= 0

    >>> percent_str(3, 0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 47, in percent_str
    ValueError: must have total > 0
    """
    if count < 0:
        raise ValueError("must have count >= 0")
    if total <= 0:
        raise ValueError("must have total > 0")
    percent = count / total * 100 #calculate percentage
    return "{0:.4f}".format(percent) #return string formatted to 4 decimal places


def word_count_percent(text, word):
    """
    Returns the number of instances of the given word in the given text and a formated string
    listing the percentage of the text which is comprized of this word.

    Parameters
    ----------
    text : nltk.text.Text
        An NLTK text.
    word : str
        A word to search for in the given text.

    Returns
    -------
    str
        The percentage of the text which is comprised of this word, rounded to two decimal places.

    Example
    -------
    >>> from nltk.book import *
    >>> word_count_percent(text1, 'and')
    '2.3096'
    """
    percent = text.count(word) / len(text) * 100 #the percentage as a number
    return "{0:.4f}".format(percent) #returns the formatted string


def problem_4c():
    """
    Returns the percent_strs for several words in several NLTK texts.

    Note: the outer loop should be over words and the inner loop should be over texts.
    Note: you'll get 0 points for this problem (and any other) if you hardcode the solution.

    Returns
    -------
    Tuple[Tuple[str, str, str]]
        The percent strings for each of the words:
            "the",
            "and",
            "King", and
            "president"
        in each of the texts
            text1 (Moby Dick),
            text6 (Monty Python and the Holy Grail), and
            text7 (Wall Street Journal).

    Examples
    --------
    >>> problem_4c()
    (('5.2607', '1.7622', '4.0178'), ('2.3096', '0.7957', '1.5009'), ('0.0142', '0.1591', '0.0000'), ('0.0004', '0.0000', '0.1321'))
    """
    from nltk.book import text1, text6, text7
    words = ['the','and','King','president']
    texts = [text1,text6,text7]
    return [[word_count_percent(t,w) for t in texts] for w in words]
    
    


#############
# Problem 5 #
#############
def plot_wordlength_histogram(input_text):
    """
    Plot a histogram of the wordlengths in the given text.

    The x- and y-labels of the plot must be accurately set and the title must be set to "Word
    Length Histogram for " followed by the name of the input_text.  Finally, the plot's x limits
    must be set to [1, the-maximum-word-length-in-this-text].

    Note that we're working with the figure and axes objects directly rather than using plt.
    This pattern is useful for separating out code which creates your plots from code which shows
    or saves your plots, though it's easy to move between it and the "plt." pattern you'll see in
    a lot of online tutorials and help documents.  (Specifically, plt.gcf() will get a handle to
    the current figure and plt.gca() will get a handle to the axes on the current figure.)

    In this fig-and-axes pattern, normal plotting commands such as plt.plot(...) or plt.hist(...)
    become axes.plot(...) and axes.hist(...), while commands for setting properties of the plot
    such as plt.xlabel(...) and plt.xlim(...) become axes.set_xlabel(...) and axes.set_xlim(...).
    You can show your plot by running fig.show() or save it with fig.savefig(...).  These commands
    accomplish the exact same things as their counterparts in plt, they're simply more explicit.

    The use of plt.subplots() at the start also makes it easy to expand this pattern to include
    several axes on the same figure which is often quite useful.

    Parameters
    ----------
    input_text : nltk.text.Text
        An NLTK text.

    Returns
    -------
    tuple(fig, axes)
        A handle to a matplotlib figure and a handle to the lone set of axes on that figure.

    Example
    -------
    >>> from nltk.book import *
    >>> fig, axes = plot_wordlength_histogram(text1)
    >>> type(fig), type(axes)
    (<class 'matplotlib.figure.Figure'>, <class 'matplotlib.axes._subplots.AxesSubplot'>)
    """
    fig, axes = plt.subplots()
    axes.set_xlabel("Word Length")
    axes.set_ylabel("Number of Tokens")
    axes.set_title("Word Length Histogram for " + input_text.name)
    fd = input_text.vocab(); #frequency distribution for input_text
    wordLengths = []
    for w in fd.keys():
        for i in range(0,fd[w]):
            wordLengths.append(len(w))
    axes.set_xlim(1, max(wordLengths))
    axes.hist(wordLengths,max(wordLengths))
    return fig, axes
	


if __name__ == '__main__':
    # You may use this section as you please, but its contents won't be
    # graded.
    pass

