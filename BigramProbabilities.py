import collections


def make_bigrams(input_list):
    """
    Generator to convert given list of words into bigrams
    :param input_list: list of words to be converted into bigrams
    :return: generator that yields all bigrams in subsequent calls
    """
    it = iter(input_list)
    old = next(it, None)
    for new in it:
        yield old, new
        old = new


def laplace_smooth(unigram_counts, bigram_counts):
    """
    Computes LaPlace smoothed probabilities of bigrams
    :param unigram_counts: collections.Counter with unigrams as key and frequency as value
    :param bigram_counts: collections.Counter with bigrams as key and frequency as value
    :return: list of all possible bigrams from unigrams is corpus
    """

    # size of vocabulary
    v = len(unigram_counts)

    print("Calculating laplace smoothed probabilities.", flush=True)

    laplace_smoothed_bigrams = collections.Counter({(i, j): (bigram_counts[(i, j)] + 1) / (unigram_counts[i] + v) for i in unigram_counts for j in unigram_counts})

    print("\nWriting laplace smoothed values to file.", flush=True)

    with open('laplace.txt', 'w') as laplace_output_file:
        for k, v in laplace_smoothed_bigrams.items():
            laplace_output_file.write("("+str(k)+" , "+str(v)+")\n")

    return laplace_smoothed_bigrams.keys()


def good_turing(bigram_counts, all_possible_bigrams, corpus_size):
    """
    Prints Good Turing discounting based smoothed bigram probabilities to file.
    :param bigram_counts: collections.Counter of with bigrams in corpus as keys and their frequencies as values
    :param all_possible_bigrams: list of all possible bigrams formed from combination of unigrams in corpus
    :param corpus_size: number of words in corpus
    :return: None
    """
    print("\nMaking Good-Turing Histogram.")

    # Make histogram of bigram frequencies
    hist = collections.Counter([bigram_counts[bigram] for bigram in all_possible_bigrams])

    gt_bigram_probabilities = collections.Counter()

    for bigram in all_possible_bigrams:
        bc = bigram_counts[bigram]
        if bc == 0:
            gt_bigram_probabilities[bigram] = hist[1]/corpus_size
        else:
            gt_bigram_probabilities[bigram] = ((bc+1)*hist[bc+1]/hist[bc])/corpus_size

    print("\nWriting GT discounting smoothed values to file.", flush=True)

    with open('gt.txt', 'w') as gt_output_file:
        for k, v in gt_bigram_probabilities.items():
            gt_output_file.write("("+str(k)+" , "+str(v)+")\n")

    return


if __name__ == "__main__":
    with open("HW2_S18_NLP6320-NLPCorpusTreebank2Parts-CorpusA-Unix.txt", "r") as f:
        corpus = f.read()

    words = [word.strip() for word in corpus.split()]
    N = len(words)

    print("Making bigrams.")
    bigrams = list(make_bigrams(words))

    bigrams_and_counts = collections.Counter(bigrams)

    unigrams_and_counts = collections.Counter(words)

    # No smoothing probabilites
    print("Calculating un-smoothed probabilities.", flush=True)

    bigram_probabilities = collections.Counter({k: v / unigrams_and_counts[k[0]] for k, v in bigrams_and_counts.items()})

    print("\nWriting un-smoothed probabilities to file.", flush=True)

    with open('unsmoothed.txt', 'w') as f:
        for k, v in bigram_probabilities.items():
            f.write("("+str(k)+" ,  "+str(v)+")\n")

    all_bigrams = laplace_smooth(unigrams_and_counts, bigrams_and_counts)

    good_turing(bigrams_and_counts, all_bigrams, N)

    print("\nDone.")
