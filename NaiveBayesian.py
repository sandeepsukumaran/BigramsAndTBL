import collections


def make_bigrams(input_list):
    """
    Convert given list of words into a stream of bigrams using generator
    :param input_list: list of words to be converted into bigrams
    """
    it = iter(input_list)
    old = next(it,None)
    for new in it:
        yield old, new
        old = new


def calculate_probabilities(tagged_corpus):
    """
    Calculate P(word|tag) and P(tag_i|tag_i-1) of given corpus.
    Returns P(word|tag) as ((word,tag),probability) list.
    Returns P(tag_i-1|tag_i) as ((tag_i-1,tag_i),probability) list.
    :param tagged_corpus: Corpus to calculate model of.
    :return word_tag_prob: P(word|tag)
    :return tag_bigram_prob: P(tag_i-1|tag_i)
    """

    word_tag_tuples = [(tagged_word.split('_')[0], tagged_word.split('_')[1]) for tagged_word in tagged_corpus.split()]

    word_tag_tuples_and_counts = collections.Counter(word_tag_tuples)

    corpus_of_tags = [word_tag_tuple[1] for word_tag_tuple in word_tag_tuples]

    tags_and_counts = collections.Counter(corpus_of_tags)

    tag_bigrams_and_counts = collections.Counter(list(make_bigrams(corpus_of_tags)))

    word_tag_prob = {k: v/tags_and_counts[k[1]] for k, v in word_tag_tuples_and_counts.items()}

    tag_bigram_prob = {k: v/tags_and_counts[k[0]] for k, v in tag_bigrams_and_counts.items()}

    return word_tag_prob, tag_bigram_prob


if __name__ == "__main__":
    with open("HW2_S18_NLP6320_POSTaggedTrainingSet-Unix.txt", "r") as f:
        corpus = f.read()

    word_tag_probabilities, tag_bigram_probabilities = calculate_probabilities(corpus)

    print("P(Word_i|Tag_i):")

    for word_tag_tuple, prob in word_tag_probabilities.items():
        print(str(word_tag_tuple) + " " + str(prob))

    print("P(tag_i|tag_i-1):")

    for tags_tuple, probability in tag_bigram_probabilities.items():
        print(str(tags_tuple) + " " + str(probability))
