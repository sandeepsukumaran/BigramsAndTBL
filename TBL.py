import collections
import functools
from sys import maxsize

# tagset = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP",
# "PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","$","#",
# u"\u201C",u"\u201D","(",")",",",".",":"] 

tag_set = ["NN", "VB"]


def unigram_modelling(original_tagged_corpus):
    """
        Calculate unigram model of given corpus.
        Returns tagged_corpus as (word,tag) list.
        Returns tagged_corpus counts as ((word,tag),count) list.
        Returns unigram counts.
        Returns size of corpus.
    """
    print("Unigram Modelling.", end="")

    word_tag_tuples = [(tagged_word.split('_')[0], tagged_word.split('_')[1]) for tagged_word in original_tagged_corpus.split()]
    print(".", end="")

    corpus_tags = list(set([word_tag_tuple[1] for word_tag_tuple in word_tag_tuples]))
    print(".", end="")

    unigramcounts = collections.Counter([word_tag_tuple[0] for word_tag_tuple in word_tag_tuples])
    print(".", end="")

    corpus_size = functools.reduce(lambda x, y: x + y, unigramcounts.values())
    print(".", end="")

    word_tag_tuples_and_counts = collections.Counter(word_tag_tuples)
    print(".", end="")

    print(".End")

    return word_tag_tuples, word_tag_tuples_and_counts, corpus_tags, unigramcounts, corpus_size


def assign_most_likely_tags_to_words(word_tag_tuples_and_counts, unigrams_and_counts):
    """
        Calculate and return most likely tags for each word
        in corpus.
        Output is dictionary of words and most likely tags.
    """
    print("Assign most likely tags to words.", end="")
    # Dictionary to store most likely tag for each unigram
    tags_ml = {}

    # Find most likely tag: [P(tag|word) is approximated using count(word,tag)]
    for unigram in unigrams_and_counts:
        # Find relevant entries
        lst = [(k, v) for k, v in word_tag_tuples_and_counts.items() if k[0] == unigram]
        # Find maximum count (equivalent to probability) and declare as winner
        winner_tag = max(lst, key=lambda x: x[1])[0][1]
        tags_ml[unigram] = winner_tag

    print(".End")
    return tags_ml


def ml_tagged_corpus(truth_tagged_corpus, ml_tagged_words):
    """
        Return the corpus tagged with most likely tags.
        Format of output is:
        (word,tag)
    """
    print("Assign most likely tags to corpus.", end="", flush=True)
    untagged_corpus = [tagged_word.split('_')[0] for tagged_word in truth_tagged_corpus.split()]
    print(".", end="", flush=True)
    ml_tagged_corpus_list = [(word, ml_tagged_words[word]) for word in untagged_corpus]
    print(".End", flush=True)
    return ml_tagged_corpus_list


def evaluate_end_condition():
    """
        Evaluate and return if error metric is low enough for termination.
    """
    # If latest best transform has score less than threshold, stop
    newest_transform = transforms_queue[-1]

    return newest_transform.score <= 0


class Best:
    """
        Data Structure to store the best transform as
        calculated by the get_best_instance function
    """
    to_tag = None
    from_tag = None
    prev_tag = None
    score = -maxsize


def get_best_instance(truth_tagged_corpus, current_tagged_corpus, tags_in_true_corpus, size_of_corpus):
    """
        Get best transform of given template type.
    """
    # Template type not taken as argument since only one template exists
    best_instance = Best()
    for from_tag in tag_set:
        for to_tag in tag_set:
            # Experimental
            if from_tag == to_tag:
                continue
            # End experimental
            num_good_transforms = collections.Counter()
            num_bad_transforms = collections.Counter()
            for pos in range(size_of_corpus):
                if pos == 0:
                    continue
                if to_tag == truth_tagged_corpus[pos][1] and from_tag == current_tagged_corpus[pos][1]:
                    num_good_transforms[current_tagged_corpus[pos - 1][1]] += 1
                elif from_tag == truth_tagged_corpus[pos][1] and from_tag == current_tagged_corpus[pos][1]:
                    num_bad_transforms[current_tagged_corpus[pos - 1][1]] += 1
            # end for pos
            good_bad_diff = {tag: num_good_transforms[tag] - num_bad_transforms[tag] for tag in tags_in_true_corpus}
            best_z = max(good_bad_diff, key=lambda x: good_bad_diff[x])
            if good_bad_diff[best_z] > best_instance.score:
                best_instance.from_tag = from_tag
                best_instance.to_tag = to_tag
                best_instance.prev_tag = best_z
                best_instance.score = good_bad_diff[best_z]
    # end for to_tag
    # end for from_tag
    return best_instance


def get_best_transform(truth_tagged_corpus, current_tagged_corpus, tags_in_true_corpus, size_of_corpus):
    """
        Find the best transform across all possible templates.
    """
    # No loop since only one template exists
    print("Getting best transform.")
    return get_best_instance(truth_tagged_corpus, current_tagged_corpus, tags_in_true_corpus, size_of_corpus)


def apply_transform(best_transformation, current_tagged_corpus, size_of_corpus):
    """
        Apply the best transform onto given corpus and return new tagged corpus
        Format of the output is same as that of input, namely:
        (word,tag)
    """
    print("Applying best transform to corpus.", end="", flush=True)
    # new_corpus = [(word,best_transform.to_tag) if old_tag==best_transform.from_tag else (word,old_tag) for (word,
    # old_tag) in tagged_corpus]
    new_corpus = [current_tagged_corpus[0]]
    prev = current_tagged_corpus[0][1]
    for i in range(1, size_of_corpus):
        word, old_tag = current_tagged_corpus[i]
        if old_tag == best_transformation.from_tag and prev == best_transformation.prev_tag:
            new_corpus.append((word, best_transformation.to_tag))
            prev = best_transformation.to_tag
        else:
            new_corpus.append((word, old_tag))
            prev = old_tag
    # end for i
    print(".End")
    return new_corpus


def print_queued_transforms(transformations_queue):
    """
        Print the transformations generated by the program.
        Main output of the program.
    """
    print("Calculated Transforms:")
    print("=" * 22)
    for transform in transformations_queue:
        print("Change tag from " + str(transform.from_tag) + " to " + str(transform.to_tag) + " if prev tag is " + str(
            transform.prev_tag))
        print("Score: " + str(transform.score))


if __name__ == "__main__":
    with open("HW2_S18_NLP6320_POSTaggedTrainingSet-Unix.txt", "r") as f:
        corpus = f.read()

    transforms_queue = []

    true_tagged_corpus, word_tag_tuples_and_counts, corpus_tags, unigram_counts, corpus_size = unigram_modelling(corpus)

    most_likely_tagged_words = assign_most_likely_tags_to_words(word_tag_tuples_and_counts, unigram_counts)

    tagged_corpus = ml_tagged_corpus(corpus, most_likely_tagged_words)

    print("Most likely tag for sentence:")
    sentence = ['The', 'president', 'wants', 'to', 'control', 'the', 'board', '\'s', 'control']
    for word in sentence:
        print(word+"_"+most_likely_tagged_words[word], end=" ")
    print(".")

    while True:
        # Get potential relevant templates - irrelevant since
        # only one template

        best_transform = get_best_transform(true_tagged_corpus, tagged_corpus, corpus_tags, corpus_size)

        tagged_corpus = apply_transform(best_transform, tagged_corpus, corpus_size)

        transforms_queue.append(best_transform)
        print("Enqueued a transform.")

        if evaluate_end_condition():
            del transforms_queue[-1]
            break
        # if len(transforms_queue) >= 20:
        #   break
    # end while

    print_queued_transforms(transforms_queue)
