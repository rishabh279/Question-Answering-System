import numpy as np
import keras.backend as K
import re
import tarfile

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda, Reshape, add, dot, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras.utils.data_utils import get_file

path = get_file(
    'babi-tasks-v1-2.tar.gz',
    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

# relevant data in the tar file
# there's lots more data in there, check it out if you want!
challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}


def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def get_stories(f):

    data = []
    story = []

    printed = False
    for line in f:
        line = line.decode('utf-8').strip()

        nid, line = line.split(' ', 1)

        if int(nid) == 1:
            story = []

        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)

            story_so_far = [[str(i)] + s for i, s in enumerate(story) if s]

            data.append((story_so_far, q, a))
            story.append('')
        else:
            story.append(tokenize(line))

    return data


def should_flatten(el):
    return not isinstance(el, (str, bytes))


def flatten(l):
    for el in l:
        if should_flatten(el):
            yield from flatten(el)
        else:
            yield el


def vectorize_stories(data, word2idx, story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([[word2idx[w] for w in s] for s in story])
        queries.append([word2idx[w] for w in query])
        answers.append([word2idx[answer]])

    return (
        [pad_sequences(x, maxlen=story_maxlen) for x in inputs],
        pad_sequences(queries, maxlen=query_maxlen),
        np.array(answers)
    )


def stack_inputs(inputs, story_maxsents, story_maxlen):
    for i, story in enumerate(inputs):
        inputs[i] = np.concatenate(
          [
            story,
            np.zeros((story_maxsents - story.shape[0], story_maxlen), 'int')
          ]
        )
    return np.stack(inputs)


def get_data(challenge_type):

    challenge = challenges[challenge_type]

    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

    stories = train_stories + test_stories

    story_maxlen = max(len(s) for x, _, _ in stories for s in x)
    story_maxsents = max((len(x)) for x, _, _, in stories)
    query_maxlen = max(len(x) for _, x, _ in stories)

    vocab = sorted(set(flatten(stories)))
    vocab.insert(0, '<PAD>')
    vocab_size = len(vocab)

    word2idx = {c:i for i, c in enumerate(vocab)}

    inputs_train, queries_train, answers_train = vectorize_stories(
        train_stories,
        word2idx,
        story_maxlen,
        query_maxlen
    )
    print(len(inputs_train))

    inputs_test, queries_test, answers_test = vectorize_stories(
        test_stories,
        word2idx,
        story_maxlen,
        query_maxlen
    )
    inputs_train = stack_inputs(inputs_train, story_maxsents, story_maxlen)
    print(inputs_train.shape)
    inputs_test = stack_inputs(inputs_test, story_maxsents, story_maxlen)

    return train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test,\
           answers_test, story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size


train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, \
answers_test, story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size = get_data('single_supporting_fact_10k')

embedding_dim = 15

input_story = Input((story_maxsents, story_maxlen))
embedded_story = Embedding(vocab)