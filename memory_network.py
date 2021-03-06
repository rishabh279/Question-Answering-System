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
embedded_story = Embedding(vocab_size, embedding_dim)(input_story)
embedded_story = Lambda(lambda x: K.sum(x, axis=2))(embedded_story)
print("input_story_.shape, embedded_story.shape:", input_story.shape, embedded_story.shape)

input_question = Input((query_maxlen,))
embedded_question = Embedding(vocab_size, embedding_dim)(input_question)
embedded_question = Lambda(lambda x: K.sum(x, axis=1))(embedded_question)

embedded_question = Reshape((1, embedding_dim))(embedded_question)
print("inp_q.shape, emb_q.shape:", input_question.shape, embedded_question.shape)

x = dot([embedded_story, embedded_question], 2)
x = Reshape((story_maxsents,))(x)
x = Activation('softmax')(x)
story_weights = Reshape((story_maxsents, 1))(x)
print("story_weights.shape:", story_weights.shape)

x = dot([story_weights, embedded_story], 1)
x = Reshape((embedding_dim,))(x)
ans = Dense(vocab_size, activation='softmax')(x)

model = Model([input_story, input_question], ans)

model.compile(
    optimizer=RMSprop(lr=1e-2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)
model.summary()
r = model.fit(
    [inputs_train, queries_train],
    answers_train,
    epochs=4,
    batch_size=32,
    validation_data=([inputs_test, queries_test], answers_test)
)

debug_model = Model([input_story, input_question], story_weights)

story_idx = np.random.choice(len(train_stories))

i = inputs_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w = debug_model.predict([i, q]).flatten()

story, question, ans = train_stories[story_idx]
print("story:\n")
for i, line in enumerate(story):
  print("{:1.5f}".format(w[i]), "\t", " ".join(line))

print("question:", " ".join(question))
print("answer:", ans)



# pause so we can see the output
input("Hit enter to continue\n\n")

### two supporting facts ###

train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, \
answers_test, story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size = get_data('single_supporting_fact_10k')

embedding_dim = 30


def embed_and_sum(x, axis=2):
    x = Embedding(vocab_size, embedding_dim)(x)
    x = Lambda(lambda x: K.sum(x, axis))(x)
    return x


input_story = Input((story_maxsents, story_maxlen))
input_question = Input((query_maxlen, ))

embedded_story = embed_and_sum(input_story)
embedded_question = embed_and_sum(input_question, 1)

dense_layer = Dense(embedding_dim, activation='elu')


def hop(query, story):

    x = Reshape((1, embedding_dim))(query)
    x = dot([story, x], 2)
    x = Reshape((story_maxsents,))(x)
    x = Activation('softmax')(x)
    story_weights = Reshape((story_maxsents, 1))(x)

    story_embedding2 = embed_and_sum(input_story)
    x = dot([story_weights, story_embedding2], 1)
    x = Reshape((embedding_dim, ))(x)
    x = dense_layer(x)
    return x, story_embedding2, story_weights


ans1, embedded_story, story_weights1 = hop(embedded_question, embedded_story)
ans2, _, story_weights2 = hop(ans1, embedded_story)

ans = Dense(vocab_size, activation='softmax')(ans2)

model2 = Model([input_story, input_question], ans)

model2.compile(
    optimizer=RMSprop(lr=5e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

r = model2.fit(
    [inputs_train, queries_train],
    answers_train,
    epochs=1,
    batch_size=32,
    validation_data=([inputs_test, queries_test], answers_test)
)

debug_model2 = Model(
    [input_story, input_question],
    [story_weights1, story_weights2]
)

story_idx = np.random.choice(len(train_stories))

i = inputs_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w1, w2 = debug_model2.predict([i, q])
w1 = w1.flatten()
w2 = w2.flatten()

story, question, ans = train_stories[story_idx]
print("story:\n")
for j, line in enumerate(story):
  print("{:1.5f}".format(w1[j]), "\t", "{:1.5f}".format(w2[j]), "\t", " ".join(line))

print("question:", " ".join(question))
print("answer:", ans)
print("prediction:", vocab[ np.argmax(model2.predict([i, q])[0])])
