from a_config import *
from methods import *
from numpy.random import seed
import sys

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    tf.random.set_seed(seed)


sd = int(sys.argv[1])
seed_tensorflow(sd)


###############################
#### run model and get acc ####
###############################
def get_x_y_a(train_file, num_classes, word2vec_len, input_size, word2vec, data_index=None, ori_lines=None):
    # read in lines
    train_lines = open(train_file, 'r', encoding='utf-8').readlines()
    if data_index:
        train_lines = [train_lines[ind] for ind in data_index]
    if ori_lines:
        train_lines.extend(ori_lines)

    shuffle(train_lines)
    num_lines = len(train_lines)
    print('num of lines: ', num_lines)

    # initialize x and y matrix
    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, word2vec_len))
    except:
        print("Error!", num_lines, input_size, word2vec_len)
    y_matrix = np.zeros((num_lines, num_classes))

    # insert values
    for i, line in enumerate(train_lines):

        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]

        # insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]]  # cut off if too long
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]

        # insert y
        y_matrix[i][label] = 1.0

    return x_matrix, y_matrix


def line2matrix(lines):
    shuffle(lines)
    num_lines = len(lines)
    # initialize x and y matrix
    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, word2vec_len))
    except:
        print("Error!", num_lines, input_size, word2vec_len)
    y_matrix = np.zeros((num_lines, num_classes))

    # insert values
    for i, line in enumerate(lines):

        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]

        # insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]]  # cut off if too long
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]

        # insert y
        y_matrix[i][label] = 1.0

    return x_matrix, y_matrix


def run_cnn(train_file, test_file, num_classes, data_index, ori_lines=None, dev_data=None):
    # initialize model
    model = build_cnn(input_size, word2vec_len, num_classes)

    # load data
    train_x, train_y = get_x_y_a(train_file, num_classes, word2vec_len, input_size, word2vec, data_index, ori_lines)
    test_x, test_y = get_x_y_a(test_file, num_classes, word2vec_len, input_size, word2vec)

    # implement early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    if dev_data:
        model.fit(train_x,
                  train_y,
                  epochs=100000,
                  callbacks=callbacks,
                  validation_data=dev_data,
                  batch_size=1024,
                  shuffle=True,
                  verbose=0)
    else:
        # train model
        model.fit(train_x,
                  train_y,
                  epochs=100000,
                  callbacks=callbacks,
                  validation_split=0.1,
                  batch_size=1024,
                  shuffle=True,
                  verbose=0)
    # model.save('checkpoints/lol')
    # model = load_model('checkpoints/lol')

    # evaluate model
    y_pred = model.predict(test_x)
    test_y_cat = one_hot_to_categorical(test_y)
    y_pred_cat = one_hot_to_categorical(y_pred)
    acc = accuracy_score(test_y_cat, y_pred_cat)

    # clean memory???
    train_x, train_y, test_x, test_y, model = None, None, None, None, None
    gc.collect()

    # return the accuracy
    # print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
    return acc


###############################
############ main #############
###############################

if __name__ == "__main__":
    dev_split = 0.1
    aug_num = 2

    # for each method
    for a_method in a_methods:

        writer = open(f'outputs_alpha/{a_method}_{aug_num}_{get_now_str()}_seed_{str(sd)}.txt', 'a')

        # for each size dataset
        for size in sizes:

            writer.write(size + '\n')

            # get all six datasets
            dataset_folders = ['../../data/train_for_text_classification/' + s for s in datasets]

            # for storing the performances
            ori_performances = []
            performances = {alpha: [] for alpha in alphas}

            # for each dataset
            for i in range(len(dataset_folders)):
                # initialize all the variables
                dataset_folder = dataset_folders[i]
                dataset = datasets[i]
                print('%s training begins' % dataset)

                num_classes = num_classes_list[i]
                input_size = input_size_list[i]
                word2vec_pickle = dataset_folder + '/word2vec.p'
                word2vec = load_pickle(word2vec_pickle)

                # without aug model performance
                train_path = '../../data/train_for_text_classification/%s/train_ori.txt' % dataset_folder
                test_path = '../../data/train_for_text_classification/%s/test.txt' % dataset
                ori_train_lines = open(train_path, 'r', encoding='utf-8').readlines()

                # random sample
                ori_random_index = random.sample(range(0, lines[dataset]), min(lines[dataset], n_sizes[size]))
                if dev_split > 0:
                    dev_index = ori_random_index[:int(dev_split * len(ori_random_index))]
                    ori_random_index = ori_random_index[int(dev_split * len(ori_random_index)):]
                    # print(ori_random_index)
                    dev_lines = [ori_train_lines[ind] for ind in dev_index]
                    dev_data = line2matrix(dev_lines)
                else:
                    dev_data = None

                acc = run_cnn(train_path, test_path, num_classes, data_index=ori_random_index, dev_data=dev_data)
                ori_performances.append(acc)
                print('ori performance:%f' % acc)
                writer.write('%s\t%f\n' % (dataset, acc))

                ori_train_lines = [ori_train_lines[ind] for ind in ori_random_index]

                # test each alpha value
                for alpha in alphas:
                    random_index = []
                    for r in ori_random_index:
                        gap = random.sample(range(0, 5), aug_num)
                        random_index.extend([r * 5 + g for g in gap])
                    # random_index = [r * 5 + random.randint(0, 4) for r in ori_random_index]
                    train_path = '../../data/train_for_text_classification/%s/%s_%s_%s_n5.txt' % (
                        dataset, dataset, a_method, alpha)
                    test_path = '../../data/train_for_text_classification/%s/test.txt' % dataset
                    acc = run_cnn(train_path, test_path, num_classes, data_index=random_index,
                                  ori_lines=ori_train_lines, dev_data=dev_data)
                    print('alpha %s performance:%f' % (alpha, acc))
                    performances[alpha].append(acc)
                    writer.write(dataset + '_' + str(alpha) + '\t' + str(acc) + '\n')

            print(performances)
            print(ori_performances)
            print(sum(ori_performances) / len(ori_performances))
            writer.write(str('\t'.join(str(i) for i in ori_performances)) + '\n')
            for alpha in performances:
                # line = str(alpha) + '\t' + str(sum(performances[alpha]) / len(performances[alpha]))
                line = str(alpha) + '\t' + '\t'.join(
                    [str(i) for i in performances[alpha]])  # str(sum() / len(performances[alpha]))
                writer.write(line + '\n')
                # print(line)

        writer.close()
