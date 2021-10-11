import pandas as pd
from amr_utils.amr_readers import AMR_Reader


def read_file(file_path):
    with open(file_path) as csvfile:
        data = pd.read_csv(csvfile, delimiter=',', header=0)
        data = data[data['is_duplicate'] == 1]
        question1 = data['question1']
        question2 = data['question2']
        assert len(question1) == len(question2)
        print(len(question1))
        question1.to_csv('question1.txt', index=False, header=False)
        question2.to_csv('question2.txt', index=False, header=False)


def sort_AMR(file_path):
    reader = AMR_Reader()
    amrs = reader.load(file_path, remove_wiki=True)
    amrs.sort(key=lambda x: int(x.metadata['nsent']))
    # for amr in amrs:
    #     print(amr.metadata['nsent'])
    print('amr length:', len(amrs))
    reader.write_to_file(file_path.replace('.amr', '_sorted.amr'), amrs)


if __name__ == '__main__':
    # csv_file = '/home/data/zshou/corpus/quora_question_pairs/train.csv'
    # read_file(csv_file)
    amr_file = 'question2.amr'
    sort_AMR(amr_file)
