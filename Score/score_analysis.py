import codecs
from typing import List
from typing import *

import re

import os

import sys
import argparse

import random
import numpy as np

import math

from statistics import *

from metrics import *
from tokenizer import tokenize

from filemanager import FileManager


def get_class(value: float):
    '''
    return the class (from 1 to 10)
    0<=value<0.1 => 1
    0.1<=value<0.2 => 2
    ...
    '''
    temp = math.ceil(value * 10 + 0.01 * (value * 10 % 1 == 0 and value * 10 != 10))

    return temp


def get_key(record):
    first_part = record.split(":")[0].lower()
    key = "android "
    if "java" in first_part:
        key = "java "

    if "token" in first_part:
        return key + "token"
    elif "construct" in first_part:
        return key + "construct"
    else:
        return key + "block"


def check_score(input_path, score_path):
    '''
    the function checks the number of perfect prediction for each class of confidence
    (given by the T5 score).
    It checks the Levenshtein distance for each class to see if when the confidence is high
    the similarity between the real target and prediction is high (not verified)
    '''

    f = FileManager(os.path.join(input_path, "inputs.txt"))
    # f.open_file_txt_no_codecs("r")
    inputs = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "predictions.txt"))
    # f.open_file_txt_no_codecs("r")
    predictions = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "targets.txt"))
    # f.open_file_txt_no_codecs("r")
    targets = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(score_path, "scores.txt"))
    # f.open_file_txt_no_codecs("r")
    scores = f.read_file_txt()
    f.close_file()

    if not (len(predictions) == len(targets) == len(scores) == len(inputs)):
        print("ERROR: lengths do not match")

    ## GLOBAL NUMBER OF PERFECT PREDICTIONS FOR EACH CONFIDENCE CLASS

    dict_perfect = dict()
    dict_non_perfect = dict()

    for i in range(1, 11):
        dict_perfect[i] = 0
        dict_non_perfect[i] = 0

    scores = [float(s) for s in scores]

    for x, y, z in zip(predictions, targets, scores):
        is_perfect = is_perfect_prediction(x, y)

        likelihood = math.exp(z)
        class_likelihood = get_class(likelihood)

        if is_perfect:
            dict_perfect[class_likelihood] += 1
        else:
            dict_non_perfect[class_likelihood] += 1

    for i in range(1, 11):
        num_p = dict_perfect[i]
        num_non_p = dict_non_perfect[i]

        num_tot = num_p + num_non_p

        print("class {}: {} perfect predictions out of {} ({}%)".format(i, num_p, num_tot,
                                                                        round((100 * num_p / num_tot), 2)))

    ## GLOBAL NUMBER OF PERFECT PREDICTIONS FOR EACH CONFIDENCE CLASS FOR EACH DATASET

    dict_position = dict()
    dict_position["android block"] = 0
    dict_position["android construct"] = 1
    dict_position["android token"] = 2
    dict_position["java block"] = 3
    dict_position["java construct"] = 4
    dict_position["java token"] = 5

    dict_perfect_dataset = dict()
    dict_non_perfect_dataset = dict()

    for i in range(1, 11):
        dict_perfect_dataset[i] = [0] * 6
        dict_non_perfect_dataset[i] = [0] * 6

    for x, y, z, w in zip(predictions, targets, scores, inputs):

        is_perfect = is_perfect_prediction(x, y)

        likelihood = math.exp(z)
        class_likelihood = get_class(likelihood)

        key = get_key(w)
        position = dict_position[key]

        if is_perfect:
            dict_perfect_dataset[class_likelihood][position] += 1
        else:
            dict_non_perfect_dataset[class_likelihood][position] += 1

    dict_position_reverse = dict()
    for k in dict_position.keys():
        value = dict_position[k]
        dict_position_reverse[value] = k

    for i in range(6):
        print(dict_position_reverse[i])
        for j in range(1, 11):
            num_p = dict_perfect_dataset[j][i]
            num_non_p = dict_non_perfect_dataset[j][i]

            num_tot = num_p + num_non_p

            print("class {}: {} perfect predictions out of {} ({}%)".format(j, num_p, num_tot,
                                                                            round((100 * num_p / num_tot), 2)))

    ## LEVENSHTEIN

    lev = dict()

    for i in range(1, 11):
        lev[i] = list()

    for i, (x, y, z) in enumerate(zip(predictions, targets, scores)):
        if i % 20000 == 0:
            print("{} out of {}".format(i, len(scores)))

        is_perfect = is_perfect_prediction(x, y)

        if is_perfect:
            continue

        likelihood = math.exp(z)
        class_likelihood = get_class(likelihood)

        pred_tokens = tokenize(x)
        target_tokens = tokenize(y)

        distance = get_levenshtein_perc(target_tokens, pred_tokens, False)

        lev[class_likelihood].append(distance)

    for i in range(1, 11):
        mean_lev = mean(lev[i])
        median_lev = median(lev[i])

        print("class {}: mean {}, median {})".format(i, round(mean_lev, 2), round(median_lev, 2)))


def check_lengths(input_path, score_path):
    '''
    the hypotesis is that when the confidence of the prediction is high (>0.9)
    the length of the prediction is slow (meaning that it is confident and the 
    prediction is correct because the tokens to predict are a small number )
    '''
    f = FileManager(os.path.join(input_path, "inputs.txt"))
    # f.open_file_txt_no_codecs("r")
    inputs = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "predictions.txt"))
    # f.open_file_txt_no_codecs("r")
    predictions = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "targets.txt"))
    # f.open_file_txt_no_codecs("r")
    targets = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(score_path, "scores.txt"))
    # f.open_file_txt_no_codecs("r")
    scores = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "lengths.txt"))
    # f.open_file_txt_no_codecs("r")
    lengths = f.read_file_txt()
    f.close_file()

    if not (len(predictions) == len(targets) == len(scores) == len(inputs) == len(lengths)):
        print("ERROR: lengths do not match")

    ## LENGTH PERFECT PREDICTIONS AND NOT PERFECT PREDICTIONS FOR EACH CONFIDENCE CLASS

    len_perfect = dict()
    len_non_perfect_target = dict()
    len_non_perfect_prediction = dict()

    for i in range(1, 11):
        len_perfect[i] = list()
        len_non_perfect_target[i] = list()
        len_non_perfect_prediction[i] = list()

    scores = [float(s) for s in scores]

    lengths = [int(s) for s in lengths]

    for x, y, z, w in zip(predictions, targets, scores, lengths):

        is_perfect = is_perfect_prediction(x, y)

        likelihood = math.exp(z)
        class_likelihood = get_class(likelihood)

        if is_perfect:
            len_perfect[class_likelihood].append(w)
        else:
            len_non_perfect_target[class_likelihood].append(w)

            tokens = tokenize(x)
            len_non_perfect_prediction[class_likelihood].append(len(tokens))

    for i in range(1, 11):
        mean_per = mean(len_perfect[i])
        mean_non_p_target = mean(len_non_perfect_target[i])
        mean_non_p_prediction = mean(len_non_perfect_prediction[i])

        median_per = median(len_perfect[i])
        median_non_p_target = median(len_non_perfect_target[i])
        median_non_p_prediction = median(len_non_perfect_prediction[i])

        len_total = len_perfect[i]
        len_total.extend(len_non_perfect_target[i])

        mean_global = mean(len_total)
        median_global = median(len_total)

        print("class {}: mean perfect {}, median perfect {}".format(i, round(mean_per, 2), round(median_per, 2)))
        print(
            "class {}: mean non perfect target {}, median non perfect target {}".format(i, round(mean_non_p_target, 2),
                                                                                        round(median_non_p_target, 2)))
        print("class {}: mean non perfect prediction {}, median non perfect prediction {}".format(i, round(
            mean_non_p_prediction, 2), round(median_non_p_prediction, 2)))
        print("class {}: mean global {}, median global {}".format(i, round(mean_global, 2), round(median_global, 2)))


def is_perfect_prediction(pred, targ):
    if pred.replace(" ", "") == targ.replace(" ", ""):
        return True
    return False


def return_position(current, min_array, max_array):
    '''
    given a current value and a list of minimum and maximum,
    it returns in which interval the current is inside
    current=100, min_array=[0, 200], max_array=[199, 400] => return 0
    current=300, min_array=[0, 200], max_array=[199, 400] => return 1
    '''
    for i, (x, y) in enumerate(zip(min_array, max_array)):
        if current >= x and current <= y:
            return i

    return -1


def check_perfect(input_path, score_path):
    '''
    HYPOTHESIS: the data are not shuffled (all the data from the same dataset are contiguous)
    return the perfect predictions for each dataset and for each number of tokens
    '''

    f = FileManager(os.path.join(input_path, "inputs.txt"))
    inputs = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "predictions.txt"))
    predictions = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "targets.txt"))
    targets = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(score_path, "scores.txt"))
    scores = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "lengths.txt"))
    lengths = f.read_file_txt()
    f.close_file()

    if not (len(predictions) == len(targets) == len(scores) == len(inputs) == len(lengths)):
        print("ERROR: lengths do not match")

    lengths = [int(l) for l in lengths]

    dict_position = dict()
    dict_position["android block"] = 0
    dict_position["android construct"] = 1
    dict_position["android token"] = 2
    dict_position["java block"] = 3
    dict_position["java construct"] = 4
    dict_position["java token"] = 5

    min_len = [99999999] * 6
    max_len = [0] * 6

    min_index = [99999999] * 6
    max_index = [0] * 6

    # return start and end index for each class
    # e.g. class 0 starts from index 0 and ends with index 99
    #      class 1 starts from index 100 and ends with index 400

    for i, (x, y) in enumerate(zip(inputs, lengths)):
        key = get_key(x)
        position = dict_position[key]
        if min_index[position] > i:
            min_index[position] = i

        if max_index[position] < i:
            max_index[position] = i

    # compute the min and max lengths for each dataset

    for i, x in enumerate(lengths):
        position = return_position(i, min_index, max_index)
        if min_len[position] > x:
            min_len[position] = x

        if max_len[position] < x:
            max_len[position] = x

    for i in range(6):
        index_from = min_index[i]
        index_to = max_index[i]

        perfect_length = [0] * max_len[i]
        total_length = [0] * max_len[i]

        print(list(dict_position.keys())[i])

        for j, (x, y, z) in enumerate(zip(predictions, targets, lengths)):

            if j < index_from or j > index_to:
                continue

            is_perfect = is_perfect_prediction(x, y)

            if is_perfect:
                perfect_length[z - 1] += 1

            total_length[z - 1] += 1

            # if is_perfect and z>30:
            #     print(x)
            #     print(y)
            #     print("________")

        num_ok = 0  # used for grouping
        num_total = 0  # used for grouping

        num_ok_global = 0
        num_total_global = 0

        for j, (x, y) in enumerate(zip(perfect_length, total_length)):
            print("length {}: {} perfect predictions out of {} ({}%)".format(j + 1, x, y, round((100 * x / y), 2)))
            num_ok_global += x
            num_total_global += y

        print("GLOBAL: {} perfect predictions out of {} ({}%)".format(num_ok_global, num_total_global,
                                                                      round((100 * num_ok_global / num_total_global),
                                                                            2)))

        if i % 3 == 0:
            print("GROUPING BLOCK LEVEL EVERY 5")

            for j, (x, y) in enumerate(zip(perfect_length, total_length)):
                if j % 5 == 0 and j != 0:
                    print("length {}-{}: {} perfect predictions out of {} ({}%)".format(j - 4, j, num_ok, num_total,
                                                                                        round(
                                                                                            (100 * num_ok / num_total),
                                                                                            2)))
                    num_ok = 0
                    num_total = 0
                num_ok += x
                num_total += y

            print("length {}-{}: {} perfect predictions out of {} ({}%)".format(j - (j % 5), j, num_ok, num_total,
                                                                                round((100 * num_ok / num_total), 2)))

            print("END GROUPING")


def check_tokenizer(input_path):
    '''
    This test function is useful to check if the tokenizer
    works quite well.
    In the test done we predicted 115 wrong length out of 27k
    The files for the test are in file_test_tokenizer folder
    '''
    f = FileManager(os.path.join(input_path, "mask.txt"))
    # f.open_file_txt_no_codecs("r")
    mask = f.read_file_txt()
    f.close_file()
    print(len(mask))

    f = FileManager(os.path.join(input_path, "raw_data.csv"))
    # f.open_file_txt_no_codecs("r")
    raw_data = f.read_file_txt()
    f.close_file()

    raw_data = raw_data[1:]

    lengths = list()
    for r in raw_data:
        parts = r.split(",")
        lengths.append(parts[3])

    num_ok = 0
    num_ko = 0
    for i, m in enumerate(mask):
        res = tokenize(m)
        if str(len(res)) == lengths[i]:
            num_ok += 1
        else:
            num_ko += 1

    print(num_ok)
    print(num_ko)


def check_metrics(input_path):
    f = FileManager(os.path.join(input_path, "inputs.txt"))
    # f.open_file_txt_no_codecs("r")
    inputs = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "predictions.txt"))
    # f.open_file_txt_no_codecs("r")
    predictions = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "targets.txt"))
    # f.open_file_txt_no_codecs("r")
    targets = f.read_file_txt()
    f.close_file()

    f = FileManager(os.path.join(input_path, "lengths.txt"))
    # f.open_file_txt_no_codecs("r")
    lengths = f.read_file_txt()
    f.close_file()

    if not (len(predictions) == len(targets) == len(inputs) == len(lengths)):
        print("ERROR: lengths do not match")

    lengths = [int(l) for l in lengths]

    dict_position = dict()
    dict_position["android block"] = 0
    dict_position["android construct"] = 1
    dict_position["android token"] = 2
    dict_position["java block"] = 3
    dict_position["java construct"] = 4
    dict_position["java token"] = 5

    min_index = [99999999] * 6
    max_index = [0] * 6

    # return start and end index for each class
    # e.g. class 0 starts from index 0 and ends with index 99
    #      class 1 starts from index 100 and ends with index 400

    for i, (x, y) in enumerate(zip(inputs, lengths)):
        key = get_key(x)
        position = dict_position[key]
        if min_index[position] > i:
            min_index[position] = i

        if max_index[position] < i:
            max_index[position] = i

    for i in range(6):
        index_from = min_index[i]
        index_to = max_index[i]


        bleu1 = list()
        bleu2 = list()
        bleu3 = list()
        bleu4 = list()
        lev = list()

        for j, (x, y, z) in enumerate(zip(predictions, targets, lengths)):

            if j < index_from or j > index_to:
                continue

            is_perfect = is_perfect_prediction(x, y)
            predicted_tokens = tokenize(x)
            target_tokens = tokenize(y)

            b1, b2, b3, b4, lv = evaluate_metrics(target_tokens, predicted_tokens, z, is_perfect)

            bleu1.append(b1)
            if b2 is not None:
                bleu2.append(b2)
            if b3 is not None:
                bleu3.append(b3)
            if b4 is not None:
                bleu4.append(b4)
            lev.append(lv)

        print(list(dict_position.keys())[i])

        print("BLEU1: SUM {}, LEN {}, AVG {}".format(sum(bleu1), len(bleu1), mean(bleu1)))
        print("BLEU2: SUM {}, LEN {}, AVG {}".format(sum(bleu2), len(bleu2), mean(bleu2)))
        print("BLEU3: SUM {}, LEN {}, AVG {}".format(sum(bleu3), len(bleu3), mean(bleu3)))
        print("BLEU4: SUM {}, LEN {}, AVG {}".format(sum(bleu4), len(bleu4), mean(bleu4)))
        print("LEV DISTANE: SUM {}, LEN {}, AVG {}".format(sum(lev), len(lev), mean(lev)))

if __name__ == "__main__":

    # target_tokens=[['{', 'seen', '.', 'add', '(', 'serverId', ')', ';', '}']]
    # predicted_tokens=['{', 'seen', '.', 'add', '(', 'serverId', ')', ';', 'saveSeenInvitations', '(', 'seen', ')', ';', '}']
    #
    # b2=nltk.translate.bleu_score.sentence_bleu(target_tokens,
    #                                         predicted_tokens, weights=(0.5, 0.5, 0.0))
    # print(b2)
    #
    # sys.exit(0)


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        help="path for the folder that contains all the files required as input (targets.txt, predictions.txt)",
    )

    parser.add_argument(
        "--score_path",
        default=None,
        type=str,
        help="path for the folder that contains scores.txt file",
    )

    parser.add_argument("--score", help="check the score saved in scores.txt file",
                        action="store_true")

    parser.add_argument("--lengths", help="compute the lengths for each class of confidence",
                        action="store_true")

    parser.add_argument("--perfect", help="check perfect predictions based on token lengths",
                        action="store_true")

    parser.add_argument("--metrics", help="compute BLEU and Levenshtein distance for each dataset",
                        action="store_true")

    args = parser.parse_args()

    if args.score:
        check_score(args.input_path, args.score_path)

    if args.lengths:
        check_lengths(args.input_path, args.score_path)

    if args.perfect:
        check_perfect(args.input_path, args.score_path)

    if args.metrics:
        check_metrics(args.input_path)
