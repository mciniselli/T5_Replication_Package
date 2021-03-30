import numpy as np
import nltk


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def get_levenshtein_perc(target_tokens, predicted_tokens, is_perfect_prediction):
    # Word-level Levenshtein Distance
    if is_perfect_prediction:
        levenshtein_distance = 0.0
        return levenshtein_distance
    else:
        levenshtein_distance = levenshtein(predicted_tokens, target_tokens)

    prediction_length = len(predicted_tokens)
    target_length = len(target_tokens)
    lev_distance_perc = 0
    if not is_perfect_prediction:
        if prediction_length > target_length:
            lev_distance_perc = levenshtein_distance / prediction_length
        else:
            if target_length == 0:
                lev_distance_perc = -1
                print("target_length = 0")
            else:
                lev_distance_perc = levenshtein_distance / target_length

    return lev_distance_perc


def evaluate_metrics(target_tokens, predicted_tokens, len_target, is_perfect_prediction):
    '''
    given the tokens of the target mask,
    the tokens of the predicted mask and
    the number of tokens (it is equal to the length of real_tokens more than 99.9%)
    it returns BLEU1, BLEU2, BLEU3, BLEU4 and levenshtein distance

    '''

    bleu1_score = None
    bleu2_score = None
    bleu3_score = None
    bleu4_score = None

    if is_perfect_prediction:
        bleu1_score = 1.0
    else:
        bleu1_score = nltk.translate.bleu_score.sentence_bleu([target_tokens],
                                                              predicted_tokens, weights=(1.0, 0.0))

    if len_target > 1:
        if is_perfect_prediction:
            bleu2_score = 1.0
        else:
            bleu2_score = nltk.translate.bleu_score.sentence_bleu([target_tokens],
                                                                  predicted_tokens, weights=(0.5, 0.5, 0.0))

    if len_target > 2:
        if is_perfect_prediction:
            bleu3_score = 1.0
        else:
            bleu3_score = nltk.translate.bleu_score.sentence_bleu([target_tokens],
                                                                  predicted_tokens,
                                                                  weights=(0.333, 0.333, 0.333))
    if len_target > 3:
        if is_perfect_prediction:
            bleu4_score = 1.0
        else:
            bleu4_score = nltk.translate.bleu_score.sentence_bleu([target_tokens],
                                                                  predicted_tokens)
    levenshtein_distance = get_levenshtein_perc(target_tokens, predicted_tokens, is_perfect_prediction)

    return bleu1_score, bleu2_score, bleu3_score, bleu4_score, levenshtein_distance
