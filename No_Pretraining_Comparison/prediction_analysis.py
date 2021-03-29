import argparse
import os
from filemanager import FileManager

import math

import numpy as np

def is_ok(parts):
    if str(parts[4]) not in ["TRUE", "FALSE"]:
        return False
    if str(parts[5]) not in ["TRUE", "FALSE"]:
        return False
    if str(parts[6]) not in ["TRUE", "FALSE"]:
        return False

    return True

def get_key(parts):
    return parts[7][2:]



def compare_models(csv_file, output_folder):
    
    f=FileManager(csv_file)
    fields=["key", "method_id", "masked_method_id", "prediction_size is_perfect_t5", "is_perfect_RoBERTa",
      "is_perfect_ngram", "Dataset"]

    separator="!--*__!"

    f.open_file_csv("r", fields)
    data=f.read_csv_list(separator)

    dict_perfect_t5=dict()
    dict_perfect_roberta=dict()
    dict_perfect_ngram=dict()

    num_ok=0

    for row in data:
        parts=row.split(separator)
        key=get_key(parts)
        if key not in dict_perfect_t5:
            dict_perfect_t5[key]=list()
        if key not in dict_perfect_roberta:
            dict_perfect_roberta[key]=list()
        if key not in dict_perfect_ngram:
            dict_perfect_ngram[key]=list()

        if not is_ok(parts): # the record has no result for ngram (and hence roberta)
            continue
        else:
            num_ok+=1

        if str(parts[4])=="TRUE":
            dict_perfect_t5[key].append(1)
        else:
            dict_perfect_t5[key].append(0)

        if str(parts[5])=="TRUE":
            dict_perfect_roberta[key].append(1)
        else:
            dict_perfect_roberta[key].append(0)

        if str(parts[6])=="TRUE":
            dict_perfect_ngram[key].append(1)
        else:
            dict_perfect_ngram[key].append(0)

    print("{} record are OK out of {}".format(num_ok, len(data)))

    print("prediction for T5")

    num_pp_global=0
    num_totalp_global=0

    for k in dict_perfect_t5.keys():
        num_pp=np.sum(dict_perfect_t5[k])
        num_totalp=len(dict_perfect_t5[k])
        num_pp_global+=num_pp
        num_totalp_global+=num_totalp
        print("{}: {} perfect prediction out of {} ({}%)".format(k, num_pp, num_totalp, round(100*num_pp/num_totalp,2) ))

    print("Overall: {} perfect prediction out of {} ({}%)".format(num_pp_global, num_totalp_global, round(100*num_pp_global/num_totalp_global,2) ))


    print("prediction for RoBERTa")

    num_pp_global=0
    num_totalp_global=0

    for k in dict_perfect_roberta.keys():
        num_pp=np.sum(dict_perfect_roberta[k])
        num_totalp=len(dict_perfect_roberta[k])
        num_pp_global+=num_pp
        num_totalp_global+=num_totalp
        print("{}: {} perfect prediction out of {} ({}%)".format(k, num_pp, num_totalp, round(100*num_pp/num_totalp,2) ))

    print("Overall: {} perfect prediction out of {} ({}%)".format(num_pp_global, num_totalp_global, round(100*num_pp_global/num_totalp_global,2) ))

    print("prediction for ngram")

    num_pp_global=0
    num_totalp_global=0

    for k in dict_perfect_ngram.keys():
        num_pp=np.sum(dict_perfect_ngram[k])
        num_totalp=len(dict_perfect_ngram[k])
        num_pp_global+=num_pp
        num_totalp_global+=num_totalp
        print("{}: {} perfect prediction out of {} ({}%)".format(k, num_pp, num_totalp, round(100*num_pp/num_totalp,2) ))

    print("Overall: {} perfect prediction out of {} ({}%)".format(num_pp_global, num_totalp_global, round(100*num_pp_global/num_totalp_global,2) ))



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_csv",
        default=None,
        type=str,
        help="path for the result_csv file with TRUE or FALSE for T5 single task no pretraining, RoBERTa and n-gram",
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="result folder you want to save results in",
    )


    args = parser.parse_args()

    compare_models(args.result_csv, args.output_folder)




if __name__ == "__main__":
    main()
