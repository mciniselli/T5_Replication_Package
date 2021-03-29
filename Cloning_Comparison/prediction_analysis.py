import argparse
import os
from filemanager import FileManager

import math

import numpy as np


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


def comparison(input_folder, result_csv, output_folder):
    f=FileManager(result_csv)
    fields=["key", "method_id", "masked_method_id", "prediction_size is_perfect_t5",
    "Dataset"]

    separator="!--*__!"

    f.open_file_csv("r", fields)
    data=f.read_csv_list(separator)

    dict_t5=dict()

    print("Read {} records".format(len(data)))

    for row in data:
        parts=row.split(separator)
        key=parts[1]+"_"+parts[2]
        dict_t5[key]=parts[4]

    files=os.listdir(input_folder)
    files=[f for f in files if ".txt" in f]

    print(files)

    overall_perfect_predictions=0
    overall_records=0

    for ff in files:

        f=FileManager(os.path.join(input_folder,ff))

        data_cloning=f.read_file_txt()
        num_records=len(data_cloning)
        print("Read {} records".format(num_records))

        num_perfect_prediction=0

        for d in data_cloning:
            parts=d.split("|_|")
            method_id=parts[0]
            masked_method_id=parts[1]
            key="{}_{}".format(method_id, masked_method_id)
            if dict_t5[key]=="TRUE":
                num_perfect_prediction+=1

        overall_perfect_predictions+=num_perfect_prediction
        overall_records+=num_records

        print("{}: {} perfect prediction out of {} ({}%)".format(ff.replace("_raw.txt", "").replace("_", " "), num_perfect_prediction, num_records, round(100*num_perfect_prediction/num_records,2) ))

    print("Overall: {} perfect prediction out of {} ({}%)".format(overall_perfect_predictions, overall_records, round(100*overall_perfect_predictions/overall_records,2) ))





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder",
        default=None,
        type=str,
        help="path that contains the results on the 200 selectd repos for cloning",
    )

    parser.add_argument(
        "--result_csv",
        default=None,
        type=str,
        help="path to the results.csv path (used in step 8) that contain information about the T5 predicitons",
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="result folder you want to save results in",
    )


    args = parser.parse_args()

    comparison(args.input_folder, args.result_csv, args.output_folder)





if __name__ == "__main__":
    main()
