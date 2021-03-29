import argparse
import os
from filemanager import FileManager

import math

def is_perfect_prediction(pred, targ):
    if pred.replace(" ", "") == targ.replace(" ", ""):
        return True
    return False



def check_best_score(input_path, score_path, output_path, max_number):
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

    lengths = [int(l) for l in lengths]

    scores = [float(s) for s in scores]

    sorted_scores = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    num_processed = 0

    output_file=FileManager(os.path.join(output_path, "result_best_score.txt"))
    output_file.open_file_txt("w+")

    for i in range(len(sorted_scores)):
        input_curr = inputs[sorted_scores[i]]
        prediction_curr = predictions[sorted_scores[i]]
        target_curr = targets[sorted_scores[i]]
        score_curr = math.exp(scores[sorted_scores[i]])

        if is_perfect_prediction(prediction_curr, target_curr):
            continue


        input_split=input_curr.split("<extra_id_0>")
        output_file.write_file_txt("{}".format(num_processed+1))
        output_file.write_file_txt(input_split[0])
        output_file.write_file_txt("PRED: {}".format(prediction_curr))
        output_file.write_file_txt("TARG: {}".format(target_curr))
        output_file.write_file_txt(input_split[1])
        output_file.write_file_txt("_________________________________")

        # print(input_curr)
        # print(prediction_curr)
        # print(target_curr)
        # print(score_curr)
        # print("________")
        num_processed += 1

        if num_processed >= max_number:
            break

    output_file.close_file()


def main():
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

    parser.add_argument(
        "--output_path",
        default=".",
        type=str,
        help="path for the folder where you want to write the result",
    )

    parser.add_argument(
        "--max_number",
        default=200,
        type=int,
        help="number of predictions with highest scores you want to export",
    )



    parser.add_argument("--check_best",
                        help="check the wrong predictions with highest scores to see if they have the same behaviour",
                        action="store_true")

    args = parser.parse_args()

    if args.check_best:
        check_best_score(args.input_path, args.score_path, args.output_path, args.max_number)


if __name__ == "__main__":
    main()
