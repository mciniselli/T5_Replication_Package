import codecs
from typing import List

import re

import os
import csv

import sys
import argparse

import random


class FileManager:
    def __init__(self, file_path: str):
        '''
        this class is a filemanager. It allows you to read and write txt or csv file
        e.g.
        condition_ok = FileManager("condition_ok.txt")
        condition_ok.open_file_txt("a+")
        condition_ok.write_file_txt("hello!")
        condition_ok.close_file()
        data=condition_ok.read_file_txt()
        '''
        self.file = None
        self.writer = None
        self.fieldnames = None
        self.file_path = file_path

    def open_file_csv(self, mode: str, fieldnames: List[str], force_write_header: bool = False):
        write_header = True
        if os.path.exists(self.file_path):
            write_header = False
        self.file = open(self.file_path, mode=mode, encoding="utf-8")
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if write_header or force_write_header:
            self.writer.writeheader()

    def write_file_csv(self, element_list: List[str]):
        self.writer.writerow(element_list)

    def close_file(self):
        self.file.close()

    def open_file_txt(self, mode: str):
        self.file = codecs.open(self.file_path, mode=mode, encoding="utf-8")

    def open_file_txt_no_codecs(self, mode: str):
        self.file = open(self.file_path, mode=mode, encoding="utf-8")

    def read_file_txt(self):
        try:
            self.open_file_txt("r")

            content = self.file.readlines()
            # print("LEN: {}".format(len(content)))
            c_ = list()
            for c in content:
                r = c.rstrip("\n").rstrip("\r")
                c_.append(r)

        except Exception as e:
            print("Error ReadFile: " + str(e))
            c_ = []
        finally:
            self.close_file()
        return c_

    def read_file_txt_no_codecs(self):
        try:
            self.open_file_txt_no_codecs("r")

            content = self.file.readlines()
            c_ = list()
            for c in content:
                r = c.rstrip("\n").rstrip("\r")
                c_.append(r)

        except Exception as e:
            print("Error ReadFile: " + str(e))
            c_ = []
        finally:
            self.close_file()
        return c_

    def read_csv(self):
        dict_result = dict()
        try:
            with open(self.file_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:

                        for r in row:
                            dict_result[r] = list()
                            dict_result[r].append(row[r])

                        line_count += 1
                    else:
                        for r in row:
                            dict_result[r].append(row[r])
                        line_count += 1
        except Exception as e:
            dict_result = dict()
        return dict_result

    def read_csv_list(self, separator="|--|"):
        list_result = list()
        try:
            with open(self.file_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:

                    if line_count == 0:

                        row_curr = list()

                        keys = row.keys()

                        for k in keys:
                            row_curr.append(row[k])

                        list_result.append(separator.join(row_curr))

                        line_count += 1
                    else:
                        row_curr = list()

                        keys = row.keys()

                        for k in keys:
                            row_curr.append(row[k])

                        list_result.append(separator.join(row_curr))
                        line_count += 1
                        # print(separator.join(row_curr))
        except Exception as e:
            dict_result = dict()
        return list_result

    def write_file_txt(self, element):  # write generic file

        self.file.write(element + "\n")



def check_predictions(predictions_file, targets_file):


    f = FileManager(predictions_file)
    # f.open_file_txt_no_codecs("r")
    predictions = f.read_file_txt()
    f.close_file()

    f = FileManager(targets_file)
    # f.open_file_txt_no_codecs("r")
    targets = f.read_file_txt()
    f.close_file()

    if len(predictions) != len(targets):
        print("ERROR: lengths do not match")

    number_equal=0

    for x,y in zip(predictions, targets):
        if x.replace(" ", "")== y.replace(" ", ""):
            number_equal+=1


    print("{} predictions equal out of {} ({} %)".format(number_equal, len(predictions), round(100*number_equal/len(predictions),2)))


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_path",
        default=None,
        type=str,
        help="path to the target file ",
    )

    parser.add_argument(
        "--prediction_path",
        default=None,
        type=str,
        help="path to the prediction file",
    )


    args = parser.parse_args()

    check_predictions(args.prediction_path, args.target_path)