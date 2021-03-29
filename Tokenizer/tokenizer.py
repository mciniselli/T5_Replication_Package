import sentencepiece as spm
import argparse


def main():
    print("ciao")

    parser = argparse.ArgumentParser()

    parser.add_argument("-input", "--input", type=str, default="data/train.txt",
                        help="tokenizer input file")

    parser.add_argument("-model_prefix", "--model_prefix", type=str, default="m",
                        help="prefix for the model")

    parser.add_argument("-vocab_size", "--vocab_size", type=int, default=32000,
                        help="the size of the vocabulary")

    parser.add_argument("-character_coverage", "--character_coverage", type=float, default=0.995,
                        help="amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanse or Chinese and 1.0 for other languages with small character set")

    parser.add_argument("-bos_id", "--bos_id", type=int, default=-1,
                        help="begin of sentence id")
    parser.add_argument("-eos_id", "--eos_id", type=int, default=1,
                        help="end of sentence id")
    parser.add_argument("-unk_id", "--unk_id", type=int, default=2,
                        help="unknown id")
    parser.add_argument("-pad_id", "--pad_id", type=int, default=0,
                        help="padding id")

    args = parser.parse_args()



    # spm.SentencePieceTrainer.train('--input=train_pretraining_clean.txt --model_prefix=dl4se --vocab_size=32000 --bos_id=-1  --eos_id=1 --unk_id=2 --pad_id=0')
    spm.SentencePieceTrainer.train(input=args.input, model_prefix=args.model_prefix, vocab_size=args.vocab_size, character_coverage=args.character_coverage,
                                   bos_id=args.bos_id, eos_id=args.eos_id, unk_id=args.unk_id, pad_id=args.pad_id)



if __name__=="__main__":
    main()
