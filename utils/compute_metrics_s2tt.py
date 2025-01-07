
import argparse
import csv
import json
import logging
import os, sys
from re import sub

import jieba
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# speech2text en-zh


def _text_preprocess(sentence):
    sentence = sentence.strip()
    # sentence = sentence.lower()
    # sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    # sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence



def compute_bleu(predicted, original):
    audioIds = sorted(predicted.keys())

    predicted = {key: predicted[key] for key in audioIds}
    original = {key: original[key] for key in audioIds}

    metrics = {}
    data = []
    bleu_list = []

    for key in audioIds:
        label = original[key][0]
        prediction = predicted[key][0]    
        # 使用 jieba 分词，将句子转换成词列表
        label_tokens = list(jieba.cut(label))
        prediction_tokens = list(jieba.cut(prediction))

        # 将参考翻译转为嵌套列表的形式，因为 BLEU 支持多参考句
        reference = [label_tokens]
        candidate = prediction_tokens

        # 使用平滑函数，防止出现低分或零分的情况
        smooth_fn = SmoothingFunction().method4

        # 计算 BLEU 分数
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
        bleu_list.append(bleu_score)
        
        meta = {
            "name": key,
            "bleu": bleu_score,
            "predicted": prediction,
            "Original": label
        }
        data.append(meta)

    metrics["bleu"] = sum(bleu_list) / len(bleu_list)
    metrics["data"] = data

    return metrics



def compute_metrics(predict_file, ref_file):
    ref_dict = {}
    with open(ref_file, "r", encoding="utf8") as reader:
        for line in reader:
            obj = json.loads(line)
            name = obj["key"]
            caption = obj["label"].split("\t")
            # caption_rex = [_text_preprocess(cap) for cap in caption]
            caption_rex = caption
            # if name in ref_dict:
            #     ref_dict[name].append(caption_rex)
            # else:
            #     ref_dict[name] = [caption_rex]
            ref_dict[name] = caption_rex

    predict_dict = {}
    with open(predict_file, "r", encoding="utf8") as reader:
        for line in reader:
            temp = line.strip("\n").split("\t")
            predict_dict[temp[0]] = [temp[1]]

    res_dir = os.path.dirname(predict_file)
    res_prefix = os.path.basename(predict_file).replace(".txt", "")
    metrics = compute_bleu(predict_dict, ref_dict)
    logging.info(
        "bleu {}".format(
            round(metrics["bleu"], 5),
        )
    )

    eval_file = os.path.join(
        res_dir,
        "{}_bleu{}.csv".format(
            res_prefix,
            round(metrics["bleu"], 5),
        ),
    )
    with open(eval_file, "w+", encoding="utf8", newline="") as csvfile:
        csv_writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "name",
                "bleu",
                "predicted",
                "Original",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(metrics["data"])
    logging.info(
        "End eval translation for {}, {}, ref {}, eval {}".format(
            predict_file, res_prefix, ref_file, eval_file
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test with your special model")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--predict_dir", required=True, help="predict result file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(args)

    predict_files = [
        "beam_1",
        "beam_2",
        "beam_3",
        "top-p_0.85",
        "top-p_0.9",
        "top-p_0.95",
    ]
    for pf in predict_files:
        p_file = os.path.join(args.predict_dir, pf + ".txt")
        if os.path.isfile(p_file):
            compute_metrics(p_file, args.test_data)
        p_llm_file = os.path.join(args.predict_dir, pf + "_llm.txt")
        if os.path.isfile(p_llm_file):
            compute_metrics(p_llm_file, args.test_data)
