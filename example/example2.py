#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

def clean(sentence_ws, sentence_pos):
  short_with_pos = []
  short_sentence = []
  stop_pos = set(['Nep', 'Nh', 'Nb']) # 這 3 種詞性不保留
  for word_ws, word_pos in zip(sentence_ws, sentence_pos):
    # 只留名詞和動詞
    is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N")
    # 去掉名詞裡的某些詞性
    is_not_stop_pos = word_pos not in stop_pos
    # 只剩一個字的詞也不留
    is_not_one_charactor = not (len(word_ws) == 1)
    # 組成串列
    if is_N_or_V and is_not_stop_pos and is_not_one_charactor:
      short_with_pos.append(f"{word_ws}({word_pos})")
      short_sentence.append(f"{word_ws}")
  return (short_sentence, short_with_pos)

def main():
    # Show version
    print(__version__)

    # Initialize drivers
    print("Initializing drivers ... WS")
    ws_driver = CkipWordSegmenter(model="bert-base")
    print("Initializing drivers ... POS")
    pos_driver = CkipPosTagger(model="bert-base")
    print("Initializing drivers ... NER")
    ner_driver = CkipNerChunker(model="bert-base")
    print("Initializing drivers ... done")
    print()

    # Input text
    lines = open('sample2', 'r', encoding='utf8')
    text = [line.replace('\n', '') for line in lines]
    
    # text = [
    #     "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
    #     "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
    #     "空白 也是可以的～",
    # ]

    # Run pipeline
    print("Running pipeline ... WS")
    ws = ws_driver(text)
    print("Running pipeline ... POS")
    pos = pos_driver(ws)
    print("Running pipeline ... NER")
    ner = ner_driver(text)
    print("Running pipeline ... done")
    print()

    # Show results
    print('=====')
    for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
        print("原文：")
        print(sentence)
        (short, res) = clean(sentence_ws, sentence_pos)
        print(f'short {short}')
        print(f'res {res}')
        print("斷詞後：")
        print(short)
        print("斷詞後+詞性標注：")
        print(res)
        print('=====')


if __name__ == "__main__":
    main()
