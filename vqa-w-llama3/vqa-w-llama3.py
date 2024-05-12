# -*- coding: utf-8 -*-
"""
Created on Sun May 12 23:33:31 2024

@author: sjung_local
"""

FILE_A = "mscoco_val2014_annotations.json"
FILE_Q = "MultipleChoice_mscoco_val2014_questions.json"
BATCHSIZE = 128

###################################
#                                 #
#         VQA Benchmark           #
#                                 #
###################################

import json, torch, tqdm.notebook as tqdm, numpy as np, matplotlib.pyplot as plt

with open(FILE_A, "r") as f:
    labels = json.load(f)
with open(FILE_Q, "r") as f:
    vqa = json.load(f)

question_only = {q["question_id"]: q["question"] for q in vqa["questions"]}
question_answers = {q["question_id"]: q["multiple_choices"] for q in vqa["questions"]}

print(
    question_only[3506232],
    question_answers[3506232],
    len(question_answers[3506232])
)

###################################
#                                 #
#         VQA Prompt              #
#                                 #
###################################

prompt = """
You are a first-class question-answering model that answer multiple choice questions.
For each question, there are 18 possible answers given.
Only one of those answers is correct.

You are given questions in the following format:

Question: "Here will be the question you are supposed to answer."
0: "The first answer."
1: "The second answer."
2: "The third answer."
3: "The fourth answer."
4: "The fifth answer."
5: "The fifth answer."
6: "The sixth answer."
7: "The seventh answer."
8: "The eighth answer."
9: "The nineth answer."
10: "The tenth answer."
11: "The eleventh answer."
12: "The twelfth answer."
13: "The thirteenth answer."
14: "The fourteenth answer."
15: "The fifteenth answer."
16: "The sixteenth answer."
17: "The seventeenth answer."

Your answer can only contain the number of the correct answer (from 0 to 17), nothing more. Do not explain your answer.
Here is an example of your response, beginning after "Answer: ".
Answer: 5

Now answer the following question.

Question: "{}"
0: "{}."
1: "{}."
2: "{}."
3: "{}."
4: "{}."
5: "{}."
6: "{}."
7: "{}."
8: "{}."
9: "{}."
10: "{}."
11: "{}."
12: "{}."
13: "{}."
14: "{}."
15: "{}."
16: "{}."
17: "{}."

Answer: """

prompts = {qid: prompt.format(q,*question_answers[qid]) for qid, q in question_only.items()}

###################################
#                                 #
#         VQA                     #
#                                 #
###################################

from llama import Llama
import os, fire, torch
# required for Windows
# torch.distributed.init_process_group(backend="gloo", init_method="tcp://localhost:10001", rank=0, world_size=1)

generator = Llama.build(
    ckpt_dir = "D:/Models/llama3/Meta-Llama-3-8B-Instruct",
    tokenizer_path = "D:/Models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model",
    max_seq_len = 512,
    max_batch_size = BATCHSIZE,
)

bs = BATCHSIZE

prompts_all = list(prompts.values())
results_all = []
for i in tqdm.tqdm(range(len(prompts)//bs)):
    p = prompts_all[i*bs : (i+1)*bs]
    if len(p) > 0:
        results = generator.text_completion(
            p,
            max_gen_len = 1,
            temperature = 0,
            top_p = 0.9,
        )
        results_all += results

results = {qid: results_all[i]["generation"] for i, qid in enumerate(question_only.keys())}
torch.save(results, "llama3-qa.torch")
