"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
from tqdm import tqdm
import torch

import blobfile as bf
import pandas
import requests

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .eval_types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult

class EntropixSampler(SamplerBase):
    def __init__(self, model):
        self.model = model
    
    def __call__(self, messages: MessageList) -> str:
        # Convert messages to a single prompt
        prompt = messages[0]['content']  # For simple case, just take first message
        return self.model.generate(prompt)
    
    def _pack_message(self, content: str, role: str) -> dict:
        return {"role": role, "content": content}
    

class GPQAEval(Eval):
  def __init__(
    self,
    n_repeats: int = 4,
    variant: str = "diamond",
    num_examples: int | None = None,  # restrict to a subset of the data for debugging
  ):
    url = f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    df = pandas.read_csv(url)
    examples = [row.to_dict() for _, row in df.iterrows()]
    if num_examples:
      assert n_repeats == 1, "n_repeats only supported for num_examples = None"
      rng = random.Random(0)
      examples = rng.sample(examples, num_examples)
    examples = examples * n_repeats
    rng = random.Random(0)
    examples = [
      example | {"permutation": rng.sample(range(4), 4)} for example in examples
    ]
    self.examples = examples
    self.n_repeats = n_repeats

  def __call__(self, sampler: SamplerBase, batch_size=3) -> EvalResult:
    def fn(row: dict):
      choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
      ]
      choices = [choices[i] for i in row["permutation"]]
      correct_index = choices.index(row["Correct Answer"])
      correct_answer = "ABCD"[correct_index]
      choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
      )
      prompt_messages = [
        sampler._pack_message(
          content=format_multichoice_question(choices_dict), role="user"
        )
      ]
      print(prompt_messages)
      response_text = sampler(prompt_messages)
      match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
      extracted_answer = match.group(1) if match else None
      score = 1.0 if extracted_answer == correct_answer else 0.0
      html = common.jinja_env.from_string(HTML_JINJA).render(
        prompt_messages=prompt_messages,
        next_message=dict(content=response_text, role="assistant"),
        score=score,
        correct_answer=correct_answer,
        extracted_answer=extracted_answer,
      )
      convo = prompt_messages + [dict(content=response_text, role="assistant")]
      return SingleEvalResult(
        html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
      )
    
    results = []
    
    for i in tqdm(range(0, len(self.examples), batch_size)):
      torch.cuda.empty_cache()
      batch = self.examples[i:min(i + batch_size, len(self.examples))]
      if batch:  # Only process if batch is not empty
        batch_results = common.map_with_progress(fn, batch)
        results.extend(batch_results)
    #results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
  
  # h 
  '''
  def hf_call(self):
    def fn(row: dict):
      choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
      ]
      choices = [choices[i] for i in row["permutation"]]
      correct_index = choices.index(row["Correct Answer"])
      correct_answer = "ABCD"[correct_index]
      choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
      )
      prompt_messages = [
        sampler._pack_message(
          content=format_multichoice_question(choices_dict), role="user"
        )
      ]
      response_text = sampler(prompt_messages)
      match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
      extracted_answer = match.group(1) if match else None
      score = 1.0 if extracted_answer == correct_answer else 0.0
      html = common.jinja_env.from_string(HTML_JINJA).render(
        prompt_messages=prompt_messages,
        next_message=dict(content=response_text, role="assistant"),
        score=score,
        correct_answer=correct_answer,
        extracted_answer=extracted_answer,
      )
      convo = prompt_messages + [dict(content=response_text, role="assistant")]
      return SingleEvalResult(
        html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
      )

    results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
    '''
