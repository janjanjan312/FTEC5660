## Reproducibility Work (FTEC5660) — Presentation Script (≤ 15 minutes)

Hi everyone.  
My name is **[Your Name]**.  
My student ID is **[Your ID]**.  
Today I’ll present my reproducibility work.

---

## Project summary

The project I reproduced is called **MetaMind**.  
It is a multi-agent system for social reasoning, inspired by **ToM**, which stands for **Theory of Mind**.  
The core idea is to infer hidden mental states and then use that to respond.

The pipeline has three agents.  
The **ToM (Theory of Mind) Agent** generates several hypotheses about what the user *really* means.  
It tries to infer things like beliefs, intentions, desires, or emotions.  
The **Domain Agent** critiques these hypotheses using context and social norms, then refines and selects the most plausible one.  
Finally, the **Response Agent** writes the final output based on that selected hypothesis.  
It aims to be coherent and socially appropriate.

---

## Reproduction target(s) + metric definition

My target is **ToMBench multiple-choice accuracy**.  
I scoped it down to one task and a small subset.

Specifically, I used the **Ambiguous Story Task**, and I evaluated the first **30** examples, with checkpoints at **10, 20, and 30**.

My metric is Accuracy, which equals **correct / total**.

I used the repo evaluation script: `evaluations/tombench/eval_tombench.py`.  
The prompt asks the model to output only A, B, C, or D.

---

## Results (my numbers vs reported numbers)

First, I ran the baseline with `hypothesis_count = 7`.  
I ran it three times on the same 30-example subset.  
The three final accuracies were **50.00%** (15/30), **70.00%** (21/30), and **56.67%** (17/30).  
The mean accuracy was **58.89%**, with a sample standard deviation of **10.18** percentage points.

So the baseline has noticeable run-to-run variance under this setup.

For “reported numbers”, the upstream repo/paper mentions overall ToMBench averages under GPT-4.  
For example, the repo README cites **74.8%** for base GPT-4.  
But my setup is different and my evaluation is only a small subset, so I do not treat those as directly comparable.

---

## Modification + results after modification

For the assignment, I made one small and measurable modification.  
I changed one ToM Agent parameter:

`hypothesis_count`: **7 → 3**.

I also ran this setting three times.  
The three final accuracies were **63.33%** (19/30), **60.00%** (18/30), and **66.67%** (20/30).  
The mean accuracy was **63.33%**, with a sample standard deviation of **3.34** percentage points.

So, on average, `hypothesis_count=3` is higher than `hypothesis_count=7` on this subset. 
Why might this happen in this setup?  

One reason is parsing.  
The evaluator extracts the **first** standalone A/B/C/D from the final response, which is the original design.  
Longer outputs can mention multiple letters.  
Then the first letter might not be the intended final choice.

Another reason is hypothesis noise.  
More hypotheses can also mean more low-quality hypotheses.  
That can add noise to downstream selection.

Also, items are not strictly independent.  
The evaluator reuses one application instance and updates **SocialMemory** across questions.  
So earlier items can affect later items.

And the sample is small.  
It is only 30 questions.  
So a few questions can move the percentage a lot.




One more observation.
`hypothesis_count=7` had higher variance in my three trials. The standard deviation was **3.34** for `hypothesis_count=3`, compared to **10.18** for `hypothesis_count=7`.
This is plausibly because it creates more branching and more multi-stage reasoning, so small differences can compound.




---

## Conclusions: what is reproducible, what isn’t, and why

Here is what I can reproduce.  
If I keep the same setup and the same ToMBench subset, the system runs end-to-end and gives me an accuracy number.  
And one ToM parameter, `hypothesis_count`, clearly changes that number.

In my runs on the first 30 Ambiguous Story Task examples, the mean accuracy was **58.89%** for `hypothesis_count=7` and **63.33%** for `hypothesis_count=3`.



What is not directly reproducible or comparable is the paper’s GPT-4 headline ToMBench averages.  
That is because the underlying model/provider differs, and I evaluated only a small subset rather than the full benchmark.

The results are sensitive to evaluation details, including small sample size, simple letter parsing, and shared SocialMemory state, so accuracy can fluctuate.

---

## Key lessons + recommendations to future users

Here are my key learnings about the system architecture.  
First, multi-agent pipelines amplify small evaluation choices.  
If outputs get longer, simple parsing rules can affect measured accuracy.

Second, the ToM stage is a real control knob.  
Changing `hypothesis_count` changes the upstream hypothesis set, and that can shift downstream decisions.

Third, SocialMemory makes the system stateful across items.  
So results can be sensitive to the order of questions and earlier outputs.

Finally, cost and latency grow with more hypotheses and more agent steps.  
So scoping the evaluation is practical.


---

## Closing

To summarize, I reproduced ToMBench accuracy on a scoped subset, changed `hypothesis_count` from 7 to 3, and accuracy increased from **50.00%** to **63.33%** on 30 examples.  
This result is scoped and sensitive to evaluation details.

Thanks for listening.  
I’m happy to take questions.

