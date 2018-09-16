# end-to-end-negotiator

Implementation of following paper in Tensorflow

Name: Deal or No Deal? End-to-End Learning for Negotiation Dialogues

Link: https://arxiv.org/abs/1706.05125

Original repository: https://github.com/facebookresearch/end-to-end-negotiator


- run_sv_versus_rl.py - for generating sample conversation between supervised trained model and model trained with reinforcement learning, optionally you may set USE_ROLLOUTS to True to make RL agent perform rollouts (see paper). Sample logs available at *.log files.

- sv_agent.py, rl_agent.py define Models and Agents that write and read the dialogue.

- train_sv.py trains model in supervised setting, from data.

- train_rl.py initialize sv_agent, rl_agent from pretrained model and continue training rl_agent with policy gradient

- model folder contains trained models

- end-to-end-negotiator/src/data/negotiate contains data from original repository


### Results (higher the better, min 0, max 10)

RL normal (without rollouts) versus SV: <br />
6.20 versus 4.93 

RL with rollouts versus SV: <br />
7.05 versus 4.57 


### Examples

From sv_versus_rl-rollout.log (line 1564): 

RlAgent (YOU) : book=(count:2 value:0) hat=(count:1 value:10) ball=(count:2 value:0) <br />
SVAgent (THEM) : book=(count:2 value:1) hat=(count:1 value:6) ball=(count:2 value:1) <br />
----- <br />
THEM: i would like the hat and both balls . \<eos> <br />
YOU: you can have the balls if i can have the hat and the books \<eos> <br />
THEM: i need a ball , you can have the hat and one book . \<eos> <br />
YOU: deal . \<eos> <br />
THEM: \<selection> <br />
----- <br />
book=1 hat=1 ball=0 book=1 hat=0 ball=2 <br />
----- <br />
counter 89, rl reward 10, sv reward 3, rl aver 7.04, sv aver 4.60 <br />

Explanation: RlAgent plays as "YOU" and SVAgent plays as "THEM". They see same number of objects 2 books, 1 hat, 2 balls and have different values for each of them. Their goal is to maximize their own reward using natural language to negotiate with each other. <br />
At the end we see that they devide items so that RlAgent takes 1 book and 1 hat (book=1 hat=1 ball=0) and SVAgent takes (book=1 hat=0 ball=2). It was due to "THEM" said to "YOU", to take these items (line 3) and "YOU" said "deal", after that attention mechanism over dialogue decided that last thing that was said is the most important and devided items in the way (book=1 hat=1 ball=0 book=1 hat=0 ball=2). <br />
Explanation: first three outputs (book=1 hat=1 ball=0) define how many of each object RlAgent got and last three (book=1 hat=0 ball=2) define how many of each object SVAgent got. For more examples check "sv_versus_rl-normal.log" and "sv_versus_rl-rollout.log" files.
