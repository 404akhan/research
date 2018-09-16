# A3C

### Description
I adapt paper distributional DQN variant into a3c framework, and try possible updates into value distribution of the state. Performance on Pong was not better.  

### Run
To test trained model on PongDeterministic-v4 or PongDeterministic-v0
```sh
$ python3.5 main.py --env-name "PongDeterministic-v4" --testing True --load-dir models-a3c/pongDetv4-distr.pth
$ python3.5 main.py --env-name "PongDeterministic-v0" --testing True --load-dir models-a3c/pongDetv0-distr.pth
```
To train a new model
```sh
$ python3.5 main.py --env-name "PongDeterministic-v4" --num-processes 16 --model-name pong-model-name
