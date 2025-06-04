# Soft Actor Critic

Soft Actor Critic is a standard off-policy actor-critic algorithm

## Tricks / Optimizations / Modifications

- Observations are stored once to conserve CPU/GPU memory. For unknown reasons (someone should research this!), large replay buffers (around 1M samples) are typically needed for best training performance.
- Weight init is applied by default in the code. All biases are set to 0, linear layers are orthogonalized, and conv layers are initialized with delta orthogonal initialization.
- Cudagraphs and compile support are provided for faster training speeds.
- For visual RL a shared encoder is used for the Q-functions and actor by default. The encoder is only updated during the critic updates.
- This SAC code supports ensembling Q-functions as well. When the config `--sac.num_q` and `--sac.min_q` are set to 2, you get original SAC.


## Notes

- Depending on machine hardware using a CPU memory based replay buffer may be decently fast. Some tricks are used to help speed up CPU to GPU data transfer speeds. Generally it is still recommended to use a GPU based replay buffer when possible.

## Citation

If you use this baseline please cite the following

```
@inproceedings{DBLP:conf/icml/HaarnojaZAL18,
  author       = {Tuomas Haarnoja and
                  Aurick Zhou and
                  Pieter Abbeel and
                  Sergey Levine},
  editor       = {Jennifer G. Dy and
                  Andreas Krause},
  title        = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
                  with a Stochastic Actor},
  booktitle    = {Proceedings of the 35th International Conference on Machine Learning,
                  {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
                  10-15, 2018},
  series       = {Proceedings of Machine Learning Research},
  volume       = {80},
  pages        = {1856--1865},
  publisher    = {{PMLR}},
  year         = {2018},
  url          = {http://proceedings.mlr.press/v80/haarnoja18b.html},
  timestamp    = {Wed, 03 Apr 2019 18:17:30 +0200},
  biburl       = {https://dblp.org/rec/conf/icml/HaarnojaZAL18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

You should also cite RedQ if you ensemble Q-functions.

```
@inproceedings{DBLP:conf/iclr/ChenWZR21,
  author       = {Xinyue Chen and
                  Che Wang and
                  Zijian Zhou and
                  Keith W. Ross},
  title        = {Randomized Ensembled Double Q-Learning: Learning Fast Without a Model},
  booktitle    = {9th International Conference on Learning Representations, {ICLR} 2021,
                  Virtual Event, Austria, May 3-7, 2021},
  publisher    = {OpenReview.net},
  year         = {2021},
  url          = {https://openreview.net/forum?id=AY8zfZm0tDd},
  timestamp    = {Wed, 23 Jun 2021 17:36:40 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/ChenWZR21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```