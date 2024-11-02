# About
This is the official codebase of 'A dynamical clipping approach with task feedback for Proximal Policy Optimization' [paper_link](https://arxiv.org/abs/2312.07624) (our lastest version arxiv will be released soon)

In this research, we treat the clipping bounds secltion as a multi-arm bandit problem. And solve this problem via introducing Upper Confidence Bound (UCB), recommending the clipping bound with the highest UCB value in each iterations.

## Project Requirement
Linux Platform
stable_baselines3
torch

## Running Examples


## Citations

*If you utilize our codebase, please cite below:*
```c
@misc{zhang2024dynamicalclippingapproachtask,
      title={A dynamical clipping approach with task feedback for Proximal Policy Optimization}, 
      author={Ziqi Zhang and Jingzehua Xu and Zifeng Zhuang and Jinxin Liu and Donglin wang and Shuai Zhang},
      year={2024},
      eprint={2312.07624},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2312.07624}, 
}
```

## Thanks 

Our codebase is built upon stable_baselines3 [program_link](https://github.com/DLR-RM/stable-baselines3)
