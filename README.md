# Trust Region Proximal Policy Optimization with Smooth Roll-Back



> This is the implementation of paper **Trust Region Proximal Policy Optimization with Smooth Roll-Back (TR-PPO-SRB) algorithm**  on Pytorch version, the corresponding paper is submitted to IEEE.
> See the record in https://www.researchgate.net/publication/370809773_Trust_Region_Proximal_Policy_Optimization_with_Smooth_Roll-Back

## Algorithms

- PPO-Clip
- PPO-KL
- Trust Region PPO (TR-PPO)
- PPO Smooth (PPO-S)
- Trust Region PPO with Roll-Back (TR-PPO-RB)
- Trust Region PPO with Smooth Roll-Back (TR-PPO-SRB)


## Packages recommodation

- python=3.8.5
- torch>=1.13.1
- gym==0.10.5


## Train agent with specific algorithm

For example, we train the agent with TR-PPO-SRB algorithm on the GPU with 8 parallel worker:

```shell
python main.py -GPU=True -number-of-parallel=8 -algorithm=TR-PPO-SRB
```

## Test agent with specific algorithm

For example, we test the agent with TR-PPO-SRB algorithm:

```shell
python main.py -evaluate=True -algorithm=TR-PPO-SRB
```

## Plot the training results

```shell
python main.py -plot=True -env-name=Swimmer-v2
```
