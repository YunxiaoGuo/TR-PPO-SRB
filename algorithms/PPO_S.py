import torch
import torch.optim as opt
from algorithms import common

def algorithm(args,memory,network):
    ###########-----------------PPO-S-------------#############
    alpha = 0.2
    optimizer = opt.Adam(network.parameters(), lr=args.lr)
    common_process = common.common_process(memory,args,network)
    batch_size, oldlogproba, returns, advantages = common_process.sample_data()
    for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
        minibatch_newlogproba,minibatch_oldlogproba,minibatch_advantages,minibatch_returns,minibatch_newvalues = common_process.get_minbatch()
        ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
        surr2 = torch.where(ratio <= 1 - args.clip, (-alpha * torch.tanh(ratio - 1) + 1 + ratio + alpha * torch.tanh(torch.tensor(args.clip))) * minibatch_advantages,
                            torch.where(ratio > 1 + args.clip,
                                        (-alpha * torch.tanh(ratio - 1) + 1 - ratio - alpha * torch.tanh(torch.tensor(args.clip))) * minibatch_advantages,
                                        ratio * minibatch_advantages))

        surr1 = ratio * minibatch_advantages
        loss_surr = - torch.mean(torch.min(surr1, surr2))
        if args.lossvalue_norm:
            minibatch_return_6std = 6 * minibatch_returns.std()
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
        else:
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
        loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)
        total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
