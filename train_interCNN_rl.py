import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import argparse

from lib import models
from lib import eval_func as val
from lib.utils_rl import collect_episode, calculate_loss
from config import paths
from builders.dataset_builder import build_dataset_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=1, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs_interCNN', type=int, default=100)
    parser.add_argument('--val_internal', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount factor')
    parser.add_argument("--n_workers", type=int, default=5, help='Number of training workers')
    parser.add_argument('--alpha', type=float, default=0.5, help='Coefficient for value loss')
    parser.add_argument('--max_grad_norm', type=float, default=50, help='Max L2-norm for the gradients')
    parser.add_argument('--save_model', default=None, help='File to save the model')
    parser.add_argument('--load_model', default=None, help='File to load the model')
    args = parser.parse_args()

    # args.lr = 0.5

    print(args.num_classes)

    datas, trainLoader, valLoader = build_dataset_train("drive", 2, "565,584",
                                                        args.batch_size, "trainval",
                                                        False, False, num_workers=4)
    args.weight = torch.from_numpy(datas['classWeights']).cuda()
    print(args.weight)

    # import the model
    cnn1 = models.autoCNN(args.num_classes).cuda()
    cnn1.load_state_dict(torch.load(paths.autoCNN_pth))
    cnn1.eval()

    cnn2 = models.A3CC(args.num_classes).cuda()
    # cnn2 = models.interCNN(args.num_classes).cuda()

    # Define optimizer
    optimizer_2 = torch.optim.Adam(cnn2.parameters(), lr=args.lr)

    # Train the Model
    loss_list = []
    it_num = 0

    for epoch in range(args.num_epochs_interCNN):
        for i, (images, labels, _, _) in enumerate(trainLoader):
            # Increase the iteration number:
            it_num = it_num + 1

            labels = labels.type(torch.LongTensor)

            # update times for each image
            for i in range(1, args.max_iterations):
                # collect data
                logits, critics, actions, rewards = collect_episode(cnn2, cnn1, images, labels, args)
                # print(logits.shape, critics.shape, actions.shape)

                # Calculate the loss value using local model
                loss = calculate_loss(logits, critics, labels.cuda(), rewards, args)

                # Calculate gradients and store them in local model
                loss.backward()

                # Clip gradients
                clip_grad_norm_(cnn2.parameters(), args.max_grad_norm)

                optimizer_2.step()

                optimizer_2.zero_grad()

            print(f"epoch {it_num}: {loss}")

            if it_num % args.val_internal == 0:
                # Validation score tracker
                iou, dice = val.A3C_test(cnn1, cnn2, valLoader, num_class=args.num_classes)

                print(
                    f"Epoch Number: {epoch}/{args.num_epochs_interCNN} IoU: {iou} Dice Score: {dice}")

                # Save the parameters
                torch.save(cnn2.state_dict(), paths.save_model_pth + 'interCNN_last_xxx.pt')
