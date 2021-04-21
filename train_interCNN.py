__author__ = 'gbredell'

# from data import NCI_ISBI_2013 as nci
# from data import data_loader as dl
from builders.dataset_builder import build_dataset_train

import numpy as np
import cupy as cp
from config import paths
from lib import models
from lib import utils
from lib import scribble_generation as sg
import torch
from lib import eval_func as val

learning_rate = 0.0001
num_epochs_interCNN = 8
batch_size = 1
val_internal = 10
max_iterations = 11
binary = True

if __name__ == "__main__":
    if binary:
        num_classes = 2
    else:
        num_classes = 3

    print(num_classes)

    datas, trainLoader, valLoader = build_dataset_train("drive", 2, "565,584",
                                                        batch_size, "trainval",
                                                        False, False, num_workers=4)
    weight = torch.from_numpy(datas['classWeights']).cuda()

    # import the model
    cnn1 = models.autoCNN(num_classes).cuda()
    cnn1.load_state_dict(torch.load(paths.autoCNN_pth))
    cnn1.eval();

    cnn2 = models.interCNN(num_classes).cuda()

    # Define optimizer
    optimizer_2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    # Train the Model
    loss_list = []
    it_num = 0

    for epoch in range(num_epochs_interCNN):
        for i, (images, labels, _, _) in enumerate(trainLoader):
            # Increase the iteration number:
            it_num = it_num + 1

            # images = images.unsqueeze(1)
            images = images.float().cuda()
            labels_user_model = labels.type(torch.LongTensor)
            labels = labels.type(torch.LongTensor).cuda()

            # Prediction from CNN1
            outputs = cnn1(images)

            # Transform output to correct size and format
            prediction = utils.prediction_converter(outputs)

            # Use image, prediction and scribbles as new input
            scribbles = sg.scribble_input(prediction, labels_user_model, initial=True)
            prediction = prediction.unsqueeze(1).float().cuda()

            # print(scribbles.shape)

            # print(cp.unique(scribbles), cp.unique(prediction), cp.unique(labels_user_model))
            # for i in cp.unique(scribbles):
            #     print(i, cp.sum(scribbles == i))
            # print(scribbles)

            # init state, i.e. env.reset()
            observation = (images, prediction, scribbles)
            prev_ce = 0

            for i in range(1, max_iterations):
                optimizer_2.zero_grad()
                outputs = cnn2(observation)

                loss = utils.cross_entropy2d(input=outputs, target=labels, weight=weight, size_average=True)

                loss.backward()
                optimizer_2.step()

                # Transform output to correct size and format
                prediction = utils.prediction_converter(outputs)

                # Use image, prediction and scribbles as new input
                scribbles = sg.scribble_input(prediction, labels_user_model, scribbles)
                prediction = prediction.unsqueeze(1).float().cuda()

                # i.e. env.step()
                observation = (images, prediction, scribbles)

            if it_num % val_internal == 0:
                # Validation score tracker
                cnn1_dc, cnn2_dc = val.interCNN_test(cnn1, cnn2, valLoader, num_class=num_classes,
                                                     checker=True)
                if it_num == val_internal:
                    class_cnn1_score = cnn1_dc
                    class_cnn2_score = cnn2_dc
                else:
                    class_cnn1_score = np.concatenate((class_cnn1_score, cnn1_dc), axis=0)
                    class_cnn2_score = np.concatenate((class_cnn2_score, cnn2_dc), axis=0)

                loss_list = np.append(loss_list, loss.item())
                # print("Epoch Number: ", epoch, '/', num_epochs_interCNN, " Dice Score: ",
                #       np.mean(cnn2_dc, axis=2).flatten(), " Loss: ", loss.data.cpu())
                print(
                    f"Epoch Number: {epoch}/{num_epochs_interCNN} Dice Score: {np.mean(cnn2_dc, axis=2).flatten()} Loss: {loss.data.cpu()}")

                # Save the parameters
                np.save(paths.save_val_pth + 'class_cnn1_score.npy', class_cnn1_score)
                np.save(paths.save_val_pth + 'class_cnn2_score.npy', class_cnn2_score)
                torch.save(cnn2.state_dict(), paths.save_model_pth + 'interCNN_last_xxx.pt')

                if len(loss_list) > 51:
                    # Save the cnn with the best validation score out of the average
                    if np.mean(class_cnn2_score[-51:-1, :]) < np.mean(class_cnn2_score[-50:, :]):
                        torch.save(cnn2.state_dict(), paths.save_model_pth + 'interCNN_best_xxx.pt')
