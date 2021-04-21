import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.dataset_builder import build_dataset_test
from lib import models, utils
from lib.utils import save_predict, zipDir
import lib.scribble_generation as sg

from config import paths


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', default="interCNN", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="drive", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./server/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args


def predict(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    # k = enumerate(test_loader)
    if args.dataset == "camvid":
        for i, (input, _, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
            start_time = time.time()
            output = model(input_var)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # Save the predict greyscale output for Cityscapes official evaluation
            # Modify image name to meet official requirement
            name[0] = name[0].rsplit('_', 1)[0] + '*'
            save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)
            # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
            #              output_grey=True, output_color=False, gt_color=False)

    elif args.dataset == "cityscapes":
        for i, (input, _, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
            start_time = time.time()
            output = model(input_var)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # Save the predict greyscale output for Cityscapes official evaluation
            # Modify image name to meet official requirement
            name[0] = name[0].rsplit('_', 1)[0] + '*'
            save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)
            # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
            #              output_grey=True, output_color=False, gt_color=False)


    elif args.dataset == "drive":
        for i, (input, _, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
                start_time = time.time()
                output = model(input_var)
                # print(input.shape, output.shape)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
                # print(output.shape)
                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                # print(output.shape)

                # output[output == 1] = 255
                # Save the predict greyscale output for Cityscapes official evaluation
                # Modify image name to meet official requirement
                # name[0] = name[0].rsplit('_', 1)[0] + '*'
                # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                #              output_grey=False, output_color=True, gt_color=False)

            name = str(int(name[0].split('_')[0]))
            print(f"# {name}.png")
            save_predict(output, None, name, args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)

    else:
        for i, (input, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
                start_time = time.time()
                print(">>>>>>>>>>", input.shape)
                output = model(input_var)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                # output[output == 1] = 255
                # Save the predict greyscale output for Cityscapes official evaluation
                # Modify image name to meet official requirement
                # name[0] = name[0].rsplit('_', 1)[0] + '*'
                # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                #              output_grey=False, output_color=True, gt_color=False)
                # print(name)
            name = name[0].split('\\')
            name = name[1].split('/')
            save_predict(output, None, name[1], args.dataset, args.save_seg_dir,
                         output_grey=True, output_color=False, gt_color=False)


def predict_interact(args, test_loader, model, model_aux, val_iterations=20):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    model_aux.eval()
    total_batches = len(test_loader)
    # k = enumerate(test_loader)
    if args.dataset == "camvid":
        for i, (input, size, _, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
            start_time = time.time()
            output = model(input_var)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # Save the predict greyscale output for Cityscapes official evaluation
            # Modify image name to meet official requirement
            name[0] = name[0].rsplit('_', 1)[0] + '*'
            save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)
            # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
            #              output_grey=True, output_color=False, gt_color=False)

    elif args.dataset == "cityscapes":
        for i, (input, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
            start_time = time.time()
            output = model(input_var)
            torch.cuda.synchronize()
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
            output = output.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # Save the predict greyscale output for Cityscapes official evaluation
            # Modify image name to meet official requirement
            name[0] = name[0].rsplit('_', 1)[0] + '*'
            save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)
            # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
            #              output_grey=True, output_color=False, gt_color=False)

    elif args.dataset == "drive":
        for i, (input, labels, size, name) in enumerate(test_loader):
            with torch.no_grad():
                images = input.cuda()
                labels_of_user = labels  # simulate

                start_time = time.time()
                
                output_aux = model_aux(images)
                
                # Convert prediction and labels
                prediction_aux = utils.prediction_converter(output_aux)

                # init
                scribbles = sg.scribble_input(prediction_aux, labels_of_user, initial=True)
                prediction = prediction_aux.unsqueeze(1).float().cuda()

                for i in range(0, val_iterations):
                    # Make new prediction
                    outputs = model(images, prediction, scribbles)
                    new_prediction = utils.prediction_converter(outputs)

                    # updata
                    scribbles = sg.scribble_input(new_prediction, labels_of_user, scribbles)
                    prediction = new_prediction.unsqueeze(1).float().cuda()

                prediction = torch.squeeze(prediction)
                # print(prediction.shape)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
                output = prediction.cpu().data[0].numpy()
                # output = output.transpose(1, 2, 0)
                # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                name = str(int(name[0].split('_')[0]))
                print(f"# {name}.png")
                save_predict(output, None, name, args.dataset, args.save_seg_dir,
                             output_grey=False, output_color=True, gt_color=False)

    else:
        for i, (input, _, size, name) in enumerate(test_loader):
            with torch.no_grad():
                input_var = input.cuda()
                start_time = time.time()
                print(">>>>>>>>>>", input.shape)
                output = model(input_var)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time
                print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                # output[output == 1] = 255
                # Save the predict greyscale output for Cityscapes official evaluation
                # Modify image name to meet official requirement
                # name[0] = name[0].rsplit('_', 1)[0] + '*'
                # save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                #              output_grey=False, output_color=True, gt_color=False)
                # print(name)
            name = name[0].split('\\')
            name = name[1].split('/')
            save_predict(output, None, name[1], args.dataset, args.save_seg_dir,
                         output_grey=True, output_color=False, gt_color=False)


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    # model = build_model(args.model, num_classes=args.classes)

    # model = eval(args.model)
    model = eval(f"models.{args.model}")
    ck_path = eval(f"paths.{args.model}_pth")

    model = model(args.classes).cuda()
    model.load_state_dict(torch.load(ck_path))
    model.eval()

    if args.model == 'interCNN':
        model_aux = models.autoCNN(args.classes).cuda()
        model_aux.load_state_dict(torch.load(paths.autoCNN_pth))
        model_aux.eval()

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning testing")
    print("test set length: ", len(testLoader))

    if args.model == 'interCNN':
        predict_interact(args, testLoader, model, model_aux)
    else:
        predict(args, testLoader, model)


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'remote':
        args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', 'results')
    else:
        args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'seed':
        args.classes = 2
    elif args.dataset == 'remote':
        args.classes = 7
    elif args.dataset == 'drive':
        args.classes = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
