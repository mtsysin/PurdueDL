# from ComputationalGraphPrimer import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tvt
from sklearn.metrics import confusion_matrix
from dataset import SentimentAnalysisDataset, TRAIN_DATASET, TEST_DATASET
import torch.nn as nn
from model import GRUnetWithEmbeddings, CustomRNN, RNNWrapper
import copy
import time

LR = 0.0001
EPOCHS = 10
SAVE_MODEL_PATH= "model"

if torch.cuda.is_available()== True: 
    device = torch.device("cuda:0")
else: 
    device = torch.device("cpu")

print(device)

def train(net, dataloader, display_train_loss=True, network_type = "original"): 
            print()
            filename_for_out = "performance_numbers_" + network_type + str(EPOCHS) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(device)
            
            ##  Note that the GRUnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = torch.optim.Adam(net.parameters(), 
                        lr=LR,        
                )
            training_loss_tally = []
            start_time = time.perf_counter()
            for epoch in range(EPOCHS):
                print("")
                running_loss = 0.0
                for i, data in enumerate(dataloader):    
                    review_tensor, sentiment = data
                    review_tensor = review_tensor.to(device)
                    sentiment = sentiment.to(device)

                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()

                    optimizer.zero_grad()

                    if network_type == "original":
                        hidden = net.init_hidden().to(device)
                        for k in range(review_tensor.shape[1]):
                            output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)

                    elif network_type == "custom-rnn":
                        output = net(review_tensor[0,...])

                    elif network_type == "pure-gru":
                        # print(review_tensor.shape)
                        output = net(review_tensor)

                    elif network_type == "pure-gru-bidirectional":
                        hidden = net.init_hidden().to(device)
                        output = net(review_tensor[0,...])

                    loss = criterion(output, torch.argmax(sentiment, 1))
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.5f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            torch.save(net.state_dict(), SAVE_MODEL_PATH + network_type + '.pt')
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n\n")
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig(f"training_loss{network_type}_{EPOCHS}.png")

def test(net, test_dataloader, network_type = "original"):
    net.load_state_dict(torch.load(SAVE_MODEL_PATH + network_type + '.pt'))
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    confusion_matrix = torch.zeros(2,2)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            review_tensor, sentiment = data

            if network_type == "original":
                hidden = net.init_hidden()
                for k in range(review_tensor.shape[1]):
                    output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                predicted_idx = torch.argmax(output).item()

            elif network_type == "custom-rnn":
                output = net(review_tensor[0,...])
                predicted_idx = torch.argmax(output).item()

            elif network_type == "pure-gru":
                output = net(review_tensor)
                predicted_idx = torch.argmax(output).item()


            elif network_type == "pure-gru-bidirectional":
                pass

            gt_idx = torch.argmax(sentiment).item()
            if i % 100 == 99:
                print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            if gt_idx == 0: 
                negative_total += 1
            elif gt_idx == 1:
                positive_total += 1
            confusion_matrix[gt_idx,predicted_idx] += 1
    print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
    out_percent = np.zeros((2,2), dtype='float')
    out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
    out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
    out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
    out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
    print("\n\nNumber of positive reviews tested: %d" % positive_total)
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                      "
    out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
    print(out_str + "\n")
    for i,label in enumerate(['true negative', 'true positive']):
        out_str = "%12s:  " % label
        for j in range(2):
            out_str +=  "%18s%%" % out_percent[i,j]
        print(out_str)


def main():

    dataset_train = SentimentAnalysisDataset(dataset_file = TRAIN_DATASET)
    dataset_test = SentimentAnalysisDataset(dataset_file = TEST_DATASET)

    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                batch_size=1, shuffle=False, num_workers=2)


    # model = GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)
    # model = CustomRNN(input_size=300, hidden_size=100, output_size=2)
    model = RNNWrapper(input_size=300, hidden_size=100, output_size=2, num_layers=2, bidirectional=True)
    NETWORK_TYPE = "pure-gru"
    # NETWORK_TYPE = "custom-rnn"
    # NETWORK_TYPE = "original"

    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    train(model, train_dataloader, network_type=NETWORK_TYPE)
    test(model, test_dataloader, network_type=NETWORK_TYPE)

if __name__=="__main__":
    main()

