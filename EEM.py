from helptrain import mylib
import pandas as pd
import torch

class Net(torch.nn.Module):
    def __init__(self,num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.linear1 = torch.nn.Linear(num_inputs, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 3)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.2)

    def forward(self, x):
        H1 = self.dropout1(self.relu(self.linear1(x.reshape(-1, self.num_inputs))))
        H2 = self.dropout2(self.relu(self.linear2(H1)))
        H3 = self.relu(self.linear3(H2))
        H4 = self.linear4(H3)
        return H4



def predict(net, test_iter):
    clses = []
    preds = []
    pred_score = []
    for x, y in test_iter:
        clses.append(y.squeeze(0).numpy())
        y_hat = torch.nn.functional.softmax(net(x))
        pred_score.append(y_hat.squeeze().detach().numpy())
        pred = y_hat.argmax(axis=1)
        preds.append(pred.squeeze(0).numpy())

   
    print("Saving files to txt....")
    with open("pr_curve.txt", 'w') as pr:
        for i in range(len(clses)):
            pr.write(str(clses[i]) + " " + 
                     str(preds[i]) + " " + 
                     str(format(pred_score[i][0], '.4f')) + " " +
                     str(format(pred_score[i][1], '.4f')) + " " + str(format(pred_score[i][2], '.4f'))+ "\n")
    print("All files have been written!")
    
    import numpy as np
    import matplotlib.pyplot as plt

    score_path = "./pr_curve.txt"  
    with open(score_path, 'r') as f:
        files = f.readlines()      

    lis_all = []
    for file in files:
        _, _, s1, s2, s3 = file.strip().split(" ")
        lis_all.append(s1)
        lis_all.append(s2)
        lis_all.append(s3)
    lis_order = sorted(set(lis_all))   

    macro_precis = []
    macro_recall = []

    for i in lis_order:

        true_p0 = 0         
        true_n0 = 0         
        false_p0 = 0        
        false_n0 = 0        

        true_p1 = 0
        true_n1 = 0
        false_p1 = 0
        false_n1 = 0

        true_p2 = 0
        true_n2 = 0
        false_p2 = 0
        false_n2 = 0
        for file in files:
            cls, pd, n0, n1, n2 = file.strip().split(" ")       
                                                               

            if float(n0) >= float(i) and cls == '0':               
                true_p0 = true_p0 + 1                             
            elif float(n0) >= float(i) and cls != '0':             
                false_p0 = false_p0 + 1                          
            elif float(n0) < float(i) and cls == '0':
                false_n0 = false_n0 + 1

            if float(n1) >= float(i) and cls == '1':               
                true_p1 = true_p1 + 1
            elif float(n1) >= float(i) and cls != '1':
                false_p1 = false_p1 + 1
            elif float(n1) < float(i) and cls == '1':
                false_n1 = false_n1 + 1

            if float(n2) >= float(i) and cls == '2':                
                true_p2 = true_p2 + 1
            elif float(n2) >= float(i) and cls != '2':
                false_p2 = false_p2 + 1
            elif float(n2) < float(i) and cls == '2':
                false_n2 = false_n2 + 1

        prec0 = (true_p0+0.00000000001) / (true_p0 + false_p0 + 0.00000000001)      
        prec1 = (true_p1+0.00000000001) / (true_p1 + false_p1 + 0.00000000001)
        prec2 = (true_p2+0.00000000001) / (true_p2 + false_p2 + 0.00000000001)

        recall0 = (true_p0+0.00000000001)/(true_p0+false_n0 + 0.00000000001)       
        recall1 = (true_p1+0.00000000001) / (true_p1 + false_n1+0.00000000001)
        recall2 = (true_p2+0.00000000001)/(true_p2+false_n2 + 0.00000000001)

        precision = (prec0 + prec1 + prec2)/3
        recall = (recall0 + recall1 + recall2)/3              
        macro_precis.append(precision)
        macro_recall.append(recall)

        # macro_precis.append(1)
        # macro_recall.append(0)
        print(macro_precis)
        print(macro_recall)

    x = np.array(macro_recall)
    y = np.array(macro_precis)
    plt.figure()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR curve')
    plt.plot(x, y)
    plt.show()



if  __name__ == '__main__':
    train_data = pd.read_csv(r'data\eye_data\TrainData.csv')
    train_data = train_data.iloc[:, 1:]

    val_data = pd.read_csv(r'data\eye_data\PredictionData.csv')
    val_data = val_data.iloc[:, 1:]

    train_features = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.iloc[:, -1].values.reshape(-1), dtype=torch.long)
    train_labels -= 1

    val_features = torch.tensor(val_data.iloc[:, :-1].values, dtype=torch.float32)
    val_labels = torch.tensor(val_data.iloc[:, -1].values.reshape(-1), dtype=torch.long)
    val_labels -= 1

    train_iter = mylib.load_array((train_features, train_labels), batch_size=8)
    val_iter = mylib.load_array((val_features, val_labels), batch_size=1)
    num_inputs = train_features.shape[1]
    net = Net(num_inputs)
    loss = torch.nn.CrossEntropyLoss()
    updater = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    # Training
    # train_epoch_loss, train_epoch_acc, val_epoch_acc = mylib.train_classify(net, 
    #                                                                     train_iter, val_iter=val_iter, 
    #                                                                     loss=loss, 
    #                                                                     num_epochs=30, updater=updater, device=torch.device('cuda:0'),
    #                                                                     save_epoch_module_root=r'four\weights')
    # print("train dataset max acc is {}, val dataset max acc is {}".format(max(train_epoch_acc), max(val_epoch_acc)))
    # predicting
    weight = r'four\weights\epoch-12-modules'
    net.load_state_dict(torch.load(weight)['net'])
    predict(net, val_iter)
    
