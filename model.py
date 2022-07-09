import torch.nn as nn
import torch.optim as optim
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create models to train and test data: Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self, hid_sz):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=N_CHAR, hidden_size=hid_sz)

    def forward(self, x):
        """
        x: tensor (len, b, in_sz)
        return last hid state: tensor (b, hid_sz)
        """

        output, (h_n, c_n) = self.lstm(x)
        return torch.squeeze(h_n)


class Decoder(nn.Module):
    def __init__(self, hid_sz, max_len):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTMCell(input_size=N_CHAR + hid_sz, hidden_size=hid_sz)
        self.out = nn.Linear(hid_sz, N_CHAR)
        self.hid_sz = hid_sz
        self.max_len = max_len

    def forward(self, hid):
        """
        hid: tensor (b, hid_sz)
        return tensor (b, n_char, max_len)
        """
        input = input0_tensor(b_sz).to(device)
        cell = torch.randn(b_sz, self.hid_sz).to(device)
        outputs = []
        for _ in range(self.max_len):
            hid, cell = self.rnn(torch.cat((input, hid), dim=1), (hid, cell))
            outputs.append(self.out(hid))
        return torch.stack(outputs, dim=2)

# Adjust your parameters for the training and testing
b_sz = 1238
hid_sz = 256
epoch = 1000
lr = 0.001

# # Train model
path_tr = 'training-data'
training_words = read_txt_in_folder(path_tr)
training_words = remove_n(training_words) #remove /n

MAX_LEN = find_max_len(training_words) + 1

encoder = Encoder(hid_sz)
encoder = encoder.to(device)
decoder = Decoder(hid_sz, MAX_LEN)
decoder = decoder.to(device)

cost_fn = nn.CrossEntropyLoss(reduction='none')
opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)

# Variable to save training caches
encoder_sav = path_tr + '-encoder.sav'
decoder_sav = path_tr + '-decoder.sav'
optimizer_sav = path_tr + '-optimizer.sav'

plot_loss = []
for j in range(epoch):
    i = 0
    while i + b_sz <= len(training_words):
        y = training_words[i:i + b_sz]
        x = []
        for _ in y:
            x.append(str_rand_err(_))

        t_y, coef, y_len = label2tensor(y)  # tensor (b, len, in_sz)
        y_len = y_len.to(device)
        coef = coef.to(device)
        t_y = t_y.to(device)
        t_x = word2tensor(x).to(device)  # tensor (b, len, in_sz)
        t_x = t_x.permute(1, 0, 2).to(device)  # tensor (len, b, in_sz)

        opt.zero_grad()
        hid = encoder(t_x)
        outputs = decoder(hid)  # (b, n_char, max_len)

        loss = cost_fn(outputs[:, :, :t_y.size(1)], t_y)
        loss = (loss * coef).sum(dim=1).to(device) / y_len
        loss = loss.mean()


        # if j % print_every_x == 0 and i == 0:
        if j%1 == 0 and i == 0:
            with torch.no_grad():
                predict = torch.argmax(outputs, dim=1)
                z = tensor2str(predict)
                n_correct = 0
                for n in range(len(y)): # y: ground-truth list
                    if y[n] == z[n]:
                        n_correct += 1

                plot_loss.append(loss.item())
                print('Training: %.2f%%' % (j * 100 / epoch), 'Completed')
                print(f'Epoch: {j}, Iter: {i}, Loss: {loss.item()}')
                print('n_correct: ', n_correct, ',\tb_sz: ', b_sz,
                      ',\tAccuracy: %.2f%%' % (n_correct * 100 / b_sz))
                print(f'Input words\t\t: {x}')
                print(f'Predicted\t\t: {tensor2str(predict)}')
                print(f'Ground Truth\t: {y}\n')

                torch.save(encoder.state_dict(), encoder_sav)
                torch.save(decoder.state_dict(), decoder_sav)
                torch.save(opt.state_dict(), optimizer_sav)

        loss.backward()
        opt.step()
        i += b_sz


# Testing Section
path_te = 'testing-data/output.txt'
testing_words = read_from_txt(path_te)
testing_words = remove_n(testing_words) # remove /n

MAX_LEN = find_max_len(testing_words) + 1

encoder = Encoder(hid_sz).to(device)
decoder = Decoder(hid_sz, MAX_LEN).to(device)

encoder.load_state_dict(torch.load(encoder_sav))
decoder.load_state_dict(torch.load(decoder_sav))

encoder.eval()
decoder.eval()

i = 0
n_correct = 0

b_sz = len(testing_words)

predicted = []
while i + b_sz <= len(testing_words):
    y = testing_words[i:i + b_sz] # input testing words
    x = []
    for _ in y:
        x.append(str_rand_err(_))


    t_y, coef, y_len = label2tensor(y)  # tensor (b, len, in_sz)
    t_y = t_y.to(device)
    coef = coef.to(device)
    y_len = y_len.to(device)
    t_x = word2tensor(x)  # tensor (b, len, in_sz)
    t_x = t_x.permute(1, 0, 2)  # tensor (len, b, in_sz)
    t_x = t_x.to(device)

    hid = encoder(t_x)
    outputs = decoder(hid)  # (b, n_char, max_len)

    predict = torch.argmax(outputs, dim=1)
    z = tensor2str(predict)

    for word in z:
        predicted.append(word)

    n_correct = 0
    for n in range(len(y)):
        if y[n] == z[n]:
            n_correct += 1
    i += b_sz

# Write predicted word to a new file
predicted_path = 'testing-data/predicted.txt'
with open(predicted_path, 'w', encoding='utf-8') as new_file:
    for word in predicted:
        new_file.write(word)
        new_file.write('\n')


print(f'Training Summary: '
      f'\n\tTraining size: {len(training_words)}'
      f'\n\tTesting size: {len(testing_words)}'
      f'\n\tBatch Size: {b_sz}'
      f'\n\tHidden Size: {hid_sz}'
      f'\n\tLearning Rate: {lr}'
      f'\n\tTraining Epoch (loop): {epoch}')
# print('Testing ACCURACY: %.2f%%' % (n_correct * 100.0 / len(testing_words)))
print('Testing ACCURACY: %.2f%%' % (n_correct * 100.0 / b_sz))