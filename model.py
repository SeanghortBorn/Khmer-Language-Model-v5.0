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
b_sz = 1000
hid_sz = 128
epoch = 5000
lr = 0.0005
print_every_x = epoch*0.005

# Train model
path_tr = 'training-data'
all_words = read_txt_in_folder(path_tr)

training_size = p2int_selection(all_words, 1)
training_words = rand_selection(all_words, training_size)
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

for j in range(epoch):
    # random.shuffle(training_words)
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


        if j % print_every_x == 0 and i == 0:
            with torch.no_grad():
                predict = torch.argmax(outputs, dim=1)
                z = tensor2str(predict)
                n_correct = 0
                for n in range(len(y)): # y: ground-truth list
                    if y[n] == z[n]:
                        n_correct += 1

                # plot_loss.append(loss.item())
                print('Training: %.2f%%' % (j * 100 / epoch), 'Completed,')
                print(f'Epoch: {j}, Iter: {i}, Loss: {loss.item()}')
                print('n_correct: ', n_correct, ',\ttraining_size: ', training_size,
                      ',\tAccuracy: %.2f%%' % (n_correct * 100 / training_size))
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
# testing_size = len(all_words) - training_size  # 30% of all words
path_te = 'testing-data/output.txt'
all_words = read_from_txt(path_te)
testing_size = p2int_selection(all_words, 1)
testing_words = rand_selection(all_words, testing_size)

MAX_LEN = find_max_len(testing_words) + 1

encoder = Encoder(hid_sz).to(device)
decoder = Decoder(hid_sz, MAX_LEN).to(device)

encoder.load_state_dict(torch.load(encoder_sav))
decoder.load_state_dict(torch.load(decoder_sav))

encoder.eval()
decoder.eval()

i = 0
n_correct = 0

# b_sz = testing_size

predicted_x = []
predicted_z = []
while i + b_sz <= len(testing_words):
    # random.shuffle(testing_words)
    y = testing_words[i:i + b_sz] # input testing words
    x = [] # list of predicted word
    for _ in y:
        x.append(str_rand_err(_))

    # print(f'x: {x}, \ny: {y}')
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

    for word_z in z:
        predicted_z.append(word_z)

    for word_x in x:
        predicted_x.append(word_x)

    for n in range(len(y)):
        if y[n] == z[n]:
            n_correct += 1

    x_correct = 0
    z_correct = 0
    for n in range(len(y)):
        if y[n] == x[n]:
            x_correct += 1
        if y[n] == z[n]:
            z_correct += 1
    i += b_sz

print(f'x_correct: {x_correct}, z_correct: {z_correct}')

# print(predicted_words)
# Write predicted word to a new file
predicted_path = 'testing-data/predicted_z.txt'
with open(predicted_path, 'w', encoding='utf-8') as new_file:
    for word in predicted_z:
        new_file.write(word)
        new_file.write('\n')

predicted_path = 'testing-data/predicted_x.txt'
with open(predicted_path, 'w', encoding='utf-8') as new_file:
    for word in predicted_x:
        new_file.write(word)
        new_file.write('\n')

print(f'Training Summary: '
      f'\n\tTotal Dataset size: {len(all_words)}'
      f'\n\tTraining size: {training_size}'
      f'\n\tTesting size: {testing_size}'
      f'\n\tBatch Size: {b_sz}'
      f'\n\tHidden Size: {hid_sz}'
      f'\n\tLearning Rate: {lr}'
      f'\n\tTraining Epoch (loop): {epoch}')
print('Model ACCURACY: %.2f%%' % (z_correct * 100.0 / len(testing_words)))