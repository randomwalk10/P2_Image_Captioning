import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.num_layers_ = num_layers
        self.hidden_size_ = hidden_size
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, \
                            num_layers=num_layers, batch_first=True, \
                            dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # word embedding (batch_size, n_seq) -> (batch_size, n_seq, embed_size)
        captions = self.embeds(captions)
        # combine features and captions into input of size (batch_size, n_seq,
        # embed_size)
        features = features.view(captions.shape[0], 1, captions.shape[2])
        x = torch.cat((features, captions), 1)
        x = x[:, :-1, :]
        # lstm
        h0 = torch.zeros(self.num_layers_, captions.shape[0], self.hidden_size_)
        c0 = torch.zeros(self.num_layers_, captions.shape[0], self.hidden_size_)
        h0 = h0.to(device)
        c0 = c0.to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # fc layer
        output = output.contiguous().view(-1, self.hidden_size_)
        output = self.fc(output)
        output = output.view(captions.shape[0], captions.shape[1], -1)
        # return
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # reshape inputs to (batch_size, n_seq, embed_size)
        x = inputs.view(1, 1, -1)
        # init h and c
        if not states:
            h = torch.zeros(self.num_layers_, 1, self.hidden_size_)
            c = torch.zeros(self.num_layers_, 1, self.hidden_size_)
        else:
            h, c = states
        h = h.to(device)
        c = c.to(device)
        # sample
        outputs = []
        for i in range(max_len):
            # lstm
            x, (h, c) = self.lstm(x, (h, c))
            # fc
            x = x.contiguous().view(-1, self.hidden_size_)
            x = self.fc(x)
            # predict
            pred = x.argmax()
            # append to outputs
            outputs.append(pred.item())
            # break if it is the end word
            if pred.item() is 1:
                break
            # embed
            x = self.embeds(pred.view(1, 1))
        # return
        return outputs
