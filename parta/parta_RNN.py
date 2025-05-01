from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print(torch.cuda.device_count()) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

# 定义语言处理类
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# 转换非ASCII字符为ASCII字符
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# 规范化句子
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())  # 转为小写并去掉非英文字符
    s = re.sub(r"([.!?])", r" \1", s)  # 在标点后加空格
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)  # 去除非字母字符
    return s.strip()

# 读取数据并进行规范化
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # 读取数据并按行分割
    lines = open('cmn-eng/cmn.txt', encoding='utf-8').read().strip().split('\n')
    
    # 每行拆分为三个部分，保留英文和中文句子，忽略版权信息
    pairs = []
    for line in lines:
        parts = line.strip().split('\t')  # 按制表符分割
        if len(parts) >= 2:  # 确保每行有两个部分
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            pairs.append([english_sentence, chinese_sentence])  # 只保留英文和中文句子
    
    # 如果需要反转句子对
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# 过滤不符合要求的句子对
MAX_LENGTH = 100  # 最大句子长度
#eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re ")

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 数据预处理
# 句子转换为token ID
def tokenize_sentences(sentences, tokenizer):
    return [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]

# 数据预处理函数中，直接替换原有的 `Lang` 类
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    #print("Counting words...")

    input_sentences = [pair[0] for pair in pairs]
    output_sentences = [pair[1] for pair in pairs]

    # 使用GPT-2 tokenizer处理输入和输出句子
    input_tokenized = tokenize_sentences(input_sentences, tokenizer)
    output_tokenized = tokenize_sentences(output_sentences, tokenizer)

    # 直接返回tokenizer， 不需要手动创建词汇表
    return tokenizer, tokenizer, pairs


# 创建 DataLoader
# Define the tokenizer (using GPT2Tokenizer for Chinese and English)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # GPT-2 tokenizer should handle English and Chinese well

########The Seq2Seq Model
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
            # 如果解码器预测了EOS_token，则提前停止
            #print('decoder_input',decoder_input)
            # if (decoder_input == EOS_token).any():
            #     break
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def indexesFromSentence(tokenizer, sentence):
    # 使用GPT2Tokenizer的encode方法将句子转换为token ID
    return tokenizer.encode(sentence, add_special_tokens=True)

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('en', 'zh', True)
    print(random.choice(pairs))
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        #print('data',data)
        input_tensor, target_tensor = data
        #print('input_tensor, target_tensor = data',input_tensor, target_tensor)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.0002,
               print_every=1, plot_every=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def showPlot(points, filename="training_loss.pdf"):
    plt.figure(figsize=(10, 6))  # 设置图像大小
    fig, ax = plt.subplots()
    # 横坐标是epochs
    epochs = range(1, len(points) + 1)
    # 绘制损失曲线
    ax.plot(epochs, points, label="Loss", color='tab:blue', linewidth=2)
    # 设置坐标轴标签
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    # 设置标题
    ax.set_title('Training Loss Over Epochs', fontsize=14)
    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 设置y轴的刻度间隔
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # 自动调整x轴的刻度数量
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # 保存图像
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(tokenizer.decode([idx.item()]))

    return decoded_words, decoder_attn

def showAttention(input_sentence, output_words, attentions):
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
    #font_path = '/home/yaodong/.fonts/SimHei.ttf'  # SimHei字体路径
    #font_prop = font_manager.FontProperties(fname=font_path)
    font = FontProperties(fname="/home/yaodong/.fonts/SimHei.ttf") 
    fig = plt.figure()
    # 设置Matplotlib使用该字体
    #plt.rcParams['font.family'] = font_prop.get_name()
    ax = fig.add_subplot(111)
    #print('attentions.cpu().numpy()',attentions.cpu().numpy())
    print(attentions.shape)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=0, fontproperties=font)
    ax.set_yticklabels([''] + output_words, fontproperties=font)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # 清理input_sentence作为文件名，去掉空格和特殊字符
    safe_input_sentence = re.sub(r'[^\w\s-]', '', input_sentence)  # 去掉所有非字母、数字、下划线的字符
    safe_input_sentence = safe_input_sentence.replace(' ', '_')   # 用下划线替代空格

    # 保存图像为PDF，使用input_sentence作为文件名
    pdf_filename = f"{safe_input_sentence}.pdf"
    fig.savefig(pdf_filename, format='pdf')
    print(f"图像已保存为 {pdf_filename}")
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('attention:',attentions)
    print(attentions.shape)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])

def evaluateRandomly(encoder, decoder, n=10):
    input_lang, output_lang, pairs = prepareData('en', 'zh', True)
    for i in range(n):
        pair = random.choice(pairs)
        print(i,pair)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        evaluateAndShowAttention(pair[0])
        print('')
    

hidden_size = 128
batch_size = 32

#input_lang, output_lang, train_dataloader,test_dataloader = get_dataloader(batch_size)
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
encoder = EncoderRNN(input_lang.vocab_size, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.vocab_size).to(device)

train(train_dataloader, encoder, decoder, n_epochs=200, print_every=1, plot_every=1)
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)




evaluateAndShowAttention('今天天气不好，很多云')

#evaluateAndShowAttention('The weather is bad today.')

#evaluateAndShowAttention('好久不见')

#evaluateAndShowAttention('Long time no see.')

#evaluateAndShowAttention('这附近有商场吗？我想去购物')

#evaluateAndShowAttention('Is there a shopping mall nearby? I want to go shopping')

#evaluateAndShowAttention('做一个独立的人')

#evaluateAndShowAttention('Be an independent person.')

#evaluateAndShowAttention('他已经顺利毕业了')

#evaluateAndShowAttention('He has successfully graduated.')