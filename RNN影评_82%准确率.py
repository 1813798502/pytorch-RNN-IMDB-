import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


# 词干化：将词汇转换为它们的词干形式，例如将“running”、“runs”和“ran”都转换为“run”。NLTK也提供了词干化的工具。
def stem_words(text):
    return [stemmer.stem(word) for word in text]


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# 去除停用词：停用词是在文本处理中没有太多意义的常见词语，比如“and”、“the”等。可以使用NLTK库来移除这些停用词。
def remove_stopwords(text):
    return [word for word in text if word.lower() not in stop_words]


# 标记化：将文本转换为单词或字符的序列。可以使用Python的字符串方法或者NLTK库来完成这个任务。
def tokenize_text(text):
    return text.split()  # 这是一个简单的例子，可以使用更复杂的标记化方法


re_tag = re.compile(r'<[^>]+>')


def rm_tags(text):
    return re_tag.sub('', text)


start = time.time()
# 定义数据集文件夹路径
train_data_folder = '/content/aclImdb_v1/train'
test_data_folder = '/content/aclImdb_v1/test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_rating_from_filename(filename):
    # 使用正则表达式从文件名中提取评分（例如，文件名为"7_3.txt"）
    pattern = r"(\d+)_\d+.txt"  # 假设文件名格式为"数字_数字.txt"
    match = re.search(pattern, filename)
    if match:
        rating = int(match.group(1))
        return rating
    return None


def preprocess_text(text):
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 转换为小写
    text = text.lower()
    return text


# 读取数据函数
def load_imdb_data(folder):
    data = []
    labels = []
    for label in ['pos', 'neg']:
        folder_path = os.path.join(folder, label)

        for filename in os.listdir(folder_path):
            # rating = extract_rating_from_filename(filename)
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # 在这里可以进行文本预处理，例如去除标点符号、转换成小写等
                text = preprocess_text(text)
                text = tokenize_text(text)
                text = stem_words(text)
                text = remove_stopwords(text)
                data.append(text)  # 将文本拆分为单词列表
                # label = 'pos' if rating > 5 else 'neg'
                labels.append(label)
    return data, labels


# 加载训练数据集和测试数据集
train_texts, train_labels = load_imdb_data(train_data_folder)
test_texts, test_labels = load_imdb_data(test_data_folder)

# 将标签编码为数字
label_to_int = {'pos': 1, 'neg': 0}
encoded_train_labels = [label_to_int[label] for label in train_labels]
encoded_test_labels = [label_to_int[label] for label in test_labels]

# 将单词映射到索引
word_counter = Counter(word for text in train_texts + test_texts for word in text)
word_to_int = {word: idx + 1 for idx, (word, _) in enumerate(word_counter.most_common(10000))}

# 将单词列表转换为索引列表
indexed_train_texts = [[word_to_int.get(word, 0) for word in text] for text in train_texts]
indexed_test_texts = [[word_to_int.get(word, 0) for word in text] for text in test_texts]

# 确保所有序列长度相同（padding）
max_train_len = max(len(text) for text in indexed_train_texts)
max_test_len = max(len(text) for text in indexed_test_texts)
max_len = max(max_train_len, max_test_len)

padded_train_texts = [torch.tensor(text + [0] * (max_len - len(text))) for text in indexed_train_texts]
padded_test_texts = [torch.tensor(text + [0] * (max_len - len(text))) for text in indexed_test_texts]


# 数据集类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.stack(texts).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# 创建数据加载器
train_dataset = IMDBDataset(padded_train_texts, encoded_train_labels)
test_dataset = IMDBDataset(padded_test_texts, encoded_test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers=2):
        super(RNN, self).__init__()
        self.batch_size = None
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(batch_size)
        self.batch_size = x.size(0)
        embedded = self.embedding(x)
        # print(embedded.shape)
        embedded = self.dropout(embedded)
        # print(embedded.shape)
        output, hidden = self.rnn(embedded)
        # print(output,hidden)
        output = output[:, -1, :]  # Consider using the last time step's output
        # print(output)
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden


# 初始化模型、损失函数和优化器
INPUT_DIM = len(word_to_int) + 1
EMBEDDING_DIM = 512
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 150
all_loss = []
all_loss1 = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    for batch_data, batch_target in train_loader:
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_target.float().view(-1, 1))
        all_loss1.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # if i % 10 == 0:
        #     print(f'epoch:{i + 1}, loss:{loss.item():.4f}')
        # i += 1
    # plt.plot(range(len(all_loss1)), all_loss1)
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.show()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    all_loss.append(total_loss)
plt.plot(range(num_epochs), all_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 在测试集上测试模型
# 验证函数
# 设置模型为评估模式
model.eval()

# 初始化一些变量以存储预测和真实标签
predictions = []
true_labels = []

# 对测试集进行迭代
for batch_data, batch_target in test_loader:
    # 将数据移动到设备（GPU/CPU）
    batch_data, batch_target = batch_data.to(device), batch_target.to(device)
    # print("Batch data shape:", batch_data.shape)
    # print("Batch target shape:", batch_target.shape)
    batch_size = batch_data.size(0)
    print(batch_size)
    # 使用模型进行预测
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        output = model(batch_data)
        # print("Model output shape:", output.shape)
        # print(output)
        # 将输出转换为标签
        predicted_labels = (output > 0.5).squeeze(1).int()  # 假设使用阈值0.5来二分类
        # print(predicted_labels)
        # print(len(predicted_labels))
        # print("Predicted labels shape:", predicted_labels.shape)
        # print("True labels shape:", batch_target.shape)
        # 将预测的标签和真实标签添加到列表中
        predictions.extend(predicted_labels.tolist())
        true_labels.extend(batch_target[:batch_size].tolist())
        # print(predictions)
        # print(true_labels)
# print(predictions)
# print(len(predictions))
# print(true_labels)
# print(len(true_labels))
# 计算模型性能指标，比如准确率
correct = sum(1 for x, y in zip(predictions, true_labels) if x == y)
# print(correct)
accuracy = correct / len(true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

end = time.time()
print('运行时间：', end - start)
# # 保存模型
torch.save(model.state_dict(), 'rnn_model.pth')