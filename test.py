import torch
import torch.nn as nn
import torch.optim as optim
from input import X_train, y_train, X_test, y_test  # 导入训练和测试数据
from ResnetLstmTransformer import ResNet18WithTransformer
import os
import sys
import numpy as np
from collections import defaultdict

os.makedirs('train', exist_ok=True)

# 获取输出类别的数量
num_classes = 9  # 数据集中有 9 种不同的调制方式

# 动态计算全连接层输入大小
input_dim = (2, X_train.shape[1])

# 实例化最终模型
model = ResNet18WithTransformer(num_classes=num_classes, input_size=input_dim)

# 检查是否有 GPU 可用，如果有则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 训练模型
num_epochs = 200
batch_size = 128

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_accuracy = 0.0  # 记录最佳模型的准确率

# 导入训练数据，包括信号的 ID，SNR，标签，特征
train_data = np.load('trainDataset.npy', allow_pickle=True).item()
SNR_values = [-10, -6, -2, 2, 6, 10, 14, 18]

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 调整学习率
    scheduler.step()

    # 打印每个 epoch 的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    from collections import defaultdict
    import numpy as np

    # 假设 train_data['SNR'] 是一个 numpy 数组
    SNR_values = sorted(set(np.array(train_data['SNR']).flatten()))  # 将 SNR 转换为一维列表后去重

    # 测试模型在测试集上的准确率
    model.eval()
    correct_per_snr = defaultdict(int)  # 每个 SNR 对应预测正确的数量
    total_per_snr = defaultdict(int)  # 每个 SNR 对应的样本总数

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 获取 SNR 值
            snr_values = [train_data['SNR'][i].item() for i in range(len(labels))]

            # 统计每个 SNR 的样本总数和预测正确数
            for i in range(len(labels)):
                snr = snr_values[i]
                total_per_snr[snr] += 1
                if predicted[i] == labels[i]:
                    correct_per_snr[snr] += 1

    # 输出每个 SNR 的统计结果
    for snr in SNR_values:
        total_samples = total_per_snr[snr]
        correct_samples = correct_per_snr[snr]
        accuracy = 100 * correct_samples / total_samples if total_samples > 0 else 0
        print(
            f'SNR {snr}dB - Total Samples: {total_samples}, Correct Predictions: {correct_samples}, Test Accuracy: {accuracy:.2f}%')

    sys.stdout.flush()  # 强制刷新输出，实时显示训练过程

    # 保存测试集上表现最好的模型
    accuracy = 100 * sum(correct_per_snr.values()) / sum(total_per_snr.values())
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'train/best_modulation_classifier.pth')
        print('Best model saved.')

print('Training complete.')

# 创建 'train' 文件夹（如果不存在的话）
os.makedirs('train', exist_ok=True)

# 保存最终模型权重
torch.save(model.state_dict(), 'train/modulation_classifier.pth')
print('Model saved successfully in train directory.')
