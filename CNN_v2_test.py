import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utkface_dataset import UTKFaceDataset
from torchinfo import summary


# Define the CNN_v2 model
class CNN_v2(nn.Module):
    def __init__(self, args):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self, args):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(36864, 128) # image size 100
        self.fc2 = nn.Linear(128, args.num_classes)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.activation(self.conv4(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return self.output_activation(x)


# Create the model
args = lambda: None
args.num_channels=1
args.num_classes=2
args.lr=0.001
args.momentum=0.5
args.verbose=False
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
# Create the model
# model = CustomCNN(args)
model = CNN_v2(args)


batch_size = 32
# summary(model, input_size=(batch_size, args.num_channels, 100, 100))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the dataset and preprocess as needed
utk_transform = transforms.Compose([transforms.Resize((32, 32))])
# utk_transform = None

dataset = UTKFaceDataset('./input/UTKFace/', train=True, transform=utk_transform)
dataset_test = UTKFaceDataset('./input/UTKFace/', train=False, transform=utk_transform)

# def train(net, optimizer, dataloader, loss_func, num_epochs, val_dataloader):
#     net.train()

#     for epoch in range(num_epochs):
#         train_epoch_loss = []
#         val_epoch_loss = []
        
#         # Training loop
#         for batch_idx, (images, labels) in enumerate(dataloader):
#             images, labels = images.to(args.device), labels.type(torch.LongTensor).to(args.device)
            
#             optimizer.zero_grad()
#             log_probs = net(images)
#             loss = loss_func(log_probs, labels)

#             loss.backward()
                
#             optimizer.step()
                
#             if args.verbose and batch_idx % 10 == 0:
#                 print('Epoch [{}/{}], Update [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
#                     epoch+1, num_epochs, batch_idx * len(images), len(dataloader.dataset),
#                            100. * batch_idx / len(dataloader), loss.item()))
                
#             train_epoch_loss.append(loss.item())
        
#         avg_train_epoch_loss = sum(train_epoch_loss) / len(train_epoch_loss)
        
#         # Validation loop
#         net.eval()
#         with torch.no_grad():
#             for val_images, val_labels in val_dataloader:
#                 val_images, val_labels = val_images.to(args.device), val_labels.type(torch.LongTensor).to(args.device)
                
#                 val_log_probs = net(val_images)
#                 val_loss = loss_func(val_log_probs, val_labels)
#                 val_epoch_loss.append(val_loss.item())
        
#         avg_val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_epoch_loss:.6f}, Val Loss: {avg_val_epoch_loss:.6f}")

#     return net.state_dict()

def train(net, optimizer, dataloader, val_dataloader, loss_func, num_epochs):
    net.train()

    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.type(torch.LongTensor).to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
                        
            loss.backward()
                
            optimizer.step()
            
            train_epoch_loss += loss.item()
            _, predicted = torch.max(log_probs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        avg_train_epoch_loss = train_epoch_loss / len(dataloader)
        train_accuracy = correct_train / total_train
        
        # Validation loop
        net.eval()
        val_epoch_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_dataloader:
                val_images, val_labels = val_images.to(args.device), val_labels.type(torch.LongTensor).to(args.device)
                
                val_log_probs = net(val_images)
                val_loss = loss_func(val_log_probs, val_labels)
                val_epoch_loss += val_loss.item()
                
                _, val_predicted = torch.max(val_log_probs.data, 1)
                total_val += val_labels.size(0)
                correct_val += (val_predicted == val_labels).sum().item()
        
        avg_val_epoch_loss = val_epoch_loss / len(val_dataloader)
        val_accuracy = correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_epoch_loss:.6f}, Train Acc: {train_accuracy*100:.2f}%, Val Loss: {avg_val_epoch_loss:.6f}, Val Acc: {val_accuracy*100:.2f}%")

    return net.state_dict(), avg_val_epoch_loss

# def train(net, optimizer, dataloader, loss_func, num_epochs=30):
#     net.train()

#     epoch_loss = []
#     for epoch in range(num_epochs):
#         for batch_idx, (images, labels) in enumerate(dataloader):
#             images, labels = images.to(args.device), labels.type(torch.LongTensor).to(args.device)
            
#             optimizer.zero_grad()
#             log_probs = net(images)
#             loss = loss_func(log_probs, labels)

                    
#             loss.backward()
                
#             optimizer.step()
                
#             if args.verbose and batch_idx % 100 == 0:
#                 print('Update [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     batch_idx * len(images), len(dataloader.dataset),
#                         100. * batch_idx / len(dataloader), loss.item()))
                
#             epoch_loss.append(loss.item())
#     avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}")
    
#     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_func = nn.CrossEntropyLoss()

# Call the train function

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
trained_state_dict, avg_val_epoch_loss = train(model, optimizer, train_dataloader, val_dataloader, loss_func, 4000)
t = datetime.datetime.now()
datetime_str = t.strftime('%Y/%m/%d')
torch.save(trained_state_dict, './'+datetime_str+'_test_cnn_v2.pth')




# # Train the model
# num_epochs = 30
# train_losses, test_losses, train_accuracy, test_accuracy = [], [], [], []

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     correct_train = 0
#     total_train = 0

#     # Training loop
#     for images, labels in dataset: 
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()

#     train_losses.append(train_loss / len(dataset))
#     train_accuracy.append(correct_train / total_train)

#     # Validation loop
#     model.eval()
#     test_loss = 0.0
#     correct_test = 0
#     total_test = 0

#     with torch.no_grad():
#         for images, labels in dataset_test:
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total_test += labels.size(0)
#             correct_test += (predicted == labels).sum().item()

#     test_losses.append(test_loss / len(dataset_test))
#     test_accuracy.append(correct_test / total_test)

#     print(f"Epoch [{epoch+1}/{num_epochs}], "
#           f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy[-1]*100:.2f}%, "
#           f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracy[-1]*100:.2f}%")

# # Plotting the loss and accuracy
# fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
# ax = ax.ravel()
# ax[0].plot(train_losses, label='Train Loss', color='royalblue', marker='o', markersize=5)
# ax[0].plot(test_losses, label='Test Loss', color='orangered', marker='o', markersize=5)
# ax[0].set_xlabel('Epochs', fontsize=14)
# ax[0].set_ylabel('Cross Entropy Loss', fontsize=14)
# ax[0].legend(fontsize=12)
# ax[0].tick_params(axis='both', labelsize=12)
# ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
# ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)
# ax[1].set_xlabel('Epochs', fontsize=14)
# ax[1].set_ylabel('Accuracy', fontsize=14)
# ax[1].legend(fontsize=12)
# ax[1].tick_params(axis='both', labelsize=12)
# fig.suptitle(x=0.5, y=0.92, t="Loss and Accuracy of CNN Model by Epochs", fontsize=16)
# plt.show()
