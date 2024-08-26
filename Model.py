from torch import nn

class CNN(nn.Module):
    """Convolutional Neural Network."""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1: 3 x 50 x 50 -> 8 x 48 x 48
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 2: 8 x 48 x 48 -> 8 x 46 x 46
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 2: 8 x 46 x 46 -> 8 x 44 x 44
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 3: 8 x 44 x 44 -> 8 x 42 x 42
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 42 * 42, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x