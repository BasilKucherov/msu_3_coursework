import torch.nn as nn
import torch.nn.functional as F
import torch

class DSCNN(nn.Module):
    def __init__(self, num_classes, in_channels, shape=(32, 32)):
        super(DSCNN, self).__init__()
        
        # C(64,10,4,2,2) // features,kernel size (time, freq), stride (time, freq)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(10, 4), padding='same')
        self.conv_1_bn = nn.BatchNorm2d(64)
        # DSC(64,3,1) // features, kernel size, stride
        self.conv_depth_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding='same')
        self.ds_bn_11 = nn.BatchNorm2d(64)
        self.conv_point_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same')
        self.ds_bn_12 = nn.BatchNorm2d(64)
        # DSC(64,3,1) // features, kernel size, stride
        self.conv_depth_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding='same')
        self.ds_bn_21 = nn.BatchNorm2d(64)
        self.conv_point_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same')
        self.ds_bn_22 = nn.BatchNorm2d(64)
        # DSC(64,3,1) // features, kernel size, stride
        self.conv_depth_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding='same')
        self.ds_bn_31 = nn.BatchNorm2d(64)
        self.conv_point_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same')
        self.ds_bn_32 = nn.BatchNorm2d(64)
        # DSC(64,3,1) // features, kernel size, stride
        self.conv_depth_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, padding='same')
        self.ds_bn_41 = nn.BatchNorm2d(64)
        self.conv_point_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding='same')
        self.ds_bn_42 = nn.BatchNorm2d(64)
        # Avg Pool
        self.pool = nn.AvgPool2d(shape)

        # fc
        self.fc =  nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(self.conv_1_bn(x))
        
        x = self.conv_depth_1(x)
        x = F.relu(self.ds_bn_11(x))
        x = self.conv_point_1(x)
        x = F.relu(self.ds_bn_12(x))

        
        x = self.conv_depth_2(x)
        x = F.relu(self.ds_bn_21(x))
        x = self.conv_point_2(x)
        x = F.relu(self.ds_bn_22(x))  

        x = self.conv_depth_3(x)
        x = F.relu(self.ds_bn_31(x))
        x = self.conv_point_3(x)
        x = F.relu(self.ds_bn_32(x))   

        x = self.conv_depth_4(x)
        x = F.relu(self.ds_bn_41(x))
        x = self.conv_point_4(x)
        x = F.relu(self.ds_bn_42(x))

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
