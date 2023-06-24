import torch.nn as nn
import torch.nn.functional as F


dropout = 0.05

class NetBatch(nn.Module):
    def __init__(self):
        super(NetBatch, self).__init__()
        #C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout)
        ) # C1 : Input Size = 32 , output_size = 32, RF = 3

       # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout)
        ) # C2 : Input Size = 32 , output_size = 32, RF = 5

        self.skip_conv1 = nn.Conv2d(in_channels = 3, out_channels=16,kernel_size=(1,1),padding = 0 , bias=False)

       # c3
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False))
        self.trans1_batch_norm = nn.BatchNorm2d(16)
        self.trans1_dropout = nn.Dropout2d(dropout)
        # Input Size = 32 , output_size = 32, RF = 5

        # TRANSITION BLOCK 1
       # P1
        self.pool1 = nn.MaxPool2d(2, 2) # Input Size = 32 , output_size = 16, RF = 6

         #C3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(), 
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 10

        #C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 14
        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 14, RF = 18

          # c6
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
           # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout)
        ) # Input Size = 14 , output_size = 14, RF = 18

    

        #P2
        self.pool2 = nn.MaxPool2d(2, 2) # Input Size = 14 , output_size = 7, RF = 20

        # CONVOLUTION BLOCK 2
        #  C7
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout)
        ) # Input Size = 7 , output_size = 7 RF = 28

        #C8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout)
        ) # Input Size =7, output_size = 7, RF = 36

        # C9
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout)
        ) # Input Size =7 output_size = 5, RF = 44

         # GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)) # Input Size =5 output_size =1, RF = 48

        # C10
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        #    nn.ReLU()
        ) # Input Size =1 output_size = 1, RF = 48

    def forward(self, x):
        input = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        skip_1 = self.skip_conv1(input)
        x = self.transblock1(x)
        out_1 = skip_1+x
        out_1 = self.trans1_batch_norm(out_1)
        out_1 = self.trans1_dropout(out_1)
        x = self.pool1(out_1)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transblock2(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        

class NetLayer(nn.Module):
    def __init__(self):
        super(NetLayer, self).__init__()
        #C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
             nn.LayerNorm((32,32)),
            nn.Dropout2d(dropout)
        ) # C1 : Input Size = 32 , output_size = 32, RF = 3
        
       # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
             nn.LayerNorm((32,32)),
            nn.Dropout2d(dropout)) # C2 : Input Size = 32 , output_size = 32, RF = 5

        self.skip_conv1 = nn.Conv2d(in_channels = 3, out_channels=16,kernel_size=(1,1),padding = 0, bias=False)

       # c3
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False))
        self.trans1_layer_norm =  nn.LayerNorm((32,32))
        self.trans1_dropout = nn.Dropout2d(dropout)
        # Input Size = 32 , output_size = 32, RF = 5

        # TRANSITION BLOCK 1
       # P1
        self.pool1 = nn.MaxPool2d(2, 2) # Input Size = 32 , output_size = 16, RF = 6

         #C3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
             nn.LayerNorm((16,16)),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 10

        #C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((16,16)),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 14
        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm((14,14)),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 14, RF = 18

          # c6
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
           # nn.ReLU(),
            nn.LayerNorm((14,14)),
            nn.Dropout2d(dropout)
        ) # Input Size = 14 , output_size = 14, RF = 18

        #P2
        self.pool2 = nn.MaxPool2d(2, 2) # Input Size = 14 , output_size = 7, RF = 20

        # CONVOLUTION BLOCK 2
        #  C7
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((7,7)),
            nn.Dropout2d(dropout)
        ) # Input Size = 7 , output_size = 7 RF = 28

        #C8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((7,7)),
            nn.Dropout2d(dropout)
        ) # Input Size =7, output_size = 7, RF = 36

        # C9
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm((5,5)),
            nn.Dropout2d(dropout)
        ) # Input Size =7 output_size = 5, RF = 44

         # GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)) # Input Size =5 output_size =1, RF = 48

        # C10
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        #    nn.ReLU()
        ) # Input Size =1 output_size = 1, RF = 48

    def forward(self, x):
        input = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        skip_1 = self.skip_conv1(input)
        x = self.transblock1(x)
        out_1 = skip_1+x
        out_1 = self.trans1_layer_norm(out_1)
        out_1 = self.trans1_dropout(out_1)
        x = self.pool1(out_1)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transblock2(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class NetGroup(nn.Module):
    def __init__(self):
        super(NetGroup, self).__init__()
        #C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
             nn.GroupNorm(4,16), #no of groups, no of channels,
            nn.Dropout2d(dropout)
        ) # C1 : Input Size = 32 , output_size = 32, RF = 3

       # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
             nn.GroupNorm(8,32),
            nn.Dropout2d(dropout)) # C2 : Input Size = 32 , output_size = 32, RF = 5

        self.skip_conv1 = nn.Conv2d(in_channels = 3, out_channels=16,kernel_size=(1,1),padding = 0, bias=False)

       # c3
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False))
        self.trans1_group_norm =  nn.GroupNorm(4,16)
        self.trans1_dropout = nn.Dropout2d(dropout)
        # Input Size = 32 , output_size = 32, RF = 5

        # TRANSITION BLOCK 1
       # P1
        self.pool1 = nn.MaxPool2d(2, 2) # Input Size = 32 , output_size = 16, RF = 6

         #C3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 10

        #C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 16, RF = 14
        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8,32),
            nn.Dropout2d(dropout)
        ) # Input Size = 16 , output_size = 14, RF = 18

          # c6
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
           # nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(dropout)
        ) # Input Size = 14 , output_size = 14, RF = 18

        #P2
        self.pool2 = nn.MaxPool2d(2, 2) # Input Size = 14 , output_size = 7, RF = 20

        # CONVOLUTION BLOCK 2
        #  C7
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(dropout)
        ) # Input Size = 7 , output_size = 7 RF = 28

        #C8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8,32),
            nn.Dropout2d(dropout)
        ) # Input Size =7, output_size = 7, RF = 36

        # C9
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8,32),
            nn.Dropout2d(dropout)
        ) # Input Size =7 output_size = 5, RF = 44

         # GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)) # Input Size =5 output_size =1, RF = 48

        # C10
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        #    nn.ReLU()
        ) # Input Size =1 output_size = 1, RF = 48

    def forward(self, x):
        input = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        skip_1 = self.skip_conv1(input)
        x = self.transblock1(x)
        out_1 = skip_1+x
        out_1 = self.trans1_group_norm(out_1)
        out_1 = self.trans1_dropout(out_1)
        x = self.pool1(out_1)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.transblock2(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)