import os
import torch
import torch.nn as nn
from torchsummary import summary

def conv_3x3_bn(input, output, stride):
    layer = nn.Sequential(
        nn.Conv2d(input, output, 3, stride, padding = 1, bias = False),
        nn.BatchNorm2d(output),
        nn.ReLU6(True))
    return layer

def conv_1x1_bn(input, output):
    layer = nn.Sequential(
        nn.Conv2d(input, output, 1, stride=1, padding = 0, bias = False),
        nn.BatchNorm2d(output),
        nn.ReLU6(True))
    return layer

class inverted_residuals(nn.Module):
    def __init__(self, input, output, stride, expand_ratio):
        super(inverted_residuals, self).__init__()
        channels = int(input * expand_ratio)
        self.identity = stride == 1 and input == output
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride = 1, padding = 1, groups = channels, bias= False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(True),

                nn.Conv2d(channels, output, 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(output)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input, channels,1,stride = 1,  padding=0, bias=False ),
                nn.BatchNorm2d(channels),
                nn.ReLU6(True),

                nn.Conv2d(channels, channels,3,stride = stride,  padding=1, groups= channels,  bias=False ),
                nn.BatchNorm2d(channels),
                nn.ReLU6(True),

                nn.Conv2d(channels, output, 1,stride = 1,  padding=0, bias=False ),
                nn.BatchNorm2d(output),
            )
    def forward(self, x):
        if self.identity:
            return x+self.conv(x)
        else:
            return self.conv(x)


class Mobilenetv2(nn.Module):
    def __init__(self, num_classes = 1000, width_multiplier = 1.0):
        super(Mobilenetv2, self).__init__()
        IR = [{'t': 1, 'c' : 16, 'n' : 1, 's' : 1},
            {'t': 6, 'c' : 24, 'n' : 2, 's' : 2},
            {'t': 6, 'c' : 32, 'n' : 3, 's' : 2},
            {'t': 6, 'c' : 64, 'n' : 4, 's' : 2},
            {'t': 6, 'c' : 96, 'n' : 3, 's' : 1},
            {'t': 6, 'c' : 160,'n' : 3, 's' : 2},
            {'t': 6, 'c' : 320,'n' : 1, 's' : 1},]

        input = 32 
        last_channel = 1280
        self.features = [conv_3x3_bn(3, input, stride = 2)]      ## 224 * 224 * 3  ---> 112 * 112 * 32
        for idx in range(len(IR)):   
            output = int(IR[idx]['c'] * width_multiplier) if IR[idx]['t'] > 1 else IR[idx]['c']
            for num in range(IR[idx]['n']):
                if num == 0:
                    self.features.append(inverted_residuals(input, output, stride = IR[idx]['s'], expand_ratio = IR[idx]['t'] )) 
                else:
                    self.features.append(inverted_residuals(input, output, stride = 1, expand_ratio= IR[idx]['t']))
                input = output
        
        self.features.append(conv_1x1_bn(input, last_channel ))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(last_channel, num_classes)
                        )
    
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

        

if __name__ == "__main__":
    model = Mobilenetv2()
    model = model.to('cuda')
    # image = torch.randn(1, 3, 224, 224)
    # image = image.to('cuda')
    # output = model(image)
    print(summary(model, (3, 224, 224)))
