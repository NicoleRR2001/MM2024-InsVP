import torch
import torch.nn as nn

def get_token_prompt(args):
    return TokenPrompt(args)



class TokenPrompt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kernel_size_1 = args.TP_kernel_1
        k = self.kernel_size_1
        if self.args.token_prompt_type == "add":
            self.p = args.p_len
        elif self.args.token_prompt_type == "token":
            self.p = args.p_len // 2
        p = self.p
        self.conv1 = nn.Conv2d(3, p, kernel_size=k, padding=int((k-1)/2))
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(4, 4)
        self.dropout1 = nn.Dropout(0.1)
        self.kernel_size_2 = args.TP_kernel_2
        k = self.kernel_size_2
        self.conv2 = nn.Conv2d(p, 3*p, kernel_size=k, padding=int((k-1)/2))
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(3, 3)
        self.dropout2 = nn.Dropout(0.1)
        self.kernel_size_3 = args.TP_kernel_3
        k = self.kernel_size_3
        self.conv3 = nn.Conv2d(3*p, 3*p, kernel_size=k, padding=int((k-1)/2))
             
    def forward(self, x, layer):
        x = self.conv1(x) 
        x = self.relu1(x)
        x = x[:,:,8:216,8:216]
        x = self.pool1(x) # [B, 9, 56, 56]
        x = self.dropout1(x)
        x = self.conv2(x) 
        x = self.relu2(x)
        x = x[:,:,2:50,2:50]
        x = self.pool2(x) # [B, 27, 16, 16]
        x = self.dropout2(x)
        x = self.conv3(x) 
        x = x.reshape(-1, self.p, 768)
        return x
    
