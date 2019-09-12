import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian(x):
	mean = torch.mean(x)
	std = torch.std(x)
	return torch.exp(-((x-mean)**2)/(torch.std(x))**2) 

class GNCNN(nn.Module):
	def __init__(self):
		super(GNCNN, self).__init__()
		self.gaussian = gaussian
		self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0, bias=True)
		self.avg_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

		self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
		self.avg_pool2 =nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

		self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)
		self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

		self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)

		self.avg_pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

		self.conv5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0, bias=True)
		self.avg_pool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
		self.fc1 = nn.Linear(256*1*1, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 2)
		
	def forward(self, x):
		out = self.avg_pool1(gaussian(self.conv1(x)))
		out = self.avg_pool2(gaussian(self.conv2(out)))
		out = self.avg_pool3(gaussian(self.conv3(out)))
		out = self.avg_pool4(gaussian(self.conv4(out)))
		out = self.avg_pool5(gaussian(self.conv5(out)))
		out = out.reshape(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out
