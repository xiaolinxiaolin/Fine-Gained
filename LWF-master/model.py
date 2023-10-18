import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

import torchvision.models as models
import torchvision.transforms as transforms

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).cuda()
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
	def __init__(self, classes, classes_map, args):
		# Hyper Parameters
		self.init_lr = args.init_lr
		self.num_epochs = args.num_epochs
		self.batch_size = args.batch_size
		self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)] #hardcoded decay schedule
		self.lr_dec_factor = 10
		
		self.pretrained = False
		self.momentum = 0.9
		self.weight_decay = 0.0001
		# Constant to provide numerical stability while normalizing
		self.epsilon = 1e-16

		# Network architecture
		super(Model, self).__init__()
		self.model = models.resnet34(pretrained=self.pretrained)
		self.model.apply(kaiming_normal_init)

		num_features = self.model.fc.in_features
		self.model.fc = nn.Linear(num_features, classes, bias=False)
		self.fc = self.model.fc  # 进行了备份
		self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])   # children是网络最上层，参考上面链接
		self.feature_extractor = nn.DataParallel(self.feature_extractor)    ## 放入gpu
		"""
		model = model.cuda() 
        device_ids = [0, 1] 	# id为0和1的两块显卡
        model = torch.nn.DataParallel(model, device_ids=device_ids)

		"""
		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
		self.n_classes = 0
		self.n_known = 0
		self.classes_map = classes_map

	def forward(self, x):    # 共享层，只是没有fc层，__init__函数提取出来的
		x = self.feature_extractor(x)    
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
 
	def increment_classes(self, new_classes):    #模型的增量改变
		"""Add n classes in the final fc layer"""
		n = len(new_classes)
		print('new classes: ', n)
		in_features = self.fc.in_features
		out_features = self.fc.out_features
		weight = self.fc.weight.data

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
		self.fc = self.model.fc
		
		kaiming_normal_init(self.fc.weight) # 先全部进行初始化
		self.fc.weight.data[:out_features] = weight    # 注意没有自己定义的这个fc层没有加bias
		self.n_classes += n

	def classify(self, images):  #进行前向运算，获得onehot的预测值。
		"""Classify images by softmax

		Args:
			x: input image batch
		Returns:
			preds: Tensor of size (batch_size,)
		"""
		_, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

		return preds

	def update(self, dataset, class_map, args):

		self.compute_means = True

		# Save a copy to compute distillation outputs
		prev_model = copy.deepcopy(self)  #保存旧网络
		"""
		python中对于对象的拷贝分为浅拷贝(copy)和深拷贝(deepcopy)两种方式。其中浅拷贝由“=”完成。而深拷贝由copy模块中deepcopy()函数担任。浅拷贝和深拷贝的区别是：
		浅拷贝只是将原对象在内存中引用地址拷贝过来了。让新的对象指向这个地址。
        而深拷贝是将这个对象的所有内容遍历拷贝过来了，相当于跟原来没关系了，所以如果你这时候修改原来对象的值跟他没关系了，不会随之更改。
        我觉得深拷贝和浅拷贝的区别在于：对于浅拷贝来说，如果有一方的内容发生了变化，另一方有可能也会发生变化而深拷贝，无论哪一方发生什么样的变化，另一方都不会发生改变。
		"""
		prev_model.cuda()

		classes = list(set(dataset.train_labels))     # 所有的类
		#print("Classes: ", classes)
		print('Known: ', self.n_known)
		if self.n_classes == 1 and self.n_known == 0:  
			new_classes = [classes[i] for i in range(1,len(classes))]
		else:
			new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

		if len(new_classes) > 0:
			self.increment_classes(new_classes)      # 增加类
			self.cuda()

		loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
											   shuffle=True, num_workers=12)

		print("Batch Size (for n_classes classes) : ", len(dataset))
		optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum = self.momentum, weight_decay=self.weight_decay) #设置优化器，开始40个epochs的训练，batch_size是64

		with tqdm(total=self.num_epochs) as pbar:
			for epoch in range(self.num_epochs):
				
				# Modify learning rate
				# if (epoch+1) in lower_rate_epoch:
				# 	self.lr = self.lr * 1.0/lr_dec_factor
				# 	for param_group in optimizer.param_groups:
				# 		param_group['lr'] = self.lr

				
				for i, (indices, images, labels) in enumerate(loader):
					seen_labels = []
					images = Variable(torch.FloatTensor(images)).cuda()
					seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
					labels = Variable(seen_labels).cuda()
					# indices = indices.cuda()

					optimizer.zero_grad()
					logits = self.forward(images)
					cls_loss = nn.CrossEntropyLoss()(logits, labels)     # L_new,注意labels是经过变化的labels
					if self.n_classes//len(new_classes) > 1:    # 判断是否是第一个task
						dist_target = prev_model.forward(images)  # 使用旧模型
						logits_dist = logits[:,:-(self.n_classes-self.n_known)]    # 之前task的所有类
						dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)    # L_old的计算
						loss = dist_loss+cls_loss
					else:       # 第一个task不用计算old_loss
						loss = cls_loss    




					loss.backward()
					optimizer.step()

					if (i+1) % 1 == 0:
						tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
							   %(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data))

				pbar.update(1)






