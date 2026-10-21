import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import args, device
from models import Extractor, Classifier, Generator, Diffusion
from utils import AvgMeter, set_seed, add_gaussian_noise


class Client():
    def __init__(self, client_id, trainset, valset, testset):
        set_seed(args.seed)
        self.id = client_id
        self.trainloader = DataLoader(trainset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=2,
                                      pin_memory=True)
        self.valloader = DataLoader(valset,
                                    batch_size=args.batch_size * 10,
                                    shuffle=False,
                                    pin_memory=True)
        self.testloader = DataLoader(testset,
                                     batch_size=args.batch_size * 10,
                                     shuffle=False,
                                     pin_memory=True)
        self.train_size = len(trainset)
        self.diffusion = Diffusion(
            noise_steps=args.noise_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            img_size=args.feature_size,
            device=device
        )
        self.init_net()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.net[model].parameters()})
        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.net[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.net[model].eval()
            else:
                self.net[model].train()

    def save_client(self):
        optim_dict = {
            "net": self.net.state_dict(),
            "EC_optimizer": self.EC_optimizer.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict()
        }
        torch.save(optim_dict, args.checkpoint_dir + f"/client{self.id}.pkl")

    def load_client(self):
        checkpoint = torch.load(args.checkpoint_dir + f"/client{self.id}.pkl")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])

    def init_net(self):
        set_seed(args.seed)
        self.net = nn.ModuleDict()
        self.net["extractor"] = Extractor()
        self.net["classifier"] = Classifier()
        self.net["generator"] = Generator()

        # 初始化优化器
        self.EC_optimizer = optim.AdamW(
            self.get_params(["extractor", "classifier"]),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.generator_optimizer = optim.AdamW(
            self.net["generator"].parameters(),
            lr=args.g_lr,
            weight_decay=args.g_weight_decay
        )

        # 损失函数
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)

        self.net.to(device)
        self.frozen_net(["extractor", "classifier"], True)
        self.frozen_net(["generator"], False)

    def local_test(self):
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                if args.add_noise:
                    noise_level = args.noise_std * self.id / (args.n_clients - 1)
                    x = add_gaussian_noise(x, mean=0., std=noise_level)
                x, y = x.to(device), y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    def local_val(self):
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.valloader:
                if args.add_noise:
                    noise_level = args.noise_std * self.id / (args.n_clients - 1)
                    x = add_gaussian_noise(x, mean=0., std=noise_level)
                x, y = x.to(device), y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    def local_train(self, current_round):
        self.net.train()
        EC_loss_meter = AvgMeter()
        Gen_loss_meter = AvgMeter()

        # 训练分类模型
        self.frozen_net(["extractor", "classifier"], False)
        self.frozen_net(["generator"], True)

        for _ in range(args.classifier_epochs):
            for x, y in self.trainloader:
                if args.add_noise:
                    noise_level = args.noise_std * self.id / (args.n_clients - 1)
                    x = add_gaussian_noise(x, mean=0., std=noise_level)

                x, y = x.to(device), y.to(device)

                # 分类任务前向
                self.EC_optimizer.zero_grad()
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                loss = self.CE_criterion(pred, y)

                # 反向传播和优化
                loss.backward()
                self.EC_optimizer.step()
                EC_loss_meter.update(loss.item())

        # 训练生成模型
        self.frozen_net(["extractor", "classifier"], True)
        self.frozen_net(["generator"], False)

        for _ in range(args.generator_epochs):
            for x, y in self.trainloader:
                x, y = x.to(device), y.to(device)

                # 获取真实特征
                with torch.no_grad():
                    real_feat = self.net["extractor"](x)

                # 扩散模型训练
                self.generator_optimizer.zero_grad()

                # 采样时间步和添加噪声
                t = self.diffusion.sample_timesteps(real_feat.size(0))
                noisy_feat, noise = self.diffusion.noise_images(real_feat, t)

                # 预测噪声
                pred_noise = self.net["generator"](noisy_feat, t, y)

                # 计算损失
                loss = self.MSE_criterion(pred_noise, noise)

                # 反向传播和优化
                loss.backward()
                self.generator_optimizer.step()
                Gen_loss_meter.update(loss.item())

        return {
            "classifier_loss": EC_loss_meter.avg,
            "generator_loss": Gen_loss_meter.avg
        }