import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import args, logger, device
from models import Extractor, Classifier, DiffusionModel  # 修改导入模型
from utils import AvgMeter, set_seed, add_gaussian_noise, get_beta_schedule  # 新增调度函数


class Client():

    def __init__(self, client_id, trainset, valset, testset):
        set_seed(args.seed)
        self.id = client_id
        # 数据加载器保持不变
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True)
        self.valloader = DataLoader(valset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                    pin_memory=False)
        self.testloader = DataLoader(testset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                     pin_memory=False)
        self.train_size = len(trainset)
        self.val_size = len(valset)
        self.test_size = len(testset)

        # 初始化扩散调度参数
        self.betas = get_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        if args.add_noise:
            self.noise_std = args.noise_std * self.id / (args.n_clients - 1)
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d, noise_std:%2.6f"
                        % (self.id, self.train_size, self.val_size, self.test_size, self.noise_std))
        else:
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d"
                        % (self.id, self.train_size, self.val_size, self.test_size))
        self.init_net()

    def get_params(self, models):
        return [{"params": self.net[model].parameters()} for model in models]

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.net[model].parameters():
                param.requires_grad = not frozen
            self.net[model].train() if not frozen else self.net[model].eval()

    def init_net(self):
        set_seed(args.seed)
        self.net = nn.ModuleDict()
        # 替换Generator为DiffusionModel
        self.net["extractor"] = Extractor()
        self.net["classifier"] = Classifier()
        self.net["diffusion"] = DiffusionModel()  # 关键修改

        # 初始化优化器包含扩散模型参数
        self.frozen_net(["extractor", "classifier", "diffusion"], True)
        self.EC_optimizer = optim.Adam(
            self.get_params(["extractor", "classifier", "diffusion"]),  # 包含扩散模型
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.net.to(device)

        # 保留必要的损失函数
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)

    def diffuse(self, x, t):
        """完整扩散前向过程"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus * noise, noise

    def local_train(self, current_round):
        set_seed(args.seed)
        logger.info(f"Training Client {self.id} Start!")

        # 新增：初始化损失记录器
        overall_EC_loss = AvgMeter()
        overall_Diff_loss = AvgMeter()

        for epoch in range(args.local_epoch):
            EC_loss_meter = AvgMeter()
            Diff_loss_meter = AvgMeter()

            # 解冻需要训练的参数
            self.frozen_net(["extractor", "classifier", "diffusion"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                # 数据预处理
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)

                # ==== 扩散过程 ====
                t = torch.randint(0, args.diffusion_steps, (x.size(0),)).to(device)
                noisy_x, true_noise = self.diffuse(x, t)  # 添加噪声

                # ==== 训练步骤 ====
                self.EC_optimizer.zero_grad()

                # 特征提取和分类
                features = self.net["extractor"](noisy_x)
                preds = self.net["classifier"](features)
                EC_loss = self.CE_criterion(preds, y)

                # 扩散模型训练（预测噪声）
                pred_noise = self.net["diffusion"](noisy_x, t, y)
                diff_loss = self.MSE_criterion(pred_noise, true_noise)

                # 联合训练
                total_loss = EC_loss + diff_loss
                total_loss.backward()
                self.EC_optimizer.step()

                # 记录损失
                EC_loss_meter.update(EC_loss.item())
                Diff_loss_meter.update(diff_loss.item())

            # 冻结参数
            self.frozen_net(["extractor", "classifier", "diffusion"], True)

            # 累计 epoch 损失
            overall_EC_loss.update(EC_loss_meter.avg)
            overall_Diff_loss.update(Diff_loss_meter.avg)

            # 记录日志
            logger.info(
                f"Client:{self.id:2d} Epoch:{epoch:2d} "
                f"EC_Loss:{EC_loss_meter.avg:.4f} "
                f"Diff_Loss:{Diff_loss_meter.avg:.4f} "
                f"Val_Acc:{self.local_val():.4f}"
            )

        # 新增：返回平均损失
        return {
            "classifier_loss": overall_EC_loss.avg,
            "diffusion_loss": overall_Diff_loss.avg
        }

    # 保留原有的测试和验证方法
    def local_test(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += (pred.argmax(1) == y).sum().item()
                total += x.size(0)
        return correct / total

    def local_val(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.valloader:
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += (pred.argmax(1) == y).sum().item()
                total += x.size(0)
        return correct / total

    # 保留模型保存/加载方法
    def save_client(self):
        torch.save({
            "net": self.net.state_dict(),
            "EC_optimizer": self.EC_optimizer.state_dict()
        }, f"{args.checkpoint_dir}/client{self.id}.pth")

    def load_client(self):
        checkpoint = torch.load(f"{args.checkpoint_dir}/client{self.id}.pth")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])