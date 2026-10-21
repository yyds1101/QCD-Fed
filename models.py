import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import resnet18
from config import args
from utils import weights_init


# ========================== 公共扩散模型组件 ==========================
class TimeEmbedding(nn.Module):
    def __init__(self, diffusion_steps, emb_dim=1024):
        super(TimeEmbedding, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, emb_dim // 4),  # 输入维度为1（标量时间步）
            nn.SiLU(),
            nn.Linear(emb_dim // 4, emb_dim)
        )

    def forward(self, t):
        t = t.unsqueeze(1).float()  # 形状：(B, 1)
        return self.time_mlp(t)  # 输出：(B, emb_dim)


class ConditionalDiffusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, diffusion_steps, time_emb_dim, num_classes):
        super().__init__()
        # 修正：正确传递diffusion_steps和time_emb_dim
        self.time_embed = TimeEmbedding(diffusion_steps, emb_dim=time_emb_dim)
        self.class_embed = nn.Embedding(num_classes, time_emb_dim)  # 类别嵌入维度与时间嵌入一致

        # 输入通道：in_ch + time_emb_dim * 2（时间和类别嵌入通道）
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + time_emb_dim * 2, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, y):
        B, C, H, W = x.shape
        t_emb = self.time_embed(t).view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, time_emb_dim, H, W)
        y_emb = self.class_embed(y).view(B, -1, 1, 1).expand(-1, -1, H, W)     # (B, time_emb_dim, H, W)
        # 拼接输入：x (B, in_ch, H, W) + t_emb + y_emb (各B, time_emb_dim, H, W)
        return self.conv(torch.cat([x, t_emb, y_emb], dim=1)) + self.residual(x)


# ========================== 模型架构定义 ==========================
if args.model == "lenet5":

    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            # 修正：使用args.image_channel作为输入通道（支持1/3通道）
            self.extractor = nn.Sequential(
                nn.Conv2d(args.image_channel, 6, 5),
                nn.AvgPool2d(2, 2),
                nn.Sigmoid(),
                nn.Conv2d(6, 16, 5),
                nn.AvgPool2d(2, 2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.extractor(x)


    class Classifier(nn.Module):
        def __init__(self, class_num=10):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, class_num),
            )

        def forward(self, x):
            return self.classifier(x)


    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_emb_dim = 64  # 时间嵌入维度（降低避免通道爆炸）
            # 修正：使用args.image_channel作为输入通道
            self.blocks = nn.ModuleList([
                ConditionalDiffusionBlock(
                    in_ch=args.image_channel,       # 输入通道匹配数据集（1/3）
                    out_ch=64,
                    diffusion_steps=args.diffusion_steps,
                    time_emb_dim=self.time_emb_dim,
                    num_classes=args.num_classes
                ),
                nn.AvgPool2d(2),
                ConditionalDiffusionBlock(64, 128, args.diffusion_steps, self.time_emb_dim, args.num_classes),
                nn.AvgPool2d(2),
                ConditionalDiffusionBlock(128, 256, args.diffusion_steps, self.time_emb_dim, args.num_classes),
                nn.Conv2d(256, args.image_channel, 1)  # 输出通道匹配输入通道
            ])

        def forward(self, x, t, y):
            for block in self.blocks:
                if isinstance(block, ConditionalDiffusionBlock):
                    x = block(x, t, y)
                else:
                    x = block(x)
            return x

elif args.model == "alexnet":
    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = nn.Sequential(
                nn.Conv2d(args.image_channel, 64, 11, 4, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(64, 192, 5, padding=2),
                nn.Sigmoid(),
                nn.MaxPool2d(3, 2),
            )

        def forward(self, x):
            return self.extractor(x)


    class Classifier(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Conv2d(192, 384, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            return self.classifier(x)


    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_emb_dim = 128
            self.blocks = nn.ModuleList([
                ConditionalDiffusionBlock(
                    args.image_channel, 64, args.diffusion_steps, self.time_emb_dim, args.num_classes
                ),
                nn.MaxPool2d(2),
                ConditionalDiffusionBlock(64, 192, args.diffusion_steps, self.time_emb_dim, args.num_classes),
                nn.MaxPool2d(2),
                ConditionalDiffusionBlock(192, 384, args.diffusion_steps, self.time_emb_dim, args.num_classes),
                ConditionalDiffusionBlock(384, 256, args.diffusion_steps, self.time_emb_dim, args.num_classes),
                nn.Conv2d(256, args.image_channel, 1)
            ])

        def forward(self, x, t, y):
            for block in self.blocks:
                if isinstance(block, ConditionalDiffusionBlock):
                    x = block(x, t, y)
                else:
                    x = block(x)
            return x

elif args.model == "resnet18":
    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            model = resnet18(pretrained=False)
            # 修正：使用args.image_channel作为输入通道（支持灰度图）
            model.conv1 = nn.Conv2d(args.image_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return x  # 移除多余的sigmoid（避免激活函数冲突）


    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            model = resnet18(pretrained=False, num_classes=args.num_classes)
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.avgpool = model.avgpool
            self.fc = model.fc

        def forward(self, x):
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return self.fc(torch.flatten(x, 1))


    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_emb_dim = 256
            # 修正：输入通道使用args.image_channel（支持1/3通道）
            self.initial = nn.Sequential(
                nn.Conv2d(args.image_channel, 64, 7, padding=3),
                nn.GroupNorm(8, 64),
                nn.SiLU()
            )
            self.down_blocks = nn.ModuleList([
                ConditionalDiffusionBlock(
                    64, 128, args.diffusion_steps, self.time_emb_dim, args.num_classes
                ),
                nn.MaxPool2d(2),
                ConditionalDiffusionBlock(
                    128, 256, args.diffusion_steps, self.time_emb_dim, args.num_classes
                ),
                nn.MaxPool2d(2)
            ])
            self.up_blocks = nn.ModuleList([
                ConditionalDiffusionBlock(
                    256, 128, args.diffusion_steps, self.time_emb_dim, args.num_classes
                ),
                nn.Upsample(scale_factor=2),
                ConditionalDiffusionBlock(
                    128, 64, args.diffusion_steps, self.time_emb_dim, args.num_classes
                ),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, args.image_channel, 3, padding=1)  # 输出通道匹配输入通道
            ])

        def forward(self, x, t, y):
            x = self.initial(x)
            for block in self.down_blocks:
                if isinstance(block, ConditionalDiffusionBlock):
                    x = block(x, t, y)
                else:
                    x = block(x)
            for block in self.up_blocks:
                if isinstance(block, ConditionalDiffusionBlock):
                    x = block(x, t, y)
                else:
                    x = block(x)
            return x

if __name__ == "__main__":
    # 测试代码（以fmnist为例，1通道，32x32图像）
    args.image_channel = 1
    args.num_classes = 10
    args.diffusion_steps = 1000

    x = torch.rand(8, args.image_channel, 32, 32)  # 1通道输入
    y = torch.randint(0, args.num_classes, (8,))
    t = torch.randint(0, args.diffusion_steps, (8,))

    # 初始化模型
    extractor = Extractor()
    classifier = Classifier()
    diffusion = DiffusionModel()

    # 前向测试
    E = extractor(x)
    print("Extractor output shape:", E.shape)  # 应输出 (8, 16, 5, 5)

    pred_noise = diffusion(x, t, y)
    print("Diffusion output shape:", pred_noise.shape)  # 应输出 (8, 1, 32, 32)

    EC = classifier(E)
    print("Classifier output shape:", EC.shape)  # 应输出 (8, 10)