import argparse
import datetime
import logging
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# 基础配置
parser.add_argument('--algorithm', type=str, default="fedgen",
                    choices=["local", "fedavg", "fedsplit", "fedprox", "fedgen", "feddf", "fedcg", "fedcg_w"])
parser.add_argument('--dataset', type=str, default="cifar",
                    choices=["fmnist", "digit", "office", "cifar", "domainnet"])
parser.add_argument('--model', type=str, default="resnet18", choices=["lenet5", "resnet18", "diffusion"])
parser.add_argument('--distance', type=str, default="mse", choices=["none", "mse", "cos"])

# 训练参数
parser.add_argument('--seed', type=int, default=1, help='随机种子')
parser.add_argument('--batch_size', type=int, default=32, help='训练批量大小')
parser.add_argument('--lr', type=float, default=3e-4, help="学习率")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="权重衰减")

# 扩散模型参数
parser.add_argument('--diffusion_steps', type=int, default=10, help="扩散过程总步数")
parser.add_argument('--beta_schedule', type=str, default="cosine",
                    choices=["linear", "cosine", "squaredcos"], help="噪声调度策略")
parser.add_argument('--beta_start', type=float, default=1e-4, help="初始beta值")
parser.add_argument('--beta_end', type=float, default=0.02, help="终止beta值")
parser.add_argument('--diffusion_lr', type=float, default=1e-4, help="扩散模型学习率")

# 联邦学习参数
parser.add_argument('--n_clients', type=int, default=5, help="客户端数量")
parser.add_argument('--local_epoch', type=int, default=1, help="客户端本地训练轮次")
parser.add_argument('--global_epoch', type=int, default=3, help="服务器聚合轮次")
parser.add_argument('--early_stop_rounds', type=int, default=3, help="早停轮次")

# 数据增强参数
parser.add_argument('--add_noise', action='store_true', help="输入加噪")
parser.add_argument('--input_noise_std', type=float, default=0.1, help="输入噪声标准差")

# 差分隐私参数
parser.add_argument('--dp', action='store_true', help="启用差分隐私")
parser.add_argument('--clip_threshold', type=float, default=1.0, help="梯度裁剪阈值")
parser.add_argument('--noise_scale', type=float, default=0.1, help="高斯噪声规模")
parser.add_argument('--epsilon', type=float, default=1.0, help="隐私预算epsilon")
parser.add_argument('--delta', type=float, default=1e-5, help="隐私预算delta")

# 系统参数
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--data_dir', type=str, default="./data/", help="数据集路径")

# 解析参数
args = parser.parse_known_args()[0]

# 数据集配置（关键修改：添加image_channel定义）
if args.dataset == "fmnist":
    args.image_channel = 1  # 灰度图（1通道）
    args.num_classes = 10
elif args.dataset == "cifar":
    args.image_channel = 3  # 彩色图（3通道）
    args.num_classes = 10
elif args.dataset == "digit":
    args.n_clients = 4
    args.image_channel = 1  # 灰度图（1通道）
    args.num_classes = 10
elif args.dataset == "office":
    args.n_clients = 4
    args.image_channel = 1  # 灰度图（1通道）
    args.num_classes = 10
elif args.dataset == "domainnet":
    args.n_clients = 5
    args.image_channel = 3  # 彩色图（3通道）
    args.num_classes = 10

# 模型架构配置
if args.model == "lenet5":
    args.image_size = 32
    args.feature_num = 16
    args.feature_size = 5
elif args.model == "resnet18":
    args.image_size = 224
    args.feature_num = 128
    args.feature_size = 28
elif args.model == "diffusion":
    args.image_size = 32 if args.dataset in ["fmnist", "cifar"] else 224
    args.feature_num = 64  # UNet基础通道数
    args.feature_size = args.image_size

# 实验路径配置
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_name = f"{args.algorithm}_{args.model}_{args.dataset}_clients{args.n_clients}"
args.checkpoint_dir = os.path.join("experiments", exp_name, "checkpoints")
os.makedirs(args.checkpoint_dir, exist_ok=True)

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s @ %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 文件日志
file_handler = logging.FileHandler(os.path.join(args.checkpoint_dir, "training.log"))
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 设备配置
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
else:
    logger.info("Using CPU")

# 参数校验
if args.dp:
    assert args.epsilon > 0, "Epsilon必须大于0"
    assert args.delta > 0, "Delta必须大于0"
    assert args.clip_threshold > 0, "梯度裁剪阈值必须大于0"

if args.diffusion_steps < 100:
    logger.warning("扩散步数建议设置至少100步以获得更好效果")

# 打印配置
logger.info("实验配置参数:")
for arg in vars(args):
    logger.info(f"{arg:20} => {getattr(args, arg)}")