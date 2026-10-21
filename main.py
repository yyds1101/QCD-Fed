from dataset import *
from utils import set_seed
from tqdm import tqdm
import numpy as np
import torch

# 根据算法选择导入对应的模块
if args.algorithm == 'local':
    from clients.local import Client
elif args.algorithm == 'fedavg':
    from clients.fedavg import Client
    from servers.fedavg import Server
elif args.algorithm == 'fedgen':  # 扩散模型专用
    from clients.fedgen import Client
    from servers.fedgen import Server
elif args.algorithm == 'fedprox':
    from clients.fedprox import Client
    from servers.fedprox import Server

def main():
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 创建数据集
    dataset_map = {
        "fmnist": FMNIST,
        "cifar": Cifar,
        "digit": Digit,
        "office": Office,
        "domainnet": Domainnet
    }
    dataset = dataset_map[args.dataset]()

    # 数据划分
    if args.algorithm == 'feddf':
        trainsets, valsets, testsets, server_dataset = dataset.data_partition()
    else:
        trainsets, valsets, testsets = dataset.data_partition()

    # 创建客户端
    clients = [Client(i, trainsets[i], valsets[i], testsets[i])
              for i in range(args.n_clients)]

    # 创建服务器（本地训练不需要）
    server = None
    if args.algorithm != "local":
        server = Server(clients)

    # 初始化最佳指标
    best_metrics = {
        'val_acc': np.zeros(args.n_clients),
        'test_acc': np.zeros(args.n_clients),
        'round': -np.ones(args.n_clients)
    }

    # 训练循环
    with tqdm(total=args.global_epoch, desc='Federated Training',
             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:

        for current_round in range(args.global_epoch):
            # 客户端本地训练
            train_metrics = {'cls_loss': [], 'diff_loss': []}  # 修改指标名称
            for client in clients:
                metrics = client.local_train(current_round)
                train_metrics['cls_loss'].append(metrics['classifier_loss'])
                train_metrics['diff_loss'].append(metrics['diffusion_loss'])  # 改为扩散损失

            # 服务器通信
            if server is not None:
                if args.algorithm == 'feddiff':
                    # 扩散模型专用通信流程
                    server.receive(["diffusion", "classifier"])  # 接收扩散模型参数
                    server.global_train()  # 全局扩散模型训练
                    server.send(["diffusion", "classifier"])  # 分发参数
                else:
                    # 常规联邦算法
                    server.aggregate(["extractor", "classifier"])
                    server.distribute(["extractor", "classifier"])

            # 验证测试
            current_accs = []
            for idx, client in enumerate(clients):
                val_acc = client.local_val()
                test_acc = client.local_test()

                if val_acc > best_metrics['val_acc'][idx]:
                    best_metrics['val_acc'][idx] = val_acc
                    best_metrics['test_acc'][idx] = test_acc
                    best_metrics['round'][idx] = current_round

                current_accs.append(test_acc)

            # 更新进度条
            avg_cls_loss = np.mean(train_metrics['cls_loss'])
            avg_diff_loss = np.mean(train_metrics['diff_loss'])  # 修改指标名称
            avg_acc = np.mean(current_accs)

            pbar.set_postfix_str(
                f'Round {current_round + 1:03d} | '
                f'Cls Loss: {avg_cls_loss:.4f} | '  # 分类损失
                f'Diff Loss: {avg_diff_loss:.4f} | '  # 扩散损失
                f'Acc: {avg_acc:.2%} | '
                f'Best: {np.max(best_metrics["test_acc"]):.2%}'
            )
            pbar.update(1)

            # 早停判断
            if (current_round - np.max(best_metrics['round'])) > args.early_stop_rounds:
                print(f"\nEarly stopping at round {current_round + 1}")
                break

    # 输出最终结果
    final_acc = np.mean(best_metrics['test_acc'])
    print(f"\n{'=' * 50}")
    print(f"Training Completed (Total Rounds: {current_round + 1})")
    print(f"Average Test Accuracy: {final_acc:.2%}")
    print(f"Client Best Rounds: {best_metrics['round'].astype(int)}")
    print(
        f"Client Best Accuracies: {np.array2string(best_metrics['test_acc'], formatter={'float_kind': lambda x: f'{x:.2%}'})}")

if __name__ == "__main__":
    main()