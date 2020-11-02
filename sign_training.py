
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ogb.nodeproppred import Evaluator

from logger import Logger


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, device, loader, evaluator):
    model.eval()

    y_pred, y_true = [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)

        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)

    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']

# run sign
# python3 sign_training.py --device 0 --dropout 0.3 --lr 0.00005 --hidden_channels 512 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign_results.txt

# run sign-xl
# python3 sign_training.py --device 1 --dropout 0.5 --lr 0.00005 --hidden_channels 2048 --num_layers 3 --embeddings_file_name sign_333_embeddings.pt --result_file_name sign-xl_results.txt


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (SIGN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embeddings_file_name', type=str, default='op_dict.pt')
    parser.add_argument('--result_file_name', type=str, default='results.txt')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    try:
        op_dict = torch.load(args.embeddings_file_name)
    except:
        raise RuntimeError('File {} not found. Need to run python preprocessing.py first'.format(args.embeddings_file_name))

    split_idx = op_dict['split_idx']
    x = torch.cat(op_dict['op_embedding'], dim=1)
    y = op_dict['label'].to(torch.long)
    num_classes = 172
    print('Input feature dimension: {}'.format(x.shape[-1]))
    print('Total number of nodes: {}'.format(x.shape[0]))

    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MLP(
        x.size(-1),
        args.hidden_channels,
        num_classes,
        args.num_layers,
        args.dropout).to(device)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: {}.'.format(num_trainable_parameters))
    
    evaluator = Evaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, info=args, file_name=args.result_file_name)

    for run in range(args.runs):
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            
            train(model, device, train_loader, optimizer)
            train_acc = test(model, device, train_loader, evaluator)
            valid_acc = test(model, device, valid_loader, evaluator)
            test_acc = test(model, device, test_loader, evaluator)

            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__=='__main__':
    main()
