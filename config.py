import argparse

parser = argparse.ArgumentParser()

# training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--n_fold', type=int, default=1)
parser.add_argument('--pk_params', type=str, default=['auc', 'cmax', 'tmax', 'hl', 'cl', 'vz', 'mrt', 'f'], nargs='+')

parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--lr_patience', type=int, default=7)
parser.add_argument('--es_patience', type=int, default=10)
parser.add_argument('--delta', type=float, default=1e-3)
parser.add_argument('--model_path', type=str, default='checkpoints/model.pth')
parser.add_argument('--results_path', type=str, default='results/results.csv')
parser.add_argument('--checkpoints_folder', type=str, default='mmpk')

# model
parser.add_argument('--num_layers_mol', type=int, default=5)
parser.add_argument('--d_mol', type=int, default=300)
parser.add_argument('--dropout_mol', type=float, default=0.1)
parser.add_argument('--gnn_type_mol', type=str, default='gat', choices=['gin', 'gcn', 'gat'])
parser.add_argument('--num_layers_sub', type=int, default=5)
parser.add_argument('--d_sub', type=int, default=300)
parser.add_argument('--dropout_sub', type=float, default=0.1)
parser.add_argument('--gnn_type_sub', type=str, default='gin', choices=['gin', 'gcn', 'gat'])
parser.add_argument('--d_model', type=int, default=600)
parser.add_argument('--num_heads_gat', type=int, default=1)
parser.add_argument('--JK', type=str, default='last', choices=['concat', 'last', 'max', 'sum'])
parser.add_argument('--graph_pooling', type=str, default='mean', choices=['sum', 'mean', 'max', 'attention'])

# predict
parser.add_argument('-p', '--output_file_pred', type=str, default='prediction.csv')
parser.add_argument('-a', '--output_file_att', type=str, default='attention.csv')

args = parser.parse_args()

if isinstance(args.pk_params, str):
    args.pk_params = [args.pk_params]

if __name__ == '__main__':
    pass
