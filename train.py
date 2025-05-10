import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from config import args
from dataloader import DataLoaderSelector
from mmpk import MMPKPredictor
from utils import *

def train(model, loader, optimizer, criterion, device, num_tasks):
    model.train()

    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, label, dose = [item.to(device) for item in batch]
        
        optimizer.zero_grad()

        y_hat, _ = model(mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose)
        y = label.view(y_hat.shape)  # (batch_size, num_tasks)

        batch_loss = []
        for task_idx in range(num_tasks):
            y_hat_task = y_hat[:, task_idx]
            y_task = y[:, task_idx]

            valid_mask = ~torch.isnan(y_task)
            # Skip task if all labels are NaN
            if valid_mask.sum() == 0:
                continue
            
            y_hat_task = y_hat_task[valid_mask]
            y_task = y_task[valid_mask]

            task_loss = criterion(y_hat_task, y_task)
            batch_loss.append(task_loss)
        
        batch_loss = sum(batch_loss) / len(batch_loss)
        
        # Skip batch if all tasks are skipped
        if batch_loss > 1e-9:
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.detach().cpu().item()

    avg_loss = total_loss / len(loader)

    return avg_loss

def test(model, loader, criterion, device, num_tasks):
    model.eval()

    total_loss = 0.0
    y_hat_tasks = [[] for _ in range(num_tasks)]
    y_tasks = [[] for _ in range(num_tasks)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, label, dose = [item.to(device) for item in batch]

            y_hat, _ = model(mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose)
            y = label.view(y_hat.shape)  # (batch_size, num_tasks)

            batch_loss = []
            for task_idx in range(num_tasks):
                y_hat_task = y_hat[:, task_idx]
                y_task = y[:, task_idx]

                valid_mask = ~torch.isnan(y_task)
                # Skip task if all labels are NaN
                if valid_mask.sum() == 0:
                    continue
                
                y_hat_task = y_hat_task[valid_mask]
                y_task = y_task[valid_mask]

                task_loss = criterion(y_hat_task, y_task)
                batch_loss.append(task_loss)
            
                y_hat_tasks[task_idx].extend(y_hat_task.detach().cpu().numpy())
                y_tasks[task_idx].extend(y_task.detach().cpu().numpy())
            
            batch_loss = sum(batch_loss) / len(batch_loss)
            total_loss += batch_loss.detach().cpu().item()

    avg_loss = total_loss / len(loader)
    gmfe_scores = []
    rmsle_scores = []
    afe_scores = []
    pearson_r_scores = []
    for task_idx in range(num_tasks):
        gmfe_scores.append(gmfe(y_tasks[task_idx], y_hat_tasks[task_idx], args.pk_params[task_idx]))
        rmsle_scores.append(rmsle(y_tasks[task_idx], y_hat_tasks[task_idx]))
        afe_scores.append(afe(y_tasks[task_idx], y_hat_tasks[task_idx], args.pk_params[task_idx]))
        pearson_r_scores.append(pearson_r(y_tasks[task_idx], y_hat_tasks[task_idx]))

    metrics = {
        'gmfe': gmfe_scores,
        'rmsle': rmsle_scores,
        'afe': afe_scores,
        'pearson_r': pearson_r_scores,
    }

    return avg_loss, metrics

if __name__ == '__main__':
    seed_everything(args.seed)
    device = torch.device(
        'cuda:' + str(args.device)
        if torch.cuda.is_available() else 'cpu'
    )
    num_tasks = len(args.pk_params)

    selector = DataLoaderSelector(args.n_fold, args.pk_params, load_ext=False)
    train_loader, val_loader, test_loader = selector.select_loader(loader_type='mmpk')

    model = MMPKPredictor(args, num_tasks=num_tasks)
    model.to(device)
    print(model)

    criterion = nn.MSELoss(reduction='mean')   
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                     patience=args.lr_patience, verbose=True)
    
    pbar = trange(args.num_epochs, desc='Training')
    early_stopping = EarlyStopping(args.es_patience, args.delta, args.model_path,
                                   trace_func=lambda msg: pbar.set_postfix_str(msg))
    
    for epoch in pbar:
        train_loss = train(model, train_loader, optimizer, criterion, device, num_tasks)
        val_loss, val_metrics = test(model, val_loader, criterion, device, num_tasks)
        _, test_metrics = test(model, test_loader, criterion, device, num_tasks)

        pbar.set_description(
            f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] < 1e-6:
            print('Lr reached the minimum threshold of 1e-6. Stopping training.')
            break

        early_stopping(val_loss, model, epoch)

        if early_stopping.early_stop:
            print(f"Early stopped at epoch {epoch}")
            break
    
    print('Loading the best model for evaluation on test set...')
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    _, test_metrics = test(model, test_loader, criterion, device, num_tasks)

    print(f"Saving test set results to {args.results_path}...")
    save_results(args.results_path, args.n_fold, args.pk_params,
                 test_metrics, early_stopping.best_epoch)
