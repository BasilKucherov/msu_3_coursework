from torch.utils.tensorboard import SummaryWriter
from sources.cluster_metric import clustering_metric_fc, clustering_metric_hv
import torch
from tqdm import tqdm
import copy

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def train(model, loss_fn, optimizer, epoch_n, train_loader, validation_loader, log_file, new_checkpoint_file, last_checkpoint_file=None, device='cpu'):
    writer = SummaryWriter('logs/' + log_file)
    
    cur_epoch = 0
    if last_checkpoint_file != None:
        checkpoint = torch.load(last_checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']

    for epoch in range(cur_epoch, cur_epoch + epoch_n):
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            model.train()

            epoch_loss = 0.
            epoch_right_pred = 0
            epoch_pred_number = 0
            epoch_fc = 0.
            epoch_hv = 0.
            epoch_hv_max = 0.

            for batch in tepoch:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                
                y_pred = model(x_batch)
                
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                preds_class = y_pred.argmax(dim=1)

                epoch_loss += loss.item()
                epoch_right_pred += torch.sum(preds_class == y_batch)
                epoch_pred_number += len(y_batch)

                with torch.no_grad():
                    model_embed = copy.deepcopy(model)
                    model_embed.fc = Identity()

                    x_embed = model_embed(x_batch)

                    metric_fc = clustering_metric_fc(x_embed, y_batch)
                    metric_hv, metric_hv_max  = clustering_metric_hv(x_embed, y_batch)

                epoch_fc += metric_fc
                epoch_hv += metric_hv
                epoch_hv_max += metric_hv_max

                tepoch.set_postfix({'loss': float(loss.item()), 'acc': 100. *  torch.sum(preds_class == y_batch) / len(y_batch), 'fc': metric_fc, 'hv': metric_hv, 'hv_max': metric_hv_max})


            epoch_loss = float(epoch_loss / len(train_loader))
            epoch_acc = float(epoch_right_pred / epoch_pred_number)
            epoch_fc = float(epoch_fc / len(train_loader))
            epoch_hv = float(epoch_hv / len(train_loader))
            epoch_hv_max = float(epoch_hv_max / len(train_loader))

            # Calculate loss and error for validation dataset
            model.eval()

            with torch.no_grad():
                batch = next(iter(validation_loader))
                x_batch, y_batch = batch

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)

                test_loss = float(loss_fn(y_pred, y_batch))
                preds_class = y_pred.argmax(dim=1)

                test_acc = float(torch.sum(preds_class == y_batch) / len(y_batch))

                model_embed = copy.deepcopy(model)
                model_embed.fc = Identity()

                x_embed = model_embed(x_batch)

                metric_fc = clustering_metric_fc(x_embed, y_batch)
                metric_hv, metric_hv_max = clustering_metric_hv(x_embed, y_batch)
                print(f"Test loss: {test_loss}, acc: {test_acc}, FC: {metric_fc}, HV: {metric_hv}, HV_MAX: {metric_hv_max}")

                writer.add_scalar('Train Loss', epoch_loss, epoch)
                writer.add_scalar('Train Accuracy', epoch_acc, epoch)
                writer.add_scalar('Test Loss', test_loss, epoch)
                writer.add_scalar('Test Accuracy', test_acc, epoch)
                writer.add_scalar('Train FC', epoch_fc, epoch)
                writer.add_scalar('Train HV', epoch_hv, epoch)
                writer.add_scalar('Train HV max', epoch_hv_max, epoch)
                writer.add_scalar('Test FC', metric_fc, epoch)
                writer.add_scalar('Test HV', metric_hv, epoch)
                writer.add_scalar('Test HV max', metric_hv_max, epoch)
        
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f"{new_checkpoint_file}_{epoch}")
    
    return model
