from torch.utils.tensorboard import SummaryWriter
from sources.cluster_metric import clustering_metric_fc, clustering_metric_hv
import torch
import numpy as np
from tqdm import tqdm

def train_npair(model, loss_fn, optimizer, epoch_n, train_loader, validation_loader, log_file, new_checkpoint_file, last_checkpoint_file=None, device='cpu'):
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
            epoch_fc = 0.
            epoch_hv = 0.
            epoch_hv_max = 0.

            for batch in tepoch:
                ########
                x_super_batch, y_super_batch = batch
                batch_len = len(y_super_batch)
                n_pairs = batch_len // 2
                base = torch.Tensor(np.arange(0, batch_len, 2)).type(torch.int32)
                positive = torch.Tensor(np.arange(1, batch_len, 2)).type(torch.int32)

                positives = positive.repeat(n_pairs, 1)
                positives = positives.flatten()[1:].view(n_pairs-1, n_pairs+1)[:,:-1].reshape(n_pairs, n_pairs-1)
                batches_indexes = torch.cat((torch.unsqueeze(base, 1), torch.unsqueeze(positive, 1), positives), axis=1)
                
                
                batch = []

                for batch_indexes in batches_indexes:
                    x_batch = x_super_batch[batch_indexes]
                    y_batch = y_super_batch[batch_indexes]

                    if not batch:
                        batch = [x_batch, y_batch]
                    else:
                        batch[0] = torch.vstack((batch[0], x_batch))
                        batch[1] = torch.cat((batch[1], y_batch))
                ########

                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                
                x_embed = model(x_batch)

                # calc metrics
                with torch.no_grad():
                    metric_fc = clustering_metric_fc(x_embed, y_batch)
                    metric_hv, metric_hv_max  = clustering_metric_hv(x_embed, y_batch)

                epoch_fc += metric_fc
                epoch_hv += metric_hv
                epoch_hv_max += metric_hv_max

                # print(x_embed.shape)
                x_embed = x_embed.view(n_pairs, n_pairs + 1, 64)

                loss = loss_fn(x_embed[:, 0, :], x_embed[:, 1, :], x_embed[:, 2:, :])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()


                tepoch.set_postfix({'loss': float(loss.item()), 'fc': metric_fc, 'hv': metric_hv, 'hv_max': metric_hv_max})

            # train_acc = epoch_right_pred / epoch_pred
            epoch_loss = float(epoch_loss / len(train_loader))
            epoch_fc = float(epoch_fc / len(train_loader))
            epoch_hv = float(epoch_hv / len(train_loader))
            epoch_hv_max = float(epoch_hv_max / len(train_loader))

            # Calculate loss and error for validation dataset
            model.eval()

            with torch.no_grad():
                batch = next(iter(validation_loader))
                
                ########
                x_super_batch, y_super_batch = batch
                batch_len = len(y_super_batch)
                n_pairs = batch_len // 2
                base = torch.Tensor(np.arange(0, batch_len, 2)).type(torch.int32)
                positive = torch.Tensor(np.arange(1, batch_len, 2)).type(torch.int32)

                positives = positive.repeat(n_pairs, 1)
                positives = positives.flatten()[1:].view(n_pairs-1, n_pairs+1)[:,:-1].reshape(n_pairs, n_pairs-1)
                batches_indexes = torch.cat((torch.unsqueeze(base, 1), torch.unsqueeze(positive, 1), positives), axis=1)
                
                
                batch = []

                for batch_indexes in batches_indexes:
                    x_batch = x_super_batch[batch_indexes]
                    y_batch = y_super_batch[batch_indexes]

                    if not batch:
                        batch = [x_batch, y_batch]
                    else:
                        batch[0] = torch.vstack((batch[0], x_batch))
                        batch[1] = torch.cat((batch[1], y_batch))
                ########

                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                x_embed = model(x_batch)

                metric_fc = clustering_metric_fc(x_embed, y_batch)
                metric_hv, metric_hv_max = clustering_metric_hv(x_embed, y_batch)

                
                x_embed = x_embed.view(n_pairs, n_pairs + 1, 64)

                test_loss = float(loss_fn(x_embed[:, 0, :], x_embed[:, 1, :], x_embed[:, 2:, :]))

                print(f"Test loss: {test_loss}, FC: {metric_fc}, HV: {metric_hv}, HV_MAX: {metric_hv_max}")

                writer.add_scalar('Train Loss', epoch_loss, epoch)
                writer.add_scalar('Test Loss', test_loss, epoch)
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