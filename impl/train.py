import torch


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an_var epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], id_var=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on_var validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys_var = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys_var.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys_var, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
