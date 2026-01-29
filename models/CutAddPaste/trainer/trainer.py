import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import *
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
# from torch.utils.tensorboard import SummaryWriter
from .early_stopping import EarlyStopping

sys.path.append("../../")
def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx):
    # Start training
    print("Training started ....")
    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx, patience=300)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss = model_train(model, model_optimizer, train_dl, config, device, epoch)
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device, epoch)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device, epoch)
        scheduler.step(train_loss)
        if epoch % 1 == 0:
            print(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'Valid Loss     : {val_loss:.4f}\t  | \n'
                     f'Test Loss     : {test_loss:.4f}\t  | \n'
                    )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(test_loss.item())
        if config.dataset == 'UCR':
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            indicator = test_score.f1(ScoreType.RevisedPointAdjusted)
            early_stopping(score_reasonable, test_affiliation, test_score, indicator, val_score_origin,
                           test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        elif config.dataset == 'SWaT' or config.dataset == 'WADI':
            early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score_reasonable
        # The UCR validation set has no anomaly, so it does not print.
        test_score_origin = early_stopping.best_predict2
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)

    elif config.dataset == 'SWaT' or config.dataset == 'WADI':
        val_score_origin = early_stopping.best_predict1
        test_score_origin = early_stopping.best_predict2
        print('best loss: {:.4f}'.format(early_stopping.best_indicator))
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    test_rpa_f1 = test_rpa_score.f1(ScoreType.RevisedPointAdjusted)
    test_rpa_precision = test_rpa_score.precision(ScoreType.RevisedPointAdjusted)
    test_rpa_recall = test_rpa_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test RAP F1")
    print(f'Test F1: {test_rpa_f1:2.4f}  | \tTest precision: {test_rpa_precision:2.4f}  | \tTest recall: {test_rpa_recall:2.4f}\n')

    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    test_pa_precision = test_pa_score.precision(ScoreType.PointAdjusted)
    test_pa_recall = test_pa_score.recall(ScoreType.PointAdjusted)
    print("Test PA F1")
    print(
        f'Test F1: {test_pa_f1:2.4f}  | \tTest precision: {test_pa_precision:2.4f}  | \tTest recall: {test_pa_recall:2.4f}\n')

    test_pw_f1 = test_pw_score.f1(ScoreType.PointAdjusted)
    test_pw_precision = test_pw_score.precision(ScoreType.PointAdjusted)
    test_pw_recall = test_pw_score.recall(ScoreType.PointAdjusted)
    print("Test PW F1")
    print(
        f'Test F1: {test_pw_f1:2.4f}  | \tTest precision: {test_pw_precision:2.4f}  | \tTest recall: {test_pw_recall:2.4f}\n')

    # writer = SummaryWriter()
    # for i in range(config.num_epoch):
    #     writer.add_scalars('loss', {'train': all_epoch_train_loss[i],
    #                                 'test': all_epoch_test_loss[i]}, i)
    # # writer.add_embedding(part_embedding_feature, metadata=part_embedding_target, tag='test embedding')
    # writer.close()

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict


def model_train(model, model_optimizer, train_loader, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.long().to(device)
        # optimizer
        model_optimizer.zero_grad()
        logits = model(data)    
        loss, score = train(logits, target, config) 
        # Update hypersphere radius R on mini-batch distances
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss




def Trainer_DAL(model, model_optimizer, train_in_loader, train_out_loader, val_dl, test_dl, device, config, idx):
    """DAL-style training loop adapted for CutAddPaste data pipeline.

    Mirrors the original `Trainer` behavior for validation/test at each epoch
    and early stopping. The training per-epoch uses a DAL inner/outer procedure
    but reporting, scheduler stepping, and early-stopping follow the same
    logic as `Trainer` so outputs and saved-best selection stay consistent.
    """
    print("DAL Training started ....")
    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx, patience=300)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []

    for epoch in range(1, config.num_epoch + 1):
        train_target, train_score, train_loss = model_train_dal(model, model_optimizer,
                                                               train_in_loader, train_out_loader,
                                                               config, device, epoch)
        all_epoch_train_loss.append(train_loss.item())

        # Evaluate on validation and test each epoch (match Trainer behavior)
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device, epoch=epoch)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device, epoch=epoch)
        all_epoch_test_loss.append(test_loss.item())

        # Match original Trainer: step scheduler with train loss
        try:
            scheduler.step(train_loss)
        except Exception:
            # fallback to stepping on scalar
            scheduler.step(float(train_loss))

        if epoch % 1 == 0:
            print(f'\nEpoch : {epoch}\n'
                  f'Train Loss     : {train_loss:.4f}\t | \n'
                  f'Valid Loss     : {val_loss:.4f}\t  | \n'
                  f'Test Loss     : {test_loss:.4f}\t  | \n')

        # Early stopping logic mirroring Trainer
        if config.dataset == 'UCR':
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            indicator = test_score.f1(ScoreType.RevisedPointAdjusted)
            early_stopping(score_reasonable, test_affiliation, test_score, indicator, val_score_origin,
                           test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        elif config.dataset == 'SWaT' or config.dataset == 'WADI':
            early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Training finished - mirror Trainer's final selection and reporting
    print("\n################## Training is Done! #########################")
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score_reasonable
        test_score_origin = early_stopping.best_predict2
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)

    elif config.dataset == 'SWaT' or config.dataset == 'WADI':
        val_score_origin = early_stopping.best_predict1
        test_score_origin = early_stopping.best_predict2
        print('best loss: {:.4f}'.format(early_stopping.best_indicator))
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')

    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')

    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    test_rpa_f1 = test_rpa_score.f1(ScoreType.RevisedPointAdjusted)
    test_rpa_precision = test_rpa_score.precision(ScoreType.RevisedPointAdjusted)
    test_rpa_recall = test_rpa_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test RAP F1")
    print(f'Test F1: {test_rpa_f1:2.4f}  | \tTest precision: {test_rpa_precision:2.4f}  | \tTest recall: {test_rpa_recall:2.4f}\n')

    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    test_pa_precision = test_pa_score.precision(ScoreType.PointAdjusted)
    test_pa_recall = test_pa_score.recall(ScoreType.PointAdjusted)
    print("Test PA F1")
    print(
        f'Test F1: {test_pa_f1:2.4f}  | \tTest precision: {test_pa_precision:2.4f}  | \tTest recall: {test_pa_recall:2.4f}\n')

    test_pw_f1 = test_pw_score.f1(ScoreType.PointAdjusted)
    test_pw_precision = test_pw_score.precision(ScoreType.PointAdjusted)
    test_pw_recall = test_pw_score.recall(ScoreType.PointAdjusted)
    print("Test PW F1")
    print(
        f'Test F1: {test_pw_f1:2.4f}  | \tTest precision: {test_pw_precision:2.4f}  | \tTest recall: {test_pw_recall:2.4f}\n')

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict




def model_train_dal(model, model_optimizer, train_in_loader, train_out_loader, config, device, epoch):
    """DAL variant of model_train. Mirrors signature and return values of
    `model_train` but implements the DAL inner/outer batch loop over
    `(train_in_loader, train_out_loader)` pairs.
    Returns: (all_target, all_predict, total_loss)
    """
    total_loss = []
    all_target, all_predict = [], []

    inner_iter = config.dal_inner_iter
    gamma = config.dal_gamma
    beta = config.dal_gamma_beta
    rho = config.dal_rho
    strength = config.dal_strength
    warmup = config.dal_warmup
    emb_init_scale = config.dal_emb_init_scale
    dal_alpha = config.dal_alpha   

    model.train()
    for batch_idx, (in_set, out_set) in enumerate(zip(train_in_loader, train_out_loader)):
        in_data, in_target = in_set[0].float().to(device), in_set[1].long().to(device)  
        out_data, out_target = out_set[0].float().to(device), out_set[1].long().to(device)  

        data = torch.cat((in_data, out_data), 0)    

        model_optimizer.zero_grad()
        x, emb = model.pred_emb(data)

        # in-distribution loss
        l_ce, score_in = train(x[:len(in_data)], in_target, config)
        l_ad, score_ad = train(x[len(in_data):], out_target, config)

        # prepare embeddings for inner-loop OE perturbation
        emb_oe = emb[len(in_data):].detach()
        emb_bias = emb_init_scale * torch.randn_like(emb_oe, device=device)

        for _ in range(inner_iter):
            emb_bias.requires_grad_()
            x_aug = model.fc(emb_bias + emb_oe)
            inner_targets = out_target[:x_aug.size(0)].long().to(device)
            # l_sur_oe = F.cross_entropy(x_aug, inner_targets)
            l_sur_oe = - (x_aug.mean(dim=1) - torch.logsumexp(x_aug, dim=1)).mean()
            r_sur = (emb_bias.abs()).mean(-1).mean()
            l_sur = l_sur_oe - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [emb_bias])[0]
            denom = (grads ** 2).sum(-1).sqrt().unsqueeze(1)
            # Stable normalization: avoid divide-by-zero / extremely small norms
            # Compute per-sample norm and clamp to a small epsilon before division.
            eps = 1e-12
            # clamp to avoid divide-by-zero or extremely small denominators
            denom = denom.clamp(min=eps)
            grads = grads / denom
            emb_bias = emb_bias.detach() + strength * grads.detach()

        if isinstance(r_sur, torch.Tensor):
            r_sur_val = float(r_sur.detach().cpu().item())
        else:
            r_sur_val = float(r_sur)
        gamma = gamma - beta * (rho - r_sur_val)
        # gamma is a Python float here; clamp using builtins to avoid
        # calling Tensor.clamp on a float. Use the configured dal_gamma_max.
        gamma = max(0.0, min(float(gamma), float(config.dal_gamma)))

        if epoch >= warmup:
            x_oe = model.fc(emb[len(in_data):] + emb_bias)
            dal_alpha = config.dal_alpha
        else:
            x_oe = model.fc(emb[len(in_data):])
            dal_alpha = 0

        l_oe = - (x_oe.mean(dim=1) - torch.logsumexp(x_oe, dim=1)).mean()
        m = torch.nn.Softmax(dim=1)
        p = m(x_oe)
        score_out = p[:, 1]        
        # score_out = F.softmax(x_oe, dim=1).max(dim=1).values     
        loss = ((1.0/(1.0+config.rate)) * l_ce + (config.rate/(1.0+config.rate)) * l_ad) + dal_alpha * (config.rate/(1.0+config.rate)) * l_oe
        # loss = l_ce + dal_alpha * l_oe + l_ad     # use this for wadi
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model_optimizer.step()

        total_loss.append(loss.detach().cpu())

        # collect scores/targets for both in and out for consistency with model_train
        all_predict.extend(score_in.detach().cpu().numpy().tolist())
        all_predict.extend(score_out.detach().cpu().numpy().tolist())
        all_target.extend(in_target.detach().cpu().numpy().tolist())
        all_target.extend(out_target.detach().cpu().numpy().tolist())

    if len(total_loss) > 0:
        total_loss = torch.stack(total_loss).mean()
    else:
        total_loss = torch.tensor(0.0)

    return all_target, all_predict, total_loss


def model_evaluate(model, test_dl, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            logits = model(data)
            loss, score = train(logits, target, config)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss


def train(logits, target, config):
    # normalize feature vectors
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(logits, target)
    m = torch.nn.Softmax(dim=1)
    p = m(logits)
    score = p[:, 1]
    return loss, score








