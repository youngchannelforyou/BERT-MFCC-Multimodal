import argparse
import torch
import logging
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_data_loader
from model import MultimodalTransformer

label = {
        "neutral" : 0,
        "happy" : 1, 
        "angry" : 2, 
        "surprise" : 3, 
        "disqust" : 4, 
        "sad" : 5, 
        "fear" : 6,
}

def evaluate(model,
             data_loader,
             device, test_name, save_path, LABEL_DICT=label):
    loss = 0
    y_true, y_pred = [], []

    model.eval()
    model.zero_grad()
    class_weights  = torch.FloatTensor([0.01,0.16,0.16,0.16,0.17,0.17,0.17]).to(device)
    loss_fct = torch.nn.CrossEntropyLoss()# weight=class_weights )#,3:0.16,4:0.17,5:0.17,6:0.17
    iterator = tqdm(enumerate(data_loader), desc='eval_steps', total=len(data_loader))
    for step, batch in iterator:
        with torch.no_grad():

            # unpack and set inputs
            batch = map(lambda x: x.to(device) if x is not None else x, batch)
            audios, a_mask, texts, t_mask, labels = batch
            labels = labels.squeeze(-1).long()
            y_true += labels.tolist()

            # feed to model and get loss
            logit, hidden = model(audios, texts, a_mask, t_mask)
            cur_loss = loss_fct(logit, labels.view(-1))
            loss += cur_loss.item()
            y_pred += logit.max(dim=1)[1].tolist()

    # evaluate with metrics
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL_DICT))),
        #labels=7,
        target_names=list(LABEL_DICT.keys()),
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    #f1 = report['macro avg']['f1-score']
    #prec = report['macro avg']['precision']
    #rec = report['macro avg']['recall']
    
    f1 = report['weighted avg']['f1-score']
    prec = report['weighted avg']['precision']
    rec = report['weighted avg']['recall']
    loss /= len(data_loader)
    
    #찬영
    try:
        df_result = pd.read_csv(os.path.join(save_path,'result.csv'))
        df_Totalresult = pd.read_csv(os.path.join(save_path,'Totalresult.csv'))
    except FileNotFoundError:
        df_result = pd.DataFrame(columns=['label', 'f1-score', 'precision', 'recall'])
        df_Totalresult = pd.DataFrame(columns=['label', 'f1-score', 'precision', 'recall'])
        
    df_result.loc[len(df_result)] = [ test_name + "TOTAL", f1, prec, rec]
    df_Totalresult.loc[len(df_Totalresult)] = [ test_name + "TOTAL", f1, prec, rec]
    #찬영
        
    # logging
    log_template = "{}\tF1: {:.4f}\tPREC: {:.4f}\tREC: {:.4f}"
    logging.info(log_template.format("TOTAL", f1, prec, rec))
    for key, value in report.items():
        if key in LABEL_DICT:
            cur_f1 = value['f1-score']
            cur_prec = value['precision']
            cur_rec = value['recall']
            df_result.loc[len(df_result)] = [ test_name + key, cur_f1, cur_prec, cur_rec]
            
            logging.info(log_template.format(key, cur_f1, cur_prec, cur_rec))
    logging.info('\n'+str(cm))
    
    df_result.to_csv(os.path.join(save_path, 'result.csv'), index=False)
    df_Totalresult.to_csv(os.path.join(save_path, 'Totalresult.csv'), index=False)
    
    return loss, f1


def main(args):
    data_loader = get_data_loader(
        args=args,
        data_path=args.data_path,
        bert_path=args.bert_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        split=args.split
    )

    model = MultimodalTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=args.n_classes,
        only_audio=args.only_audio,
        only_text=args.only_text,
        d_audio_orig=args.n_mfcc,
        d_text_orig=768,  # BERT hidden size
        d_model=args.d_model,
        attn_mask=args.attn_mask
    ).to(args.device)
    save_point = torch.load(args.model_path)
    model.load_state_dict(save_point, strict=False)

    # evaluation
    logging.info('evaluation starts')
    model.zero_grad()
    evaluate(model, data_loader, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_text', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--bert_path', type=str, default='./KoBERT')
    parser.add_argument('--model_path', type=str, default='./practice/epoch1-loss0.7895-f10.1276.pt')
    parser.add_argument('--n_classes', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)

    # architecture
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=40)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--attn_mask', action='store_false')

    # data processing
    parser.add_argument('--max_len_audio', type=int, default=400)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--resample_rate', type=int, default=16000)
    parser.add_argument('--n_fft_size', type=int, default=600)
    parser.add_argument('--n_mfcc', type=int, default=40)

    args_ = parser.parse_args()

    # -------------------------------------------------------------- #
    
    # check usage of modality
    if args_.only_audio and args_.only_text:
        raise ValueError("Please check your usage of modalities.")

    # seed and device setting
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args_.device = device_

    # log setting
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    main(args_)
