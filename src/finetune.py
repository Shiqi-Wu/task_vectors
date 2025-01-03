import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible
import time
import torch
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head


import src.datasets as datasets


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')
    ft_path = os.path.join(args.save, train_dataset, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    print(args.data_location)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, f'zeroshot.pt')
        model.module.image_encoder.save(model_path)

    loss_history = []
    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)
        
        total_loss = 0.0

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            total_loss += loss.item()
        total_loss /= len(data_loader)
        loss_history.append(total_loss)
    torch.save(loss_history, os.path.join(ckpdir, 'loss_history.pt'))
    print('Finished Training, saved to', ckpdir)


    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')  
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path


if __name__ == '__main__':
    data_location = "/home/shiqi/code/task_vectors/experimental_results/data"
    # models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    models = ['ViT-L-14']

    datasets = ['DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'SVHN']
    # datasets = ['DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'SVHN']
    # datasets = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    """
    Cars: need to download from kaggle and change the codes in src/datasets/cars.py
    RESISC45: cannot get access to the dataset
    SUN397: too large (39GB), link: http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
    SVHN
    """

    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4
    }
    # epochs = {
    #     'Cars': 1,
    #     'DTD': 1,
    #     'EuroSAT': 1,
    #     'GTSRB': 1,
    #     'MNIST': 1,
    #     'RESISC45': 1,
    #     'SUN397': 1,
    #     'SVHN': 1,
    #     'ImageNet': 1
    # }

    for model in models:
        for dataset in datasets:
            print('='*100)
            print(f'Finetuning {model} on {dataset}')
            print('='*100)
            args = parse_arguments()
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.train_dataset = dataset + 'Val'
            args.batch_size = 128
            args.model = model
            args.save = f'checkpoints/{model}'
            finetune(args)
