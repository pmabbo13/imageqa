import argparse
import json
import numpy as np
import os.path
import requests
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import pad
from transformers import CLIPProcessor, CLIPVisionModel, DistilBertTokenizer, DistilBertModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ImageQADataset(Dataset):
    
    def __init__(self, data, fpaths, img_transform, txt_transform, device):
        
        self.fpaths = fpaths
        self.data = data
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id, question, answer = self.data[idx]
        
        # get image and pass through transform to get features
        fname = self.fpaths[image_id]
        image = Image.open(fname).convert('RGB')
        image = torchvision.transforms.ToTensor()(image)
        assert image.dim() == 3 and image.size(0) == 3, image.shape
        input_img = self.img_transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        input_img = input_img.to(self.device)
        
        # tokenize question
        txt_input = self.txt_transform(question, return_tensors="pt")
        input_ids, input_attn_mask = txt_input.input_ids.squeeze(0), txt_input.attention_mask.squeeze(0)
        
        # pad tokenized question to max length of 63
        input_ids = pad(input_ids, (0,63-input_ids.size(0)), "constant", 0)
        input_attn_mask = pad(input_attn_mask, (0,63-input_attn_mask.size(0)), "constant", 0)
        assert input_ids.size(0) == 63, input_ids.shape
        assert input_attn_mask.size(0) == 63, input_attn_mask.shape
        input_ids = input_ids.to(self.device)
        input_attn_mask = input_attn_mask.to(self.device)
        
        # get id for answer
        label_ids = self.txt_transform(answer, return_tensors="pt").input_ids
        label_ids = label_ids.to(self.device)
        label = label_ids[0,1]
        label = label.to(self.device)
        
        example = {
            "input_img": input_img,
            "input_ids": input_ids,
            "input_attn_mask": input_attn_mask,
            "label": label,
            "image_fname": fname,
            "image_id": image_id,
            "question": question,
            "answer": answer,
        }
        
        return example


class CLIPQA(nn.Module):
    
    def __init__(self, img_model, txt_model, txt_transform, mlp_hidden_dim, freeze_img=True, freeze_txt=False, device='cpu'):
        
        super().__init__()

        self.mlp_hidden_dim = mlp_hidden_dim
        self.txt_transform = txt_transform
        self.img_model = img_model.to(device)
        self.txt_model = txt_model.to(device)
        self.device = device
                
        self.mapping = nn.Sequential(
            nn.Linear(img_model.config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, txt_model.config.dim),
            nn.ReLU()
        )
        self.mapping = self.mapping.to(device)
        
        if freeze_img:
            for p in self.img_model.parameters():
                p.requires_grad = False
        
        if freeze_txt:
            for p in self.txt_model.parameters():
                p.requires_grad = False
                
        self.lm_head = nn.Linear(txt_model.config.dim, txt_model.config.vocab_size, bias=False)
        self.lm_head = self.lm_head.to(device)
 
    def forward(self, input_img, input_ids, attn_mask):
        
        # get image embedding using CLIP model
        clip_output = self.img_model(pixel_values=input_img).last_hidden_state
        
        # pass it through mapping network
        if self.mlp_hidden_dim == 0:
            img_embedding = clip_output
        else:
            img_embedding = self.mapping(clip_output)
        
        # Get text embeddings of question using text model
        txt_embedding = self.txt_model.embeddings(input_ids)
        
        # conctenate image and text embeddings
        input_embeds = torch.hstack([txt_embedding, img_embedding])
        extra_attn = torch.ones(img_embedding.shape[:-1])
        extra_attn = extra_attn.to(self.device)
        input_attn_mask = torch.hstack([attn_mask, extra_attn])
        
        # pass embeddings through to language model
        txt_model_inputs = {'inputs_embeds': input_embeds, 'attention_mask': input_attn_mask}
        txt_output = self.txt_model(**txt_model_inputs)
        
        logits = self.lm_head(txt_output.last_hidden_state)
        
        return logits

def make_data(images, questions, answers, txt_transform):
    data = []
    removed = 0
    for i,q,a in zip(images, questions, answers):
        if txt_transform(a, return_tensors="pt").input_ids.shape == (1,3):
            data.append((i,q,a))
        else:
            removed += 1
    print(f"Removed {removed} examples from set")
    return data


def get_fpaths(data_dir='data/annotations'):
    fpaths = {}
    with open(f'{data_dir}/captions_train2014.json') as json_file:
        data = json.load(json_file)
        for img in data['images']:
            img_id = str(img["id"])
            prefix = '0' * (12 - len(img_id))
            fname = f"data/train2014/COCO_train2014_{prefix}{img_id}.jpg"
            assert os.path.isfile(fname), fname
            fpaths[img_id] = fname
            
    with open(f'{data_dir}/captions_val2014.json') as json_file:
        data = json.load(json_file)
        
        for img in data['images']:
            img_id = str(img["id"])
            prefix = '0' * (12 - len(img_id))
            fname = f"data/val2014/COCO_val2014_{prefix}{img_id}.jpg"
            assert os.path.isfile(fname), fname
            fpaths[img_id] = fname
    
    return fpaths


def get_split(split, cocoqa_dir, txt_transform):
    
    with open(f'{cocoqa_dir}/{split}/img_ids.txt') as f:
        images = f.read().splitlines()
        
    with open(f'{cocoqa_dir}/{split}/questions.txt') as f:
        questions = f.read().splitlines()
        
    with open(f'{cocoqa_dir}/{split}/answers.txt') as f:
        answers = f.read().splitlines()
        
    assert len(images) == len(questions) == len(answers)
    
    return make_data(images, questions, answers, txt_transform)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fxn, epochs, eval_steps):    
    # initialize lists to keep track of validation loss and accuracy at each iteration
    training_loss = []
    validation_loss = []
    validation_acc = []

    # initialize dictionary to keep track of best model based on performance on validation set
    best_model = {
        'state_dict': None,
        'loss': float("inf"),
        'acc': 0,
        'iter': 0
    }
    
    steps = 0
    for epoch in range(epochs):
        # mini-batch training
        for batch in tqdm(iter(train_dataloader), desc=f"Training Epoch: {epoch+1}"):
        #for batch in iter(train_dataloader):

            # extract features and labels from batch
            input_img = batch['input_img']
            input_ids = batch['input_ids']
            attn_mask = batch['input_attn_mask']
            labels = batch['label']

            # zero out the current gradients of parameters so that fresh gradietns computed
            optimizer.zero_grad()

            # perform forward pass to get logits
            output = model(input_img, input_ids, attn_mask)
            logits = output[:,-1]
            
            # compute the loss using the model output and true labels
            L = loss_fxn(logits, labels)
            training_loss.append(L.item())
            
            # perform backward pass to compute gradients of loss w.r.t model weights
            L.backward()

            # update model weights based on gradients
            optimizer.step()
            
            if steps % eval_steps == 0:
                # compute loss and accuracy on validation set
                val_loss, val_acc  = val_loss_acc(model, val_dataloader, loss_fxn)
                validation_loss.append(val_loss)
                validation_acc.append(val_acc)
                
                # update best model
                if val_loss <= best_model['loss']:
                    best_model['state_dict'] = model.state_dict()
                    best_model['loss'] = val_loss
                    best_model['acc'] = val_acc
                    best_model['iter'] = steps
                
                print(f"Validation loss at iteration {steps} is {val_loss}")
                print(f"Validation accuracy at iteration {steps} is {(val_acc * 100):2f}%")
            
            steps += 1
        
        if (steps-1) % eval_steps != 0:
            # compute loss and accuracy on validation set
            val_loss, val_acc  = val_loss_acc(model, val_dataloader, loss_fxn)
            validation_loss.append(val_loss)
            validation_acc.append(val_acc)

            # update best model
            if val_loss <= best_model['loss']:
                best_model['state_dict'] = model.state_dict()
                best_model['loss'] = val_loss
                best_model['acc'] = val_acc
                best_model['iter'] = steps
            
            print(f"Validation loss at iteration {steps}: {val_loss}")
            print(f"Validation accuracy at iteration {steps}: {(val_acc * 100):.2f}%")

    
    # plot traiing loss and validation loss & accuracy
    plot_iters(training_loss, title="Training Loss", fname=f"results/{args.model_name}_training_loss")
    plot_iters(validation_loss, title="Validation Loss", fname=f"results/{args.model_name}_validation_loss")

    print(f'Start training loss: {training_loss[0]}')
    print(f'Final training loss: {training_loss[-1]}')

    # return info on best model
    return best_model


def val_loss_acc(model, dataloader, loss_fxn):
    
    with torch.no_grad():

        loss = 0
        correct = 0
        total = 0
        #for batch in tqdm(iter(dataloader), desc="Computing validation loss"):
        for batch in iter(dataloader):
            # extract features and labels from batch
            input_img = batch['input_img']
            input_ids = batch['input_ids']
            attn_mask = batch['input_attn_mask']
            labels = batch['label']

            # perform forward pass to get logits
            output = model(input_img, input_ids, attn_mask)
            logits = output[:,-1]
            
            # compute the loss using the model output and true labels
            L = loss_fxn(logits, labels)

            # compute loss
            loss += L.item()
            
            # get predictions
            preds = torch.argmax(logits, 1)
            
            assert preds.shape == labels.shape
            correct += (preds==labels).sum().item()
            total += labels.size(0)

        return loss, correct/total


def plot_iters(iter_data, title, fname):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(iter_data)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    plt.show()
    plt.savefig(f'{fname}.png')

def accuracy(model, dataloader):
    
    with torch.no_grad():

        correct = 0
        total = 0
        i = 0
        #for batch in tqdm(iter(dataloader), desc="Computing validation loss"):
        for batch in iter(dataloader):
            # extract features and labels from batch
            input_img = batch['input_img']
            input_ids = batch['input_ids']
            attn_mask = batch['input_attn_mask']
            labels = batch['label']

            # perform forward pass to get logits
            output = model(input_img, input_ids, attn_mask)
            logits = output[:,-1]
            
            # get predictions
            preds = torch.argmax(logits, 1)
            
            assert preds.shape == labels.shape
            correct += (preds==labels).sum().item()
            total += labels.size(0)
            
            i += 1
            if i == 10:
                break

        return correct/total

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Model name to save checkpoint',
                        type=str)
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        default=8, help='The batch size to be used for eval',
                        type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', required=False,
                        default=1e-4, help='The training learning rate',
                        type=float)
    parser.add_argument('--hidden_dim', dest='hidden_dim', required=False,
                        default=1024, help='Size of hidden layer for mapping network',
                        type=int)
    parser.add_argument('--freeze_txt', dest='freeze_txt', required=False,
                        default=False, help='Freeze weights for text model',
                        type=bool)
    parser.add_argument('--freeze_img', dest='freeze_img', required=False,
                        default=True, help='Freeze weights for vision model',
                        type=bool)
    parser.add_argument('--eval_steps', dest='eval_steps', required=False,
                        default=200, help='Number of training iterations until evaluated on dev. set',
                        type=int)
    parser.add_argument('--epochs', dest='epochs', required=False,
                        default=1, help='Number of training epochs',
                        type=int)
    args = parser.parse_args()

    # get CLIP model
    img_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    img_transform = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # get text model
    txt_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    txt_transform = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # get image filepaths
    fpaths = get_fpaths('data/annotations')

    # get training split
    train_data = get_split('train', 'data/cocoqa', txt_transform)

    # get test split
    test_data = get_split('test', 'data/cocoqa', txt_transform)

    # create dev data split using 10% of train data
    dev_len = len(train_data) // 10
    train_len = len(train_data) - dev_len
    train_data, dev_data = random_split(
                                    train_data,
                                    [train_len, dev_len],
                                    generator=torch.Generator().manual_seed(42)
                                )

    # create datasets
    train_dataset = ImageQADataset(train_data, fpaths, img_transform, txt_transform, DEVICE)
    dev_dataset = ImageQADataset(dev_data, fpaths, img_transform, txt_transform, DEVICE)
    test_dataset = ImageQADataset(test_data, fpaths, img_transform, txt_transform, DEVICE)

    # put data into dataloaders. This will allow us to efficiently access the data in batches
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # initialize model, optimizer, and loss function
    model = CLIPQA(img_model, txt_model, txt_transform, args.hidden_dim, args.freeze_img, args.freeze_txt, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fxn = nn.CrossEntropyLoss()

    # train model
    model = model.train()
    print(f"System device: {DEVICE}")
    best_model = train(model, train_dataloader, dev_dataloader, optimizer, loss_fxn, args.epochs, args.eval_steps)

    # set model weights to best found during training
    print(f"Best model found at iteration: {best_model['iter']}")
    print(f"\tValidation Loss: {best_model['loss']}")
    print(f"\tValidation Accuracy: {(best_model['acc'] * 100):.2f}%")
    model.load_state_dict(best_model['state_dict'])

    # get test accuracy
    model = model.eval()
    model = model.to(DEVICE)
    test_accuracy = accuracy(model, test_dataloader)
    print(f"The accuracy on the test set is: {(test_accuracy * 100):.2f}%")

    # save best model
    torch.save(model.state_dict(), f'checkpoints/{args.model_name}.pt')