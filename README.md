# imageqa

This work leverages pre-trained CLIP and DistilBERT models to train a network that performs image-based question answering on the Toronto COCOQA dataset. Image embeddings are derived from a pre-trained CLIP vision encoder and concatenated to the word embedding sequence of a question. A DistilBERT language model uses the resulting representation as input to generate an answer to the image-based question. Our results show significant improvements in test accuracy when the CLIP embeddings are first passed through a small MLP network to obtain the final image embeddings. We achieve further improvements in the test accuracy after fine-tuning the DistilBERT model for the task.

### Data
Toronto COCO-QA datset can be found [here](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/).

Images are sourced from Microsoft's 2014 COCO dataset, which can be found [here](https://cocodataset.org/#download).

### Data Processing & Training

`clipqa_bert.py` takes care of preprocessing the raw data (expected in a /data folder), training the architecture based on provided parameters (shown below) and evaluating performance on the test set.
  
    --model_name: model naem used to save model checkpoint
    --checkpoint: model checkpoint to initalize model with
    --batch_size: training and eval batch size
    --learning_rate: training learning rate
    --epochs: number of training epochs
    --eval_steps: number of training iterations until evaluated on validation set
    --num_layers: number of linear layers for intermediate MLP mapping network
    --hidden_dim: number of hidden dimensions in MLP network
    --freeze_txt: whether or not language model weights should be frozen during training
    
An example command to train a model with 2 linear layers in the mapping network is shown below:
    
    python clipqa_bert.py --model clipqa_mlp2_e1 --batch_size 8 --learning_rate 1e-4 --epochs 1 --eval_steps 400 --num_layers 2 --hidden_dim 1024 --freeze_txt True
    
### Results
  
  A paper discussing our implentation and results is found in `clipqa.pdf`. The `\results` directory displays training and validtion loss plots for various architectures we experimented with.
