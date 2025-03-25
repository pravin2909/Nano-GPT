# **NanoGPT - Lightweight GPT Model for Text Generation**  

NanoGPT is a minimal yet effective implementation of a GPT-based text generation model using PyTorch. The model is trained on the Shakespeare dataset and fine-tuned for efficient text generation.  

## **Features**  
- Implements a lightweight GPT model optimized for small-scale text generation.  
- Utilizes self-attention, layer normalization, and positional encoding.  
- Supports tokenized dataset processing and batch-wise training.  
- Includes text generation with adjustable temperature and top-k sampling.  
- Implements checkpointing for resuming training from saved states.  

## **Project Structure**  
.
├── datset.py      # Defines the ShakespeareDataset class for tokenization and data loading
├── model.py       # Implements the NanoGPT transformer model
├── train.py       # Training and text generation script
├── config.py      # Stores model hyperparameters and configurations
└── README.md      # Project documentation

## **Installation**  
Ensure Python 3.8 or later is installed, then install dependencies:  

pip install torch transformers datasets tqdm tensorboard

##**Requirements**

- Python >= 3.8  
- PyTorch  
- Transformers  
- Datasets  
- TQDM  
- TensorBoard

##**License**

This project is licensed under the MIT License.
