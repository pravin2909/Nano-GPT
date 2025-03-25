import torch
from transformers import AutoTokenizer

class GPTconfig:
    def __init__(self):
        self.vocab_size = 50257  
        self.seq_len = 128    
        self.d_model = 512      
        self.n_layers = 6      
        self.n_heads = 8  
        self.d_ff = 2048  
        self.dropout = 0.1      
        self.learning_rate = 3e-4 
        self.batch_size = 32   
        self.epochs =100     
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.top_k = 50        
        self.temperature = 1.0  
        self.max_new_tokens = 100 
        self.warmup_steps = 1000  
        self.gradient_accumulation_steps = 2  
        self.use_ppo = True    
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = GPTconfig()