Task1:


Multi-Head Self-Attention Mechanism:
Self-attention mechanism  models to weigh different words in a sequence differently, it captures dependencies and relationships between words.
Multi-headed self-attention extends the idea by using multiple attention heads, where each learns different relationships within the input.
Key Concepts: Query, Key, and Value; Scaled Dot-Product Attention; Softmax and Weighted Sum; ulti-Head Concatenation: 

Feed-Forward Networks:
After the attention mechanism, each position's representation passes through a feed-forward neural network independently.
Key Concepts: Linear Transformation; Activation Function; A Second Linear Transformation.

Positional Encoding:
Traditional transformers do not inherently capture the order of words in a sequence, so positional encoding is added to provide information about the positions of tokens.
Key Concepts: Sine and Cosine Functions; Embedding Addition.
      
Difficulties:
  Training Time:Training GPT-2 could be ime consuming on personal pc hence trained on colab.
  Memory constraints: even colab has limits hence training the model with higher number of parameters difficult. To resolve this to a degree techniques like gradient accumulation or gradient checkpointing were used.
  model size versus resource: gpt were designed and implemented by corporatesand hence could manage high gpu capacity, diffficult to manage indivisually.

sources:
  -original GPT2 paper-
      @article{radford2019language,
        title={Language Models are Few-Shot Learners},
        author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
        journal={arXiv preprint arXiv:2005.14165},
        year={2020}
      }
  -nanoGPT Repository:
      @misc{karpathy2021nanoGPT,
        title={nanoGPT},
        author={Andrej Karpathy},
        year={2021},
        howpublished={\url{https://github.com/karpathy/nanoGPT}},
      }

Task2:

  Rotary Positional Embedding: about replacing the original positional embeddings with its Rotary embeddings as proposed by Su et al. in RoFormer.
  Group Query Attention: Incorporating the Group Query Attention mechanism inspired by the work of Ainslie et al. in GQA: Training Generalized Multi-Query Transformer.
  Sliding Window Attention: Adding the Sliding Window Attention mechanism based on the insights from Beltagy et al.'s Longformer.

Implementation Details: The implementation involves modifications to the TransformerBlock class, GPT2 class, and the addition of new classes for each mechanism.

Difficulty:
    Computational Complexity: additional overhead, affecting training and inference times.
    Hyperparameter Tuning: The effectiveness of each mechanism  depends on hyperparameters, requires careful tuning.
    
sources:
  Rotary Positional Embedding:
      Su, J., Li, Y., Wu, Y., Fang, Z., & Chen, C. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
      
  Group Query Attention:
      Ainslie, J., Yang, G., & Parmar, N. (2023). GQA: Training Generalized Multi-Query Transformer.
      
  Sliding Window Attention:
      Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer.


Task3:

Task 3: Training Loop Implementation
In this task, we will implement a training loop that supports single GPU training, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) for your GPT-2 model.

Single GPU Training Loop: The training loop for a single GPU is a standard PyTorch training loop.
Distributed Data Parallel (DDP): use PyTorch's torch.nn.parallel.DistributedDataParallel to enable DDP. DDP requires initializing a torch.distributed.launch wrapper for launching the script with multiple processes.
Fully Sharded Data Parallel (FSDP):FSDP involves sharding model parameters across GPUs and reducing memory usage.

FSDP i
Copy code
# Single GPU Training
python train.py --gpu 0

# Distributed Data Parallel (2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 train.py --ddp

# Fully Sharded Data Parallel (2 GPUs)
python train.py --fsdp --world_size 2

Sources:
PyTorch DDP Tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
FairScale Documentation: https://fairscale.readthedocs.io/en/latest/


Difficulties:

Single GPU Training:mStraightforward and widely used. Memory limitations might be a concern for larger models.
Distributed Data Parallel (DDP): It Requires careful setup using torch.distributed.launch. Synchronization and communication overhead.
Fully Sharded Data Parallel (FSDP):Reduced memory requirements but may introduce additional complexity.FairScale library provides useful abstractions but requires installation.





