import torch
import argparse
from pathlib import Path

from model.gpt import GPT
from utils.tokenizer import GPT2Tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    device='cpu'
):
    """
    Autoregressive text generation using greedy sampling.
    
    Args:
        model: GPT model instance
        tokenizer: GPT2Tokenizer instance
        prompt: Initial text prompt as string
        max_tokens: Maximum number of tokens to generate
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Generated text string
    """
    
    # ===== ENCODE PROMPT =====
    
    # Tokenize the input prompt
    # prompt: str -> token_ids: List[int]
    prompt_token_ids = tokenizer.encode(prompt)
    
    # Convert to torch tensor and move to device
    # List[int] -> Tensor of shape (1, prompt_len)
    sequence = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Prompt: {prompt}")
    print(f"Encoded to {len(prompt_token_ids)} tokens")
    print(f"Generating up to {max_tokens} tokens...\n")
    
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    # ===== AUTOREGRESSIVE GENERATION LOOP =====
    
    with torch.no_grad():
        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Get current sequence length
            # sequence: (1, current_seq_len)
            current_seq_len = sequence.shape[1]
            
            # Forward pass through the model
            # (1, current_seq_len) -> (1, current_seq_len, vocab_size)
            logits = model(sequence)
            
            # Extract logits for the last token position
            # (1, current_seq_len, vocab_size) -> (1, vocab_size)
            last_token_logits = logits[:, -1, :]
            
            # Greedy sampling: take the token with highest probability
            # argmax over vocabulary dimension (dim=-1)
            # (1, vocab_size) -> (1, 1)
            next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            
            # Append the generated token to the sequence
            # (1, current_seq_len) + (1, 1) -> (1, current_seq_len + 1)
            sequence = torch.cat([sequence, next_token_id], dim=1)
            
            # Check stopping condition: End-of-text token ID is 50256 in GPT-2
            if next_token_id.item() == 50256:
                print("Generated [END] token")
                break
    
    # ===== DECODE OUTPUT =====
    
    # Convert tensor back to list of token IDs
    # (1, final_seq_len) -> List[int]
    generated_token_ids = sequence[0].tolist()
    
    # Decode all tokens to text
    # List[int] -> str
    generated_text = tokenizer.decode(generated_token_ids)
    
    return generated_text


def main():
    """Main execution entrypoint for text generation."""
    
    # ===== PARSE COMMAND-LINE ARGUMENTS =====
    
    parser = argparse.ArgumentParser(
        description="Generate text using GPT model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of AI is",
        help="Initial text prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/gpt2_124m_state_dict.pt",
        help="Path to model weights state dict"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # ===== INITIALIZE MODEL =====
    
    # GPT-2 small configuration
    vocab_size = 50257  # Standard GPT-2 vocabulary size
    max_seq_len = 1024  # Maximum sequence length
    dim = 768           # Model/embedding dimension
    num_heads = 12      # Number of attention heads
    num_layers = 12     # Number of transformer blocks
    
    print(f"Initializing GPT model with {num_layers} layers, {num_heads} heads, {dim} dim")
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0
    )
    
    # ===== LOAD PRETRAINED WEIGHTS =====
    
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"Loading pretrained weights from {args.model_path}")
        try:
            GPT.load_pretrained_weights(model, args.model_path)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"Warning: Model weights file not found at {args.model_path}")
        print("Using randomly initialized weights")
    
    # Move model to device
    model.to(args.device)
    
    # ===== INITIALIZE TOKENIZER =====
    
    print("Initializing GPT-2 tokenizer")
    tokenizer = GPT2Tokenizer()
    
    # ===== RUN GENERATION =====
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        device=args.device
    )
    
    # ===== OUTPUT RESULTS =====
    
    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)


if __name__ == "__main__":
    main()
