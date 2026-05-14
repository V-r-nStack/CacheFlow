import tiktoken


class GPT2Tokenizer:
    """
    Tokenizer using tiktoken's GPT-2 encoding.
    
    Provides encode/decode functionality for text processing with the standard
    GPT-2 vocabulary and BPE (Byte Pair Encoding) scheme.
    """
    
    def __init__(self):
        """Initialize the GPT-2 tokenizer."""
        # Load the standard GPT-2 encoding from tiktoken
        self.encoding = tiktoken.get_encoding("gpt2")
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs
        """
        # Encode text using GPT-2 BPE encoding
        # Returns a list of integer token IDs
        token_ids = self.encoding.encode(text)
        return token_ids
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            Decoded string
        """
        # Handle both list and tensor inputs
        if hasattr(token_ids, 'tolist'):
            # Convert torch tensor to list
            token_ids = token_ids.tolist()
        
        # Decode token IDs back to text using GPT-2 encoding
        text = self.encoding.decode(token_ids)
        return text
    
    @property
    def vocab_size(self):
        """
        Get the vocabulary size of GPT-2 encoding.
        
        Returns:
            Vocabulary size (50257 for GPT-2)
        """
        return self.encoding.n_vocab
