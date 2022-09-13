import torch
import torch.nn as nn


def compute_scores_for_enc_inputs(encoded_src: dict, encoded_tgt: dict, model, tokenizer, length_norm=0) -> tuple:
    """
    
    Parameters
    ----------
    encoded_src: dict 
        Tokenizer dict with with input ids and attention mask attention masks
    encoded_tgt: dict
        Tokenizer dict with input_ids and, optionally attention mask. The target
        should already account for label ignore index.
    """
    # Alternative 1. Following the code in https://github.com/neulab/BARTScore/blob/main/WMT/bart_score.py
    # We will try to get the scores associated with specific sentences
    loss_fct = nn.NLLLoss(reduction="none", ignore_index=model.config.pad_token_id)
    lsm = nn.LogSoftmax(dim=1) # first applies the softmax to ensure all values are 

    output = model(
        input_ids=encoded_src["input_ids"],
        attention_mask=encoded_src["attention_mask"],
        labels=encoded_tgt["input_ids"],
    )
    # output.logits is 3d array of size 
    #  [batch_size x max_seq_length x vocab_size] 
    #
    # logits is 2d array of size 
    # [(batch_size x max_seq_length) x vocab_size]
    logits = output.logits.view(-1, model.config.vocab_size)

    # Scale all logits first apply softmax and then logarithm
    # (preserves the size)
    lsm_logits = lsm(logits)
    # Apply cross-entropy (negative log likelihood)
    # encoded_tgt["input_ids"].view(-1) is 1-d array of size
    # (batch_size x max_seq_length)
    loss = loss_fct(lsm_logits, encoded_tgt["input_ids"].view(-1))
    # loss is 2d array of size (batch_size x max_seq_length)
    loss = loss.view(encoded_tgt["input_ids"].shape[0], -1)
    # Since logs are positive (due to - y_i log(x_i)), we revert it
    # loss = -1 * loss
    # ^Note: we commented the code for guaranteeing compatibility w/
    # original code. where compute_metrics handles the sign of the log
    # probs.

    ## Combining scores in the loss using sum of logits
    # https://github.com/neulab/BARTScore/blob/main/WMT/bart_score.py#L63
    if length_norm > 0:
        loss = loss / torch.pow(
            (encoded_tgt["input_ids"] != tokenizer.pad_token_id).sum(dim=-1),
            length_norm
        )
    
    total_loss = loss.sum(dim=1)
    return total_loss.squeeze(), loss