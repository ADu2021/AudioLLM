# Reference: RAD-TTS: Parallel flow-based tts with robust alignment learning and diverse synthesis. (Shih et al., 2021)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
            attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
                            batched tensor of attention log
                            probabilities, padded to length
                            of longest sequence in each dimension
            text_lens: batch-D vector of length of
                        each text sequence
            mel_lens: batch-D vector of length of
                        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
        # construct the target sequence. Every
        # text token is mapped to a unique sequence number,
        # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_log_prob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total
    
if __name__ == "__main__":
    from forced_aligner_inference import viterbi_decode
    loss_fct = ForwardSumLoss()

    DIAGONAL = True

    text_lens = 20
    mel_lens = 500
    batch_size = 64
    list_log_prob = []
    for i in range(batch_size):
        attn_weight = torch.rand(mel_lens, text_lens)
        if DIAGONAL:
            attn_weight = torch.zeros(mel_lens, text_lens)
            for i in range(mel_lens):
                attn_weight[i][min(i//(mel_lens//text_lens), text_lens-1)] = 10
        log_prob = F.log_softmax(attn_weight, dim=1)
        log_prob[-1, :] = -1
        log_prob[:, -1] = -1
        print(log_prob)
        list_log_prob.append(log_prob)
    attn_logprob = torch.stack(list_log_prob, dim=0).unsqueeze(1) # (bs, 1, sq_mel, sq_text)
    
    loss = loss_fct(attn_logprob, [text_lens]*batch_size, [mel_lens]*batch_size)
    print(loss)

    import ipdb; ipdb.set_trace();

    decode_output = viterbi_decode(attn_logprob.squeeze(), [text_lens]*batch_size, [mel_lens]*batch_size)