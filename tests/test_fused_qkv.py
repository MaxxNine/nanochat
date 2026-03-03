import torch

from nanochat.gpt import GPT, GPTConfig


def _cfg(fused_qkv):
    return GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        fused_qkv=fused_qkv,
        window_pattern="L",
    )


def test_fused_qkv_matches_split_forward_backward():
    torch.manual_seed(1234)
    split = GPT(_cfg(fused_qkv=False))
    split.init_weights()

    fused = GPT(_cfg(fused_qkv=True))
    fused.load_state_dict(split.state_dict(), strict=True, assign=True)

    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    split.zero_grad(set_to_none=True)
    fused.zero_grad(set_to_none=True)

    loss_split = split(idx, targets)
    loss_fused = fused(idx, targets)
    assert torch.allclose(loss_split, loss_fused, atol=2e-5, rtol=2e-5)

    loss_split.backward()
    loss_fused.backward()

    head_dim = split.config.n_embd // split.config.n_head
    q_dim = split.config.n_head * head_dim
    kv_dim = split.config.n_kv_head * head_dim
    for i in range(split.config.n_layer):
        g_q = split.transformer.h[i].attn.c_q.weight.grad
        g_k = split.transformer.h[i].attn.c_k.weight.grad
        g_v = split.transformer.h[i].attn.c_v.weight.grad
        g_qkv = fused.transformer.h[i].attn.c_qkv.weight.grad
        g_q_fused, g_k_fused, g_v_fused = torch.split(g_qkv, [q_dim, kv_dim, kv_dim], dim=0)
        assert torch.allclose(g_q, g_q_fused, atol=3e-5, rtol=3e-5)
        assert torch.allclose(g_k, g_k_fused, atol=3e-5, rtol=3e-5)
        assert torch.allclose(g_v, g_v_fused, atol=3e-5, rtol=3e-5)


def test_fused_model_loads_split_checkpoint_strict():
    torch.manual_seed(7)
    split = GPT(_cfg(fused_qkv=False))
    split.init_weights()

    fused = GPT(_cfg(fused_qkv=True))
    result = fused.load_state_dict(split.state_dict(), strict=True, assign=True)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0

    for i in range(split.config.n_layer):
        w_q = split.transformer.h[i].attn.c_q.weight
        w_k = split.transformer.h[i].attn.c_k.weight
        w_v = split.transformer.h[i].attn.c_v.weight
        w_qkv_expected = torch.cat((w_q, w_k, w_v), dim=0)
        w_qkv = fused.transformer.h[i].attn.c_qkv.weight
        assert torch.allclose(w_qkv, w_qkv_expected, atol=0, rtol=0)


def test_split_model_loads_fused_checkpoint_strict():
    torch.manual_seed(9)
    fused = GPT(_cfg(fused_qkv=True))
    fused.init_weights()

    split = GPT(_cfg(fused_qkv=False))
    result = split.load_state_dict(fused.state_dict(), strict=True, assign=True)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0

    head_dim = split.config.n_embd // split.config.n_head
    q_dim = split.config.n_head * head_dim
    kv_dim = split.config.n_kv_head * head_dim
    for i in range(split.config.n_layer):
        w_qkv = fused.transformer.h[i].attn.c_qkv.weight
        w_q, w_k, w_v = torch.split(w_qkv, [q_dim, kv_dim, kv_dim], dim=0)
        assert torch.allclose(split.transformer.h[i].attn.c_q.weight, w_q, atol=0, rtol=0)
        assert torch.allclose(split.transformer.h[i].attn.c_k.weight, w_k, atol=0, rtol=0)
        assert torch.allclose(split.transformer.h[i].attn.c_v.weight, w_v, atol=0, rtol=0)
