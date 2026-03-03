import torch
import torch.nn as nn

def test_memory(use_prestaging=False):
    torch.manual_seed(42)
    device = 'cuda'
    
    # Simulate an 8-layer network with large parameter matrices
    layers = []
    for _ in range(8):
        # 4096x4096 bf16 = 32 MiB per parameter
        layers.append(nn.Linear(4096, 4096, bias=False, device=device, dtype=torch.bfloat16))
    
    model = nn.Sequential(*layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Run a forward pass to create some activation memory
    x = torch.randn(256, 4096, device=device, dtype=torch.bfloat16)
    loss = model(x).sum()
    
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**2
    
    # --- Hook Setup ---
    stacked_grads = None
    if use_prestaging:
        # Pre-allocate stacked buffer (8 layers * 32 MiB = 256 MiB)
        # Note: if we allocate this early, it increases memory early on. 
        # But we must allocate it before we can copy into it.
        params = list(model.parameters())
        stacked_grads = torch.empty((len(params), 4096, 4096), dtype=torch.bfloat16, device=device)
        
        def make_hook(idx):
            def hook(p):
                # Copy into stacked buffer
                stacked_grads[idx].copy_(p.grad)
                # Free individual grad
                p.grad = None
            return hook
            
        for i, p in enumerate(params):
            p.register_post_accumulate_grad_hook(make_hook(i))
    
    # --- Backward Pass ---
    loss.backward()
    
    bwd_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    # --- Optimizer Step ---
    torch.cuda.reset_peak_memory_stats()
    
    if not use_prestaging:
        # Standard approach: Torch.stack creates a 256 MiB copy of all 256 MiB individual grads
        stacked_grads = torch.stack([p.grad for p in model.parameters()])
        
    # Simulate Muon processing
    _ = stacked_grads * 2.0 
    
    opt_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    # Cleanup
    model.zero_grad(set_to_none=True)
    stacked_grads = None
    
    return bwd_peak - start_mem, opt_peak - start_mem


if __name__ == "__main__":
    bwd_std, opt_std = test_memory(use_prestaging=False)
    bwd_pre, opt_pre = test_memory(use_prestaging=True)
    
    print(f"--- Standard (No Hooks) ---")
    print(f"Backward Peak Memory Increase: {bwd_std:.1f} MiB")
    print(f"Optimizer Peak Memory Increase: {opt_std:.1f} MiB")
    print(f"")
    print(f"--- Muon Pre-Staging (Hooks) ---")
    print(f"Backward Peak Memory Increase: {bwd_pre:.1f} MiB")
    print(f"Optimizer Peak Memory Increase: {opt_pre:.1f} MiB")
