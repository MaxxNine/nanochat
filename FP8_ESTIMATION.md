Vamos fazer a matemática exata, colocando o seu modelo de 1.68 Bilhões de parâmetros na balança.

Para sermos precisos, vamos usar as configurações que você passou:

* **Parâmetros ($P$):** 1.68 Bilhão.
* **Elegíveis para FP8 (Matrizes Lineares):** `transformer_matrices` (864M) + `lm_head` (54.5M) $\approx$ **918.5 Milhões**.
* **Não-Elegíveis (Embeddings):** `wte` (54.5M) + `value_embeds` (709M) $\approx$ **763.5 Milhões**.


* **Lote (Batch Size - $B_{mu}$):** 16
* **Sequência ($S$):** 2048
* **Dimensão ($d$):** 1664
* **Camadas ($L$):** 26

---

### 1. Cenário A: Treinamento Padrão (Sem FP8, tudo em BF16)

**A. Memória Estática ($M_{static}$)**
Aqui, cada parâmetro usa 2 bytes (BF16). O otimizador (AdamW/Muon) guarda uma cópia do peso em alta precisão (FP32 = 4 bytes) + dois momentos (FP32 = 8 bytes), totalizando 12 bytes por parâmetro.

* **Pesos (2 bytes):** $1.68 \text{ B} \times 2 = \mathbf{3.36 \text{ GB}}$
* **Gradientes (2 bytes):** $1.68 \text{ B} \times 2 = \mathbf{3.36 \text{ GB}}$
* **Otimizador (12 bytes):** $1.68 \text{ B} \times 12 = \mathbf{20.16 \text{ GB}}$
* **Total Estático:** $3.36 + 3.36 + 20.16 = \mathbf{26.88 \text{ GB}}$
*(Atenção: Só de ligar o script, antes de passar qualquer dado, você já estourou a RTX 4090).*

**B. Memória Dinâmica ($M_{act}$)**
Sem ativação de *checkpointing* (`ckpt=off`) e com a janela deslizante `SSSL`, o fator $k_{act}$ é de aproximadamente 8. Como estamos em BF16, cada número ativado gasta 2 bytes.

* $M_{act} \approx 8 \times (B_{mu} \times S \times d \times L) \times 2 \text{ bytes}$
* $M_{act} \approx 8 \times (16 \times 2048 \times 1664 \times 26) \times 2 \text{ bytes}$
* **Total Dinâmico:** $\mathbf{\approx 22.7 \text{ GB}}$

**C. Pico de Memória (Sem FP8)**

* Pico $\approx 26.88 \text{ GB (Estático)} + 22.7 \text{ GB (Dinâmico)} = \mathbf{\approx 49.5 \text{ GB}}$
*(Perfeito para uma GPU A6000 de 48GB ou a H100 de 80GB, impossível na 4090).*

---

### 2. Cenário B: Treinamento com FP8 (Ligado via `torchao`)

**A. Memória Estática ($M_{static}$)**
Aqui é onde o mito cai. O FP8 **não** reduz o tamanho do otimizador e nem dos gradientes (que precisam de precisão para não zerar). Ele só comprime os pesos das matrizes lineares de 2 bytes para 1 byte.

* **Pesos Lineares (1 byte):** $918.5 \text{ M} \times 1 = \mathbf{0.92 \text{ GB}}$
* **Pesos Embeddings (2 bytes):** $763.5 \text{ M} \times 2 = \mathbf{1.53 \text{ GB}}$
* **Gradientes (2 bytes):** $1.68 \text{ B} \times 2 = \mathbf{3.36 \text{ GB}}$
* **Otimizador (12 bytes):** $1.68 \text{ B} \times 12 = \mathbf{20.16 \text{ GB}}$
* **Total Estático:** $0.92 + 1.53 + 3.36 + 20.16 = \mathbf{25.97 \text{ GB}}$
*(Economia de apenas ~0.9 GB na VRAM estática! Ainda dá OOM na 4090 só de carregar).*

**B. Memória Dinâmica ($M_{act}$)**
Aqui a mágica acontece. Quase todas as ativações matemáticas geradas durante o *forward pass* vêm das matrizes lineares (Q, K, V, MLP). O FP8 quantiza esses "rascunhos" salvos para 1 byte.

* $M_{act} \approx 8 \times (16 \times 2048 \times 1664 \times 26) \times \mathbf{1 \text{ byte}}$
* **Total Dinâmico:** $\mathbf{\approx 11.3 \text{ GB}}$
*(Cortou a memória variável literalmente pela metade!).*

**C. Pico de Memória (Com FP8)**

* Pico $\approx 25.97 \text{ GB (Estático)} + 11.3 \text{ GB (Dinâmico)} = \mathbf{\approx 37.2 \text{ GB}}$

---
