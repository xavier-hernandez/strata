I wanted to see if I could run Strata on a 5GB card even though requirements are for a min. 12GB card.

https://github.com/Niko1221/Strata \
https://stratallm.org/

Current setup 
---------------
AMD Ryzen 5 5600GT \
64GB RAM \
1TB NVME \
NVIDIA P2200 5GB


Not fully tested but here are some tests I did run on it.
---------------
Model: Qwen3.6-35B-A3B at UD-Q4_K_XL, a 4-bit file made by Unsloth (about 23 GB). \
GPU: about memory usage 85% \
CPU: half my CPU cores are used during processing \
RAM: takes about 25GB of RAM \
produces about 18.5 tokens/sec \
Context: 4K (about 3,000 words) This is more to be used for question and answer, and not conversational. 

Tests
-------------
**Prompt: Write me a small poem**

```
Soft light spills across the floor,
The world is waking, wanting more.
A quiet breath, a distant song,
Where yesterday has drifted on.
```
**18.5 tok/sec**

**Prompt: Explain how a refrigerator works in about 200 words.**
```
A refrigerator works by continuously moving heat from its interior to the surrounding room, keeping food cold without creating "cold" out of nothing. At its core is a sealed system filled with refrigerant, a fluid that easily shifts between liquid and gas states. 

The cycle starts in the compressor, where electricity pressurizes gaseous refrigerant, drastically raising its temperature. This hot, high-pressure gas flows through external condenser coils, releasing heat into the room and condensing into a liquid. Next, the liquid passes through an expansion valve, causing a sudden drop in pressure and temperature. 

The now-cold, low-pressure refrigerant enters the internal evaporator coils. As a fan circulates warm interior air across these coils, the refrigerant absorbs that heat and evaporates back into a gas, chilling the food compartment. Sensors and a thermostat monitor the temperature, switching the compressor on and off to maintain efficiency. 

Finally, the cool gaseous refrigerant returns to the compressor, and the loop repeats. In short, a refrigerator is a heat pump: it extracts warmth from inside, transfers it outside, and sustains a consistently cool interior through continuous thermodynamic cycling.
```
**18.2 tok/sec**




**"A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?" \
"What is 47 × 38? Show your steps." \
"Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"** 

```
Here are clear, step-by-step answers to each:

**1. Bat and Ball**  
**Answer: $0.05 (5 cents)**  
*Why:* Let the ball = `x`. Then the bat = `x + $1.00`.  
`x + (x + 1.00) = 1.10` → `2x + 1.00 = 1.10` → `2x = 0.10` → `x = 0.05`.  
Check: Ball = $0.05, Bat = $1.05. Total = $1.10. ✔️

**2. 47 × 38**  
**Answer: 1,786**  
*Steps (using the distributive property):*  
`47 × 38 = 47 × (30 + 8)`  
`= (47 × 30) + (47 × 8)`  
`= 1,410 + 376`  
`= **1,786**`

**3. Sally's Sisters**  
**Answer: 1 sister**  
*Why:* The 3 brothers are siblings to each other and to Sally. If each brother has 2 sisters, those are the *same two girls* in the family (Sally + one other sister). Since the question asks how many sisters *Sally* has, she has exactly **1**.
```

**17.6 tok/sec**

<img width="1085" height="770" alt="image" src="https://github.com/user-attachments/assets/45b2cbef-a4cc-41bb-a5dd-0334aaf69745" />


```yaml
# Strata on a 5 GB Pascal card: NVIDIA Quadro P2000 / P2200 (compute capability 6.1). Needs the NVIDIA Container Toolkit on the
# host (Linux; on Unraid the Nvidia Driver plugin provides it) or Docker Desktop with WSL 2 (Windows), and an NVIDIA driver 575
# or newer. On Unraid, run it with the Docker Compose Manager plugin or `docker compose` from the terminal.
#
#   cp .env.example .env        # optional: API_KEY, STRATA_BIND, STRATA_PORT
#   docker compose -f docker-compose.nvidia-p2000.yml up -d
#   docker compose -f docker-compose.nvidia-p2000.yml logs -f   # the first start downloads the model (about 23 GB, resumable)
#
# Why these settings (see docs/OLDER_GPUS.md and docs/QWEN36.md):
# - cuda12 image: CUDA 13 cannot run Pascal; the cuda12 engine holds sm_60/sm_61 code.
# - Qwen3.6-35B-A3B UD-Q4_K_XL: its dense weights and buffers take about 4.5 GB, the smallest of the models; the experts that do not
#   fit in VRAM run on the CPU. Qwen3.8-Flash-Next needs more VRAM than this card has.
# - CONTEXT 4k: this model keeps its context in fp32, about 44 KB per token. The default 128k (about 5.9 GB) cannot fit on 5 GB.
#   If it starts with VRAM to spare, try 8k. If it stops, the log says how much is short.
# - VISION cpu: the picture reader stays off the card (set off to skip its 0.9 GB download).
# - PARALLEL 1: every extra request at once needs its own context in VRAM.
# - RAM: the experts that do not fit in VRAM are kept in RAM, so the computer needs about 32 GB.
# Expect it to be slow: most of the model runs on the CPU. Untested on a real P2000 / P2200.
services:
  strata:
    image: ${STRATA_IMAGE:-ghcr.io/niko1221/strata:0.2.7-cuda12}
    container_name: strata
    restart: unless-stopped
    ports:
      - "${STRATA_BIND:-0.0.0.0}:${STRATA_PORT:-8080}:8080"        # 0.0.0.0 = reachable at the server's IP (keep a real API key). 127.0.0.1 = this server only.
    environment:
      NVIDIA_VISIBLE_DEVICES: ${GPU_ID:-GPU-96c5c620-b710-cf18-12b5-5e148428bdcb}
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      FAMILY: ${FAMILY:-qwen36}
      MODEL: ${MODEL:-ud-q4_k_xl}
      MODEL_PATH: ${MODEL_PATH:-}            # a GGUF you already have, as seen inside the container (mount it below)
      CONTEXT: ${CONTEXT:-4k}
      PARALLEL: ${PARALLEL:-1}
      VISION: ${VISION:-cpu}                 # off | cpu
      TUNE: ${TUNE:-skip}
      # STRATA_Q8_PACKED=0: no second, packed copy of the Q8_0 dense weights. The engine uploads that copy beside every eligible
      #   matrix while it loads and gives it back only later; on 5 GB the load ran out of VRAM first (cudaMalloc at
      #   q36_engine.cu:1107). Decode reads the GGUF layout instead: the same answers, a few percent slower.
      # prefill=512: the prompt is read in 512-token chunks, so the chunk buffers stay small.
      # STRATA_Q36_NO_MMQ=1: no streamed prompt path. That path keeps a staging pair of two whole layers' experts (~1.1 GB) in VRAM
      #   and stopped the start ("1159 MiB needed, 842 MiB free"); without it the prompt's experts run on the CPU (slower prompts).
      STRATA_SET: ${STRATA_SET:-STRATA_Q8_PACKED=0 prefill=512 STRATA_Q36_NO_MMQ=1}
      API_KEY: ${API_KEY:-}                  # empty: made on the first start; docker exec strata strata launch --show-key --preset docker
    volumes:
      # the model (about 23 GB), presets, the key. A share, not a named volume: on Unraid named volumes live in docker.img,
      # which is usually too small for the model. Survives updates of the image.
      - /mnt/user/appdata/strata-p2200:/data
      # - /mnt/user/models/qwen36:/models:ro # a GGUF you already have (then also set MODEL_PATH: /models)
    # The GPU, the way Unraid's Nvidia Driver plugin passes it (the same card as --gpus "device=GPU-..."; nvidia-smi -L lists the
    # UUIDs). Compose's deploy.resources.reservations.devices left the container without a card here ("no CUDA-capable device").
    runtime: nvidia
    ulimits:
      memlock: { soft: -1, hard: -1 }        # the engine pins the RAM copy of the experts
    shm_size: 2gb
    stop_grace_period: 2m                    # the model drains and unloads on docker stop
```
