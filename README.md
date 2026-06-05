| File | What it does |
|------|------|
| `bogo_vk.cpp` | Vulkan compute shader (RDNA via RADV) — best for AMD consumer GPUs |
| `bogo_hip2.cpp` | HIP/ROCm kernel — for AMD Instinct (MI300X, MI355X, etc.) |
| `bogo_gpu.cpp` | OpenCL fallback |
| `bogo_bridge.py` | WebSocket bridge — connects any of the above to the server |

## Build

**Vulkan:**
```bash
sudo pacman -S vulkan-headers vulkan-radeon glslang
g++ -O2 -std=c++17 bogo_vk.cpp -o bogo_vk -lvulkan
```

**HIP (CDNA):**
```bash
# check your gfx with: rocminfo | grep gfx
hipcc -O3 -std=c++17 --offload-arch=gfx942 bogo_hip2.cpp -o bogo_hip2
```

**OpenCL:**
```bash
g++ -O2 -std=c++17 bogo_gpu.cpp -o bogo_gpu -lOpenCL
```

## Run

```bash
pip install websockets

python bogo_bridge.py \
  --uuid  "your-uuid" \
  --code  "your-code" \
  --nick  "yourname" \
  --binary ./bogo_vk \
  --work-items 262144 \
  --no-cpu
```

The optimal value depends on your GPU. Run the
benchmark script to find it:

```bash
for wg in 131072 196608 262144 327680 393216; do
  echo -n "wg=$wg: "
  echo "305419896 3735928559 2000000000 0" | \
    ./bogo_vk --daemon --work-items $wg 2>/dev/null | grep -o '"rate":[0-9]*'
done
```

