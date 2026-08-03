# csrc — the `_cozy_kernels` native extension (pgw#860)

```bash
scripts/native/build_native_kernels.sh [out_dir]
```

**Shipping decision:** the kernels ship as a **prebuilt
`libcozy_kernels.so` baked into the shared cuda base image at
`/opt/cozy/native/`** (pgw#859 G0). **Never a wheel extra** — a wheel extra
reintroduces nunchaku's version-matrix failure mode (gw#405 / th#1211), where
the extension and torch must be matched by the installer and silently is not.

## Attribution

Kernel work in this tree adapts ideas and, where marked in file headers,
code from [nunchaku](https://github.com/nunchaku-ai/nunchaku) (Apache-2.0)
and references [QuTLASS](https://github.com/IST-DASLab/qutlass) (Apache-2.0)
and NVIDIA CUTLASS (BSD-3). Ported files retain upstream license headers and
state changes per Apache-2.0 §4(b).
