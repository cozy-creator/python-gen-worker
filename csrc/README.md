# csrc — the `_cozy_kernels` native extension (pgw#860)

One torch C++/CUDA extension, `TORCH_LIBRARY`-registered under
`torch.ops.cozy_kernels`. Compiled against the shared cuda base image's exact
torch pin (2.13.0+cu130); fatbin targets exactly `sm_100a` + `sm_120a`.

- Build: `scripts/native/build_native_kernels.sh [out_dir]` — compiles inside
  the pinned devel toolchain image (no host toolkit, no GPU), verifies both
  archs via `cuobjdump`, smokes op registration.
- Ship: prebuilt `libcozy_kernels.so` baked into the shared cuda base image at
  `/opt/cozy/native/` (pgw#859 G0 recommendation). Never a wheel extra —
  that reintroduces nunchaku's version-matrix failure mode (gw#405/th#1211).
- Runtime: `gen_worker.models.native_kernels` probes (load + op present +
  numerics) and degrades to the triton/baseline lane on any gap.

Currently the skeleton carries only the probe + build-info ops. Math kernels
land per the pgw#862/#863 gates; the fused svdq lane (pgw#862 B0) is pure
triton and does not need this extension.

## Attribution

Kernel work in this tree adapts ideas and, where marked in file headers,
code from [nunchaku](https://github.com/nunchaku-ai/nunchaku) (Apache-2.0)
and references [QuTLASS](https://github.com/IST-DASLab/qutlass) (Apache-2.0)
and NVIDIA CUTLASS (BSD-3). Ported files retain upstream license headers and
state changes per Apache-2.0 §4(b).
