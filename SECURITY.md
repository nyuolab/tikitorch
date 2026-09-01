# Security policy

TikiTorch is research software. It processes Python code, model files, datasets,
and optional native or FPGA artifacts supplied by users; do not treat untrusted
inputs as safe merely because they pass through TikiTorch.

## Supported versions

Security fixes are applied to the latest tagged release and to `main`. Pre-release
research branches and archived experiment commits are not supported.

## Reporting a vulnerability

GitHub private vulnerability reporting is the intended channel, but it is not
enabled on the private pre-release repository as of 2026-09-01. Enabling and
verifying that channel is a publication gate. Once enabled, use it rather than
opening a public issue. Until then, do not disclose a suspected vulnerability
publicly; coordinate through an existing private channel with the repository
owner.

Include the affected revision, platform, minimal reproduction, impact, and any
suggested mitigation. The maintainers will acknowledge a complete report, assess
scope, and coordinate a fix and disclosure. This policy does not promise a
particular response time or a bug-bounty payment.

## Security boundaries

- `tt.load` and ordinary PyTorch serialization can inherit upstream deserialization
  risks. Load only artifacts you trust.
- FPGA drivers, HLS toolchains, cloud images, and board bitstreams are separate
  privileged components. Follow their vendors' security guidance.
- TCP frame decoding is resource-bounded before payload allocation: receivers
  reject ranks above 32, axis lengths above 67,108,864, payloads above 64 MiB,
  and inconsistent shapes using overflow-safe checks. These are per-frame limits
  only; the TCP transport does not provide authentication, confidentiality, peer
  authorization, aggregate connection or rate quotas, or protection against
  repeated valid frames. Split-chain listeners default to loopback; selecting a
  reachable `listen_host` is an explicit exposure decision. Expose it only to
  trusted peers or place it behind an authenticated, encrypted tunnel and
  network access controls.
- Closure auditing classifies arithmetic in a declared executed recipe. It is not
  a security sandbox, formal verifier, or proof about unexecuted code.
