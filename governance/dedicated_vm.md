# Dedicated VM for the agent

The agent runs on its own Mosaic AI Model Serving endpoint. The serving
endpoint provisions an isolated container with its own resource quotas,
network namespace, and lifecycle. This document records the guarantees and
the controls that enforce them.

## Guarantees

- **Isolated compute.** No other workload shares the agent's endpoint.
- **Egress controlled.** Outbound network is gated by an instance profile
  that allows only the S3 / STS endpoints the agent needs; everything else
  is denied at the VPC firewall. The default agent VM policy
  (`media.control.vm_policy` row `default-vm`) sets `egress='deny'`.
- **No persisted state.** The agent's working files are in
  `media.alex.workspaces/<user>/work/`. They survive only as long as the
  underlying volume does.
- **OBO auth.** The endpoint accepts only on-behalf-of-user tokens; its
  SP has EXECUTE on a closed set of UC functions. There is no
  `iam:PassRole` path to anything outside the agent's function surface.

## Controls

- `governance/sandbox_cluster_policy.json` — the cluster policy that any
  new ad-hoc cluster used for code execution must follow.
- `governance/permissions.sql` — least-privilege GRANTs on the agent SP.
- `governance/tools.sql` — the closed set of UC callable tools.
- `agent/workspace_tools.py:_run_on_vm` — the shell tool. Shell is
  executed via `shlex.split` + list-form `subprocess.run`; no
  `shell=True`. The command is matched against an allowlist of safe
  utilities before execution.

## What is NOT in the VM

- Long-lived credentials. The agent's SP creds are minted per request via
  workload-identity federation.
- Production data. The agent reads through UC functions only; raw Delta
  files are not in its read scope.
- Internet by default. Egress is denied at the VPC layer. The `n8n`
  integration is reachable through the corporate egress proxy only.
