# Bounded sensor-gap handling

The native runtime previously latched a fault on any repeated sensor sequence,
even while the packet was fresh. Dropped sequence numbers were already allowed;
reading the same cached LowState twice was the brittle case.

The Metatron v1 launcher now uses a 60 ms repeated-state grace budget.
`--sensor-gap-grace-ms 0` restores the former strict behavior; accepted values
are 0 through 60 ms. The generic runtime default remains zero.

During a repeated but valid state:

- Do not run inference or append policy state/action history.
- Keep the last validated **position** command and gains, with no extrapolation.
- Keep the original accepted-sensor and command timestamps; duplicate packets
  cannot renew the budget.
- At startup, keep safe idle until a new sample arrives.
- Resume on fresh input within the budget without resetting the policy.

At the deadline, latch damping and require operator revival as before. This
is a bounded zero-order command hold, not balance control or a promise that a
robot cannot fall. It needs supervised hardware qualification.

The following still fault immediately: invalid/nonfinite states or commands,
backwards sequences, uninitialized sequences, backend read/write failures,
explicit emergency stop, and expired freshness/100 ms cycle limits. Torque
and velocity commands are never replayed by the grace mechanism. Operator
device loss remains a separate safety stop and is not bypassed.

Logs distinguish a temporary sensor hold, successful resumption, and a latched
fault with its reason instead of only `result=4`. The C API exposes the same
fault reason through `pm_runtime_last_fault_reason` without changing existing
enum values or callback layouts.

Verification: `deployment_contract_tests` injects repeat bursts, skipped
sequences, delayed recovery, stale/nonfinite samples, backwards sequences,
hold-write failure, unsupported command modes, and emergency stop. No robot
connection is opened by these tests.
