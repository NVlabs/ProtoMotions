# Deployment source provenance

Selected code imported from an earlier NVIDIA deployment codebase; the
integration changes are listed below.

License: Apache-2.0. The [LICENSE](LICENSE) in this directory is NVIDIA's own
copy, not a third-party upstream notice, and the per-file notices are retained.

## Integration scope

Retained: native runtime/safety, MuJoCo and Unitree interfaces, operator input,
Metatron v1 native runtime, GPC code-only tools/runners, and focused tests.
Superseded policy adapters, combined Python launchers, tracker runtime wiring,
and their documentation/benchmarks are excluded. The pre-existing ProtoMotions
top-level tracker exporter and helpers are unchanged and remain independent.

Metatron v1 is the integrated name: filenames, CMake targets and C++ namespace
use `metatron_v1`. Serialized contract/tensor names stay unchanged for artifact
compatibility. The optional download helper takes an explicitly supplied package
URL and verifies the pinned archive checksum. Offline task
reference functions and MuJoCo model preparation were extracted from the combined
host so retained qualification tools do not depend on excluded adapters.

Other integration adjustments cover repository-relative paths, explicit JSON
and Python dependency declarations, generated-output ignores, and documentation.
No checkpoint, ONNX weights, TensorRT plan, SDK checkout, build tree or source
repository `.git` directory is included. GPC has no model downloader.

LEAPP's annotation exporter and validated artifact resolver are retained from
the LEAPP revision above. The integrated Metatron v1 bundle command preserves qualified
ONNX bytes and the native preparation tool accepts YAML; excluded upstream
launchers/exporter compatibility code are not reintroduced. The LEAPP metadata
and inference path are tested with version 0.7.2 using synthetic models.

For updates, compare freshly fetched upstream `main` with the pinned revision;
review changes selectively rather than copying the entire tree. Preserve the
ProtoMotions-owned top-level modules, the restricted runtime scope, serialized
ABI, and Metatron v1 naming. Re-run native and Python contract tests, followed
by target-specific GPU and hardware qualification where applicable.
