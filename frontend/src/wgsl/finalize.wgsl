// Work-skipping, pass B: turn the work-list length into indirect dispatch args.
//
// Runs as a single thread after the compaction pass (the count is only final once
// every workgroup of that pass has finished), so the update rule can be dispatched
// with `dispatch_workgroups_indirect` and never launches a thread for a cell that
// is not going to fire.

@group(0) @binding(0) var<storage, read> count: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> indargs: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let n = atomicLoad(&count);
  indargs[0] = (n + 63u) / 64u;  // step_list.wgsl uses a 64-wide workgroup
  indargs[1] = 1u;
  indargs[2] = 1u;
}
