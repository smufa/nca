"""The engine's bind groups must match the bindings the WGSL declares.

`ShaderRunner` builds its pipelines with `layout: 'auto'`, so the *index* of an entry
in `entries: [...]` is the binding number: get the order wrong and the shader reads
the work list where it expects the state buffer. Nothing type-checks that -- the
arrays are `GPUBindGroupEntry[]` either way -- and the failure shows up in the
browser as a validation error or, worse, a plausible-looking wrong picture.

This walks both sides: for every `@binding(n)` in the shader (which must be
contiguous from 0) there is exactly one entry, in that order.

No TS parser: the arrays in `shader.ts` are plain literals, so brace matching is
enough. If the engine grows a third pipeline, add it to `PIPELINES`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER_TS = ROOT / "frontend" / "src" / "shader.ts"
WGSL = ROOT / "frontend" / "src" / "wgsl"

# (wgsl file, the local bind-group builder/s in shader.ts, pipeline entry point)
PIPELINES = [
    ("pass1.wgsl", ["bg1"], "step / step_deep"),
    ("pass2.wgsl", ["bg2"], "main"),
    ("compact.wgsl", ["bgCompact"], "main"),
    ("finalize.wgsl", None, "main"),
    ("step_list.wgsl", ["bgStepList"], "main"),
]


def wgsl_binding_count(name: str) -> int:
    src = (WGSL / name).read_text()
    bindings = sorted({int(m) for m in re.findall(r"@binding\((\d+)\)", src)})
    assert bindings == list(range(len(bindings))), f"{name}: non-contiguous bindings {bindings}"
    return len(bindings)


def entries_block(source: str, builder: str, after: int = 0) -> str:
    """The `entries: [...]` literal of the first bind-group creation after `builder`."""
    m = re.search(rf"(?:const|let)\s+{re.escape(builder)}\s*=", source[after:])
    assert m, f"no `{builder}` bind-group builder in shader.ts"
    start = source.index("entries: [", after + m.end())
    open_ = source.index("[", start)
    depth, i = 0, open_
    while i < len(source):
        if source[i] == "[":
            depth += 1
        elif source[i] == "]":
            depth -= 1
            if depth == 0:
                return source[open_ : i + 1]
        i += 1
    raise AssertionError(f"unbalanced entries literal for {builder}")


def binding_order(block: str) -> list[int]:
    return [int(n) for n in re.findall(r"binding:\s*(\d+)", block)]


def test_every_shader_binding_has_an_entry_in_order() -> None:
    source = SHADER_TS.read_text()
    for name, builders, entry in PIPELINES:
        count = wgsl_binding_count(name)
        if builders is None:
            # `finalize` binds inside `allocGrid` without a named builder; the
            # pipeline still needs the same number of entries as it declares.
            assert "pipelineFinalize" in source, "finalize pipeline is never created"
            continue
        for builder in builders:
            block = entries_block(source, builder)
            order = binding_order(block)
            assert order == list(range(count)), (
                f"{name} ({entry}) declares {count} bindings 0..{count - 1}, "
                f"but {builder} lists {order}"
            )


def test_work_skip_passes_bind_the_buffers_the_shader_reads() -> None:
    """The three work-skip passes share buffers; check the roles line up."""
    source = SHADER_TS.read_text()
    compact = entries_block(source, "bgCompact")
    step_list = entries_block(source, "bgStepList")
    # compact: state, fire, out(tmp), list, count, params
    assert ["state", "fireBuf", "tmpBuf", "listBuf", "countBuf", "paramsBuf"] == re.findall(
        r"buffer:\s*(?:this\.)?(\w+)", compact
    ), compact
    # step_list: state, list, weights, kernel, out(tmp), params, w4, cond, count
    assert [
        "state",
        "listBuf",
        "weightBuf",
        "kernelBuf",
        "tmpBuf",
        "paramsBuf",
        "weight4Buf",
        "condBuf",
        "countBuf",
    ] == re.findall(r"buffer:\s*(?:this\.)?(\w+)", step_list), step_list


def test_indirect_dispatch_buffer_is_declared_indirect() -> None:
    source = SHADER_TS.read_text()
    assert "dispatchWorkgroupsIndirect" in source, "the work-skip path never dispatches indirectly"
    m = re.search(r"this\.indirectBuf = d\.createBuffer\(\{(.*?)\}\);", source, re.S)
    assert m, "no indirect buffer is created"
    assert "INDIRECT" in m.group(1), "the indirect args buffer needs GPUBufferUsage.INDIRECT"
