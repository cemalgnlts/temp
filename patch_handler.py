"""Build-time patch for worker-comfyui's /handler.py.

Adds a per-request time budget. A request may carry "max_seconds" in its input:
the time a healthy, warm worker needs for that job. When a job runs longer, the
worker is told to restart after returning the result ("refresh_worker"), so a
degraded worker does not keep serving requests. The first job of a worker's life
gets COLD_START_ALLOWANCE extra seconds (default 30) for loading the models.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/handler.py"
source = open(path).read()

START_ANCHOR = '    job_id = job["id"]\n'
END_ANCHOR = (
    '    print(f"worker-comfyui - Job completed. Returning {len(output_data)} image(s).")\n'
    "    return final_result\n"
)
assert source.count(START_ANCHOR) == 1, "start anchor not found exactly once"
assert source.count(END_ANCHOR) == 1, "end anchor not found exactly once"

source = source.replace(
    START_ANCHOR,
    START_ANCHOR + "    _job_started = time.time()\n",
)

source = source.replace(
    END_ANCHOR,
    '''    global _JOBS_DONE
    _elapsed = time.time() - _job_started
    final_result["elapsed_seconds"] = round(_elapsed, 2)
    _budget = job_input.get("max_seconds") if isinstance(job_input, dict) else None
    try:
        _budget = float(_budget) if _budget is not None else None
    except (TypeError, ValueError):
        _budget = None
    if _budget:
        if _JOBS_DONE == 0:
            _budget += float(os.environ.get("COLD_START_ALLOWANCE", "30"))
        if _elapsed > _budget:
            print(
                f"worker-comfyui - Job took {_elapsed:.1f}s, over its {_budget:.0f}s budget: "
                "restarting this worker after the response."
            )
            final_result["refresh_worker"] = True
    _JOBS_DONE += 1
'''
    + END_ANCHOR,
)

source = source.replace("\ndef handler(job):\n", "\n_JOBS_DONE = 0\n\n\ndef handler(job):\n", 1)
assert "_JOBS_DONE = 0" in source

open(path, "w").write(source)
print("handler patched: max_seconds budget enabled")
